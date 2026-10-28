import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { Button } from '../ds/Button';
import { Icon } from '../ds/Icon';
import { SegmentedControl } from '../ds/SegmentedControl';
import { Tag } from '../ds/Tag';
import type { TagTone } from '../ds/Tag';
import banking from '../../assets/attention/banking-u13-i7.json';
import slack from '../../assets/attention/slack-u7-i1.json';
import workspace from '../../assets/attention/workspace-u8-i1.json';
import slackWebpage from '../../assets/attention/slack-u0-i5.json';
import workspaceFeedback from '../../assets/attention/workspace-u28-i2.json';
import gates from '../../assets/attention/gates_delta.json';

// Real attention from GPT-OSS-20B with the HoS gates off (base) and on, for the demo Space's five
// scenarios, computed by the repo's figure-src/attention/extract_attention.py: the attention paid by the first token
// the model generates (the query, shown after the prompt) to every prompt token and to itself. The JSON files
// carry the mean over all 24 x 64 heads, over the red heads (gate shift delta > 0) and over the blue heads
// (delta < 0), plus the step each arm generated. The full rows of the 50 reddest and 50 bluest heads are
// fetched from public/attention/ when the "One head" view is opened.

type Channel = 'trusted' | 'untrusted' | 'template';
type Arm = 'base' | 'hos';
type Color = 'red' | 'blue';
type Mode = 'all' | 'red' | 'blue' | 'single';
type Shares = Record<Channel | 'sink', number>;
interface Segment { start: number; body: number; end: number; header: string }
interface Fold { start: number; end: number; label: string }
interface Generation { reasoning: string | null; content: string | null; tool_calls: string[]; verdict: 'attacker' | 'user' | 'other' | 'response' | 'none'; finished: boolean; new_tokens: number }
interface Query { token: string; id: number; channel: number }
interface ArmData { query: Query; mean: number[]; shares: Shares; mean_red: number[]; shares_red: Shares; mean_blue: number[]; shares_blue: Shares; generation: Generation }
interface Sample { id: string; title: string; n_tokens: number; tokens: string[]; channels: number[]; segments: Segment[]; folds: Fold[]; arms: Record<Arm, ArmData> }
interface Gates { delta: number[][]; red: [number, number, number][]; blue: [number, number, number][] }

const SAMPLES = [banking, slack, workspace, slackWebpage, workspaceFeedback] as unknown as Sample[];
const LABELS: Record<string, string> = {
  'banking-u13-i7': 'Banking · update my address',
  'slack-u7-i1': 'Slack · add Charlie to a channel',
  'workspace-u8-i1': 'Workspace · add meeting participants',
  'slack-u0-i5': 'Slack · read a webpage',
  'workspace-u28-i2': 'Workspace · look up feedback scores',
};
const GATES = gates as unknown as Gates;
const N_HEADS = GATES.delta.length * GATES.delta[0].length;
const N_RED = GATES.delta.flat().filter((d) => d > 0).length;
const N_BLUE = GATES.delta.flat().filter((d) => d < 0).length;
const TOP: Record<Color, [number, number, number][]> = { red: GATES.red, blue: GATES.blue }; // [layer, head, delta], 1-based
const ARMS: Arm[] = ['base', 'hos'];
const CHANNELS: Channel[] = ['trusted', 'untrusted', 'template'];
const SHADE: Record<Channel, string> = { trusted: 'var(--fig-bracket-trusted)', untrusted: 'var(--fig-bracket-untrusted)', template: 'var(--fig-token-system)' };
const LABEL: Record<Channel, string> = { trusted: 'var(--fig-text-trusted)', untrusted: 'var(--fig-text-untrusted)', template: 'var(--fig-text-system)' };
const SINK = 'var(--color-divider-strong)';
const HEAD_RGB: Record<Color, string> = { red: '180,4,38', blue: '59,76,192' }; // --heads-red, --heads-blue
const HEAD_INK: Record<Color, string> = { red: 'var(--heads-red)', blue: 'var(--heads-blue)' };
// `response` is a text answer with no tool call; in the Space's scenarios that answer completes the user's task
// (the webpage's content, the feedback scores), so it is labelled like a user-task tool call.
const VERDICT: Record<Generation['verdict'], [string, TagTone]> = {
  attacker: ['Follows the injection', 'untrusted'], user: ['Works on the user’s task', 'trusted'], other: ['Calls another tool', 'neutral'],
  response: ['Works on the user’s task', 'trusted'], none: ['No action', 'neutral'],
};
const fmtDelta = (d: number) => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(2);

// ---- per-head data: <id>.heads.bin (version 2), written by write_heads() in extract_attention.py ----
interface HeadArm { vmax: Float32Array; shares: Uint8Array; q: Uint8Array }
interface HeadFile { K: number; N: number; heads: [number, number][]; arms: Record<Arm, HeadArm> }
const headFiles = new Map<string, Promise<HeadFile>>();

function parseHeads(buf: ArrayBuffer): HeadFile {
  const dv = new DataView(buf);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'HOSA' || dv.getUint16(4, true) !== 2) throw new Error('unexpected file format');
  const K = dv.getUint16(6, true), N = dv.getUint32(12, true), nArms = dv.getUint16(16, true);
  let off = 18;
  const pairs = new Uint8Array(buf, off, 2 * K); off += 2 * K;
  const heads = Array.from({ length: K }, (_, i) => [pairs[2 * i] + 1, pairs[2 * i + 1] + 1] as [number, number]);
  const arms = {} as Record<Arm, HeadArm>;
  ARMS.slice(0, nArms).forEach((arm) => {
    const vmax = new Float32Array(buf.slice(off, off + 4 * K)); off += 4 * K; // copied: the offset is not 4-byte aligned
    const shares = new Uint8Array(buf, off, 4 * K); off += 4 * K;
    const q = new Uint8Array(buf, off, K * N); off += K * N;
    arms[arm] = { vmax, shares, q };
  });
  return { K, N, heads, arms };
}

function loadHeads(id: string): Promise<HeadFile> {
  let p = headFiles.get(id);
  if (!p) {
    p = fetch(`${import.meta.env.BASE_URL.replace(/\/?$/, '/')}attention/${id}.heads.bin`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); })
      .then(parseHeads);
    headFiles.set(id, p);
  }
  return p;
}

/** The full attention row and channel shares of curated head number `r` (red heads first, then blue). */
function headRow(file: HeadFile, arm: Arm, r: number): { att: Float32Array; shares: Shares } {
  const a = file.arms[arm];
  const att = new Float32Array(file.N);
  const vmax = a.vmax[r];
  for (let t = 0; t < file.N; t++) { const v = a.q[r * file.N + t] / 255; att[t] = vmax * v * v; }
  const s = [0, 1, 2, 3].map((c) => a.shares[r * 4 + c] / 255);
  return { att, shares: { trusted: s[0], untrusted: s[1], template: s[2], sink: s[3] } };
}

// ---- rendering ----
const fs = (k: number) => `calc(var(--type-body) * ${k})`;
const caps = { fontSize: fs(0.74), letterSpacing: '.08em', textTransform: 'uppercase', fontWeight: 500 } as const;
const fieldCol = { display: 'flex', flexDirection: 'column', gap: 6 } as const;
const shareBar = { borderRadius: 1, transition: 'flex 320ms cubic-bezier(.2,.7,.2,1)' } as const;
// inside a panel every text is the panel's size (16px): tokens, chips, labels and tags alike; the uppercase
// labels (TRUSTED, NEXT ACTION, ...) are set smaller, as capitals at 16px read larger than mixed-case text
const PANEL_FONT = 16;
const capsIn = { ...caps, fontSize: 13 } as const;
const mono = { fontFamily: 'var(--font-mono)', fontSize: 'inherit' } as const;
const pct = (v: number, d = 0) => (v * 100).toFixed(d) + '%';
const isSpecial = (t: string) => /^<\|[a-z_]+\|>$/.test(t);
const Swatch = ({ color }: { color: Color }) => <span aria-hidden="true" style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 3, background: `rgb(${HEAD_RGB[color]})`, marginRight: 8, verticalAlign: -1 }} />;

const SCEN_OPTIONS = SAMPLES.map((s) => ({ value: s.id, label: LABELS[s.id] ?? s.title.split(' (')[0] }));
const MODE_OPTIONS = [
  { value: 'single', label: 'One head' },
  { value: 'all', label: `Mean of all ${N_HEADS.toLocaleString('en')} heads` },
  { value: 'red', label: <span><Swatch color="red" />Mean of {N_RED} red heads</span> },
  { value: 'blue', label: <span><Swatch color="blue" />Mean of {N_BLUE} blue heads</span> },
];

interface View { att: ArrayLike<number>; shares: Shares }

function Strip({ color, selected, onPick }: { color: Color; selected: number | null; onPick: (i: number) => void }) {
  const heads = TOP[color];
  const mags = heads.map((h) => Math.abs(h[2]));
  const lo = Math.min(...mags), hi = Math.max(...mags);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
        <span style={{ ...caps, color: HEAD_INK[color] }}>
          {color} heads <span style={{ color: 'var(--color-text-muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>· the {heads.length} whose gates close most on {color === 'red' ? 'untrusted' : 'trusted'} tokens</span>
        </span>
        <span className="num" style={{ fontSize: fs(0.74), color: 'var(--color-text-muted)' }}>Δ from {fmtDelta(heads[0][2])} to {fmtDelta(heads[heads.length - 1][2])}</span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {heads.map(([layer, head, delta], i) => {
          const a = 0.22 + 0.78 * (hi > lo ? (Math.abs(delta) - lo) / (hi - lo) : 1);
          return (
            <button
              key={i} type="button" onClick={() => onPick(i)} title={`Layer ${layer}, head ${head} · Δ ${fmtDelta(delta)}`} aria-label={`Layer ${layer}, head ${head}, Δ ${fmtDelta(delta)}`} aria-pressed={selected === i}
              style={{ width: 20, height: 20, border: 0, padding: 0, borderRadius: 3, cursor: 'pointer', background: `rgba(${HEAD_RGB[color]},${a.toFixed(2)})`, outline: selected === i ? '2px solid var(--color-text)' : 'none', outlineOffset: 2 }}
            />
          );
        })}
      </div>
    </div>
  );
}

function Panel({ s, arm, view, vmax, name, nameColor, stroke, viewLabel }: { s: Sample; arm: Arm; view: View; vmax: number; name: string; nameColor: string; stroke: string; viewLabel: string }) {
  const query = s.arms[arm].query; // the first generated token sits at index n_tokens
  const text = (i: number) => (i < s.n_tokens ? s.tokens[i] : query.token);
  const chan = (i: number) => CHANNELS[i < s.n_tokens ? s.channels[i] : query.channel];
  const bg = (i: number) => `color-mix(in srgb, ${SHADE[chan(i)]} ${Math.round(Math.sqrt(Math.min(1, view.att[i] / vmax)) * 100)}%, transparent)`;
  const tip = (i: number) => `${text(i).trim() || '␣'} · ${(view.att[i] * 100).toFixed(2)}%`;
  const tok = (i: number, extra?: CSSProperties, t = text(i)) => (
    <span key={i} title={tip(i)} style={{ background: bg(i), borderRadius: 2, ...extra }}>{t}</span>
  );
  const foldAt = new Map(s.folds.map((f) => {
    let start = f.start;
    while (start > 0 && /^\s+$/.test(s.tokens[start - 1]) && !s.segments.some((g) => g.body === start)) start--;
    return [start, { ...f, start }];
  }));
  const g = s.arms[arm].generation;
  const [verdict, tone] = VERDICT[g.verdict];
  const action = g.tool_calls[0] ?? g.content ?? (g.finished ? '(no output)' : `(still reasoning after ${g.new_tokens} tokens)`);
  const last = s.segments[s.segments.length - 1];

  return (
    <div className="att-panel" style={{ flex: '1 1 400px', minWidth: 0, border: `1px solid ${stroke}`, borderRadius: 'var(--radius-md)', display: 'flex', flexDirection: 'column', fontSize: PANEL_FONT }}>
      <div style={{ padding: '18px 22px 16px', borderBottom: '1px solid var(--color-divider)', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 500, lineHeight: 1.2, color: nameColor }}>{name}</span>
          <span style={{ color: 'var(--color-text-muted)' }}>{viewLabel}</span>
        </div>
        <span style={{ ...capsIn, display: 'flex', alignItems: 'center', gap: 7, color: 'var(--color-text-muted)' }}>
          <span className="tag tag-outline" style={{ padding: '0 7px', textTransform: 'none', letterSpacing: '0.02em' }}>query</span>
          <span>attention, by channel</span>
        </span>
        <div style={{ display: 'flex', gap: 2, height: 8 }}>
          {CHANNELS.map((c) => <div key={c} style={{ ...shareBar, flex: view.shares[c].toFixed(4), background: SHADE[c] }} />)}
          <div style={{ ...shareBar, flex: view.shares.sink.toFixed(4), background: SINK }} />
        </div>
        <div className="num" style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 18px' }}>
          {CHANNELS.map((c) => <span key={c} style={{ color: LABEL[c] }}>{c[0].toUpperCase() + c.slice(1)} {pct(view.shares[c])}</span>)}
          <span style={{ color: 'var(--color-text-muted)' }} title="The attention sink: the share of the row that goes to no token.">Sink {pct(view.shares.sink)}</span>
        </div>
      </div>

      <div style={{ padding: '18px 22px', display: 'flex', flexDirection: 'column', gap: 16, flex: 1 }}>
        {s.segments.slice(0, -1).map((seg) => {
          const body: ReactNode[] = [];
          for (let i = seg.body; i < seg.end; i++) {
            const f = foldAt.get(i);
            if (f) {
              body.push(
                <div key={'f' + i} style={{ margin: '4px 0' }}>
                  <span className="tag tag-neutral" title={`${f.end - f.start} tokens of ${f.label}, folded`}>
                    ⋯ {f.label} · {(f.end - f.start).toLocaleString('en')} tokens
                  </span>
                </div>,
              );
              i = f.end - 1;
            } else {
              // the token right before a fold loses its trailing line breaks, so the chip follows without a gap
              const t = foldAt.has(i + 1) ? s.tokens[i].replace(/\s+$/, '') : s.tokens[i];
              body.push(tok(i, isSpecial(s.tokens[i]) ? { ...mono, padding: '0 3px' } : undefined, t));
            }
          }
          // the message is labelled with the channel of its visible tokens; a body folded whole (system and
          // developer messages) with the channel of its content, not of the template tokens folded with it
          const count = [0, 0, 0];
          for (let i = seg.body; i < seg.end; i++) { const f = foldAt.get(i); if (f) { i = f.end - 1; continue; } count[s.channels[i]]++; }
          if (count[0] + count[1] + count[2] === 0) for (let i = seg.body; i < seg.end; i++) if (s.channels[i] !== 2) count[s.channels[i]]++;
          if (count[0] + count[1] + count[2] === 0) for (let i = seg.body; i < seg.end; i++) count[s.channels[i]]++;
          const bodyChan = CHANNELS[count.indexOf(Math.max(...count))];
          return (
            <div key={seg.start} style={{ display: 'flex', flexDirection: 'column', gap: 6, paddingLeft: 12, borderLeft: `3px solid ${SHADE[bodyChan]}` }}>
              <div style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', lineHeight: 1.6 }}>
                {/* the label is glued to the header's last token, so it always ends the header's last line */}
                <span style={mono}>{Array.from({ length: seg.body - seg.start - 1 }, (_, k) => tok(seg.start + k))}</span>
                <span style={{ whiteSpace: 'nowrap' }}>
                  <span style={mono}>{tok(seg.body - 1)}</span>
                  <span style={{ ...capsIn, color: LABEL[bodyChan], marginLeft: 10 }}>{bodyChan}</span>
                </span>
              </div>
              <div style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', lineHeight: 1.6, color: 'var(--fig-ink)' }}>{body}</div>
            </div>
          );
        })}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', paddingLeft: 12, borderLeft: `3px solid ${SHADE[chan(s.n_tokens)]}` }}>
          <span style={mono}>{Array.from({ length: last.end - last.start }, (_, k) => tok(last.start + k))}{tok(s.n_tokens, { padding: '0 3px' })}</span>
          <Tag tone="outline" style={{ fontSize: 13 }}>query (first generated token)</Tag>
        </div>
      </div>

      <div style={{ borderTop: '1px solid var(--color-divider)', padding: '16px 22px 20px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ ...capsIn, color: 'var(--color-text-muted)' }}>Next action</span>
          <Tag tone={tone} style={{ fontSize: 13 }}>{verdict}</Tag>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', lineHeight: 1.5, overflowWrap: 'anywhere' }}>{action}</div>
      </div>
    </div>
  );
}

export default function AttentionCompare() {
  const [id, setId] = useState(SAMPLES[0].id);
  const [mode, setMode] = useState<Mode>('single'); // one head first: the per-head view is the point of the widget
  const [color, setColor] = useState<Color>('red');
  const [index, setIndex] = useState(0);
  const [heads, setHeads] = useState<{ id: string; file: HeadFile } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const s = SAMPLES.find((x) => x.id === id)!;
  const single = mode === 'single';
  const ready = single && heads?.id === id;
  useEffect(() => {
    if (!single || heads?.id === id) return;
    let live = true;
    setError(null);
    loadHeads(id).then((file) => { if (live) setHeads({ id, file }); }, (e) => { if (live) setError(String(e)); });
    return () => { live = false; };
  }, [single, id, heads]);

  const [layer, head, delta] = TOP[color][index];
  const row = color === 'red' ? index : TOP.red.length + index;
  const views = useMemo((): Record<Arm, View> => {
    if (ready) return { base: headRow(heads!.file, 'base', row), hos: headRow(heads!.file, 'hos', row) };
    const pick = (a: ArmData): View =>
      mode === 'red' ? { att: a.mean_red, shares: a.shares_red } : mode === 'blue' ? { att: a.mean_blue, shares: a.shares_blue } : { att: a.mean, shares: a.shares };
    return { base: pick(s.arms.base), hos: pick(s.arms.hos) };
  }, [ready, heads, row, mode, s]);
  // Shading scale: the strongest content token across both panels. The chat template's own tokens
  // (<|start|>, assistant, <|end|>) take several times more attention than any word and are capped.
  const vmax = useMemo(() => {
    let m = 1e-9;
    for (const a of ARMS) { const att = views[a].att; for (let i = 0; i < s.n_tokens; i++) if (s.channels[i] !== 2 && !isSpecial(s.tokens[i]) && att[i] > m) m = att[i]; }
    return m;
  }, [views, s]);

  const viewLabel = ready ? `layer ${layer}, head ${head}` : mode === 'red' ? `mean of the ${N_RED} red heads` : mode === 'blue' ? `mean of the ${N_BLUE} blue heads` : `mean of all ${N_HEADS.toLocaleString('en')} heads`;
  const whose = ready ? 'one head’s' : mode === 'red' ? `the mean of the ${N_RED} red heads’` : mode === 'blue' ? `the mean of the ${N_BLUE} blue heads’` : `the mean of all ${N_HEADS.toLocaleString('en')} heads’`;
  const viewSentence = `The panels show ${whose} attention to every token of the prompt and to the query itself.`;
  const step = (d: number) => setIndex((index + d + TOP[color].length) % TOP[color].length);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, fontFamily: 'var(--font-body)', fontSize: 'var(--type-body)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px 32px', alignItems: 'flex-end' }}>
        <div className="seg-stack-phone" style={fieldCol}>
          <span className="field-label" style={{ margin: 0 }}>Sample</span>
          <SegmentedControl ariaLabel="Sample" value={id} onChange={setId} options={SCEN_OPTIONS} />
        </div>
        <div className="seg-stack-phone" style={fieldCol}>
          <span className="field-label" style={{ margin: 0 }}>Heads</span>
          <SegmentedControl ariaLabel="Heads" value={mode} onChange={(v) => setMode(v as Mode)} options={MODE_OPTIONS} />
        </div>
      </div>

      {single ? (
        <div style={{ border: '1px solid var(--color-divider)', borderRadius: 'var(--radius-md)', padding: '22px 24px 20px', display: 'flex', flexDirection: 'column', gap: 18 }}>
          <Strip color="red" selected={color === 'red' ? index : null} onPick={(i) => { setColor('red'); setIndex(i); }} />
          <Strip color="blue" selected={color === 'blue' ? index : null} onPick={(i) => { setColor('blue'); setIndex(i); }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap', borderTop: '1px solid var(--color-divider)', paddingTop: 16 }}>
            <span className="num" style={{ fontSize: fs(0.86) }}>
              <span style={{ fontWeight: 500, color: 'var(--color-text)' }}>Layer {layer}, head {head}</span> · Δ = {fmtDelta(delta)} ·{' '}
              <span style={{ color: HEAD_INK[color] }}>{color} head: closes on {color === 'red' ? 'untrusted' : 'trusted'} tokens</span>
              {error ? <span style={{ color: 'var(--color-text-muted)' }}> · the per-head data could not be loaded ({error})</span> : !ready ? <span style={{ color: 'var(--color-text-muted)' }}> · loading the per-head data…</span> : null}
            </span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, whiteSpace: 'nowrap' }}>
              <Button variant="secondary" onClick={() => step(-1)}>← Previous</Button>
              <Button variant="secondary" onClick={() => step(1)}>Next →</Button>
              <Button variant="ghost" onClick={() => { const c: Color = Math.random() < 0.5 ? 'red' : 'blue'; setColor(c); setIndex(Math.floor(Math.random() * TOP[c].length)); }}>
                <Icon name="shuffle" size={15} />Shuffle
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      <p style={{ margin: 0, lineHeight: 'var(--type-body-lh)', textAlign: 'justify', hyphens: 'auto' }}>
        Each panel shows how the <span className="tag tag-outline" style={{ padding: '0px 7px', verticalAlign: 1 }}>query</span> distributes its attention
        over the prompt. The <span className="tag tag-outline" style={{ padding: '0px 7px', verticalAlign: 1 }}>query</span> is the first token the model generates, shown after the
        prompt’s <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.86em' }}>{'<|start|>assistant'}</code>. {viewSentence} Hover any token to see the share of the query’s attention it receives.
      </p>

      <div className="wide-figure">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'stretch' }}>
          <Panel s={s} arm="base" view={views.base} vmax={vmax} name="GPT-OSS-20B" nameColor="var(--color-text)" stroke="var(--color-divider)" viewLabel={viewLabel} />
          <Panel s={s} arm="hos" view={views.hos} vmax={vmax} name="w/ Heads of Security" nameColor="var(--fig-series-hos-label)" stroke="var(--color-trusted)" viewLabel={viewLabel} />
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px 24px', alignItems: 'center', fontSize: 16, color: 'var(--color-text-muted)' }}>
          <span aria-hidden="true" />
          <span>Darker shading = more attention</span>
          {CHANNELS.map((ch) => (
            <span key={ch} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span style={{ width: '0.9em', height: '0.9em', borderRadius: 2, background: SHADE[ch] }} />
              {ch}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
