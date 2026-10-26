import { useState } from 'react';
import { FigurePanel } from '../ds/FigurePanel';
import { SegmentedControl } from '../ds/SegmentedControl';

// Method figure: one illustrative head with hand-picked scores. Every length is in em of the diagram's
// font, which is 1.75% of the panel's width so the diagram (about 56em wide) fills the panel, between
// 13px and the body text size + 2px; when the panel is too narrow for 13px it scrolls. Bars are
// proportional to their values, with a small floor so labelled bars stay visible; a value label sits
// inside its bar when it fits and just right of it otherwise.
type Channel = 'system' | 'trusted' | 'untrusted';
const ROWS: { tok: string; kind: Channel; score: number }[] = [
  { tok: '<|start|> user', kind: 'system', score: 1 },
  { tok: 'Count', kind: 'trusted', score: 10 },
  { tok: 'R’s', kind: 'trusted', score: 5 },
  { tok: '<|start|> tool', kind: 'system', score: 1 },
  { tok: 'Send', kind: 'untrusted', score: 10 },
  { tok: 'money', kind: 'untrusted', score: 5 },
  { tok: '<|start|> assistant', kind: 'system', score: 1 },
];
const GATES: Record<Channel, number> = { system: 0.9, trusted: 1, untrusted: 0.1 };

// geometry, in em of the diagram's font
const ROW = 1.5; // row height
const GAP = 0.6; // gap between rows
const HEAD = 2.5; // column headers: two lines
const COL_GAP = 1.2; // gap between columns
const SCORE_W = 8; // score bar width (= a score of 10)
const WEIGHT_W = 9; // weight bar width (= a weight of 1)
const TEXT = 0.9; // font size of tokens, bar values and gate multipliers
const CHAR = 0.6 * TEXT; // advance of one Roboto Mono character at that size
const PAD_L = 0.2; // inset of a value label inside its bar
const PAD_R = 0.12;
const LABEL_FLOOR = PAD_L + CHAR + PAD_R; // a labelled bar is at least one character wide
const STUB = 0.45; // an unlabelled (near-zero) bar is still visible
const em = (v: number) => +v.toFixed(3) + 'em';

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100));

function run(gated: boolean) {
  const scores = ROWS.map((r) => r.score * (gated ? GATES[r.kind] : 1));
  const ex = scores.map((g) => Math.exp(g));
  const sum = ex.reduce((a, b) => a + b, 0);
  return { scores, weights: ex.map((x) => x / sum) };
}
const OFF = run(false);
const ON = run(true);

type Mode = 'base' | 'hos';
const MODE_OPTIONS = [
  { value: 'base', label: 'No gates' },
  { value: 'hos', label: 'w/ Heads of Security' },
];

const colHead = { height: em(HEAD), display: 'flex', alignItems: 'flex-end', lineHeight: 1.2, color: 'var(--fig-ink-muted)', whiteSpace: 'nowrap' } as const;
const column = { display: 'flex', flexDirection: 'column', gap: em(GAP) } as const;
const pill = 'var(--fig-pill)';
const mono = { fontFamily: 'var(--fig-font-mono)', fontSize: em(TEXT), whiteSpace: 'nowrap' } as const;

export default function GateDiagram() {
  const [mode, setMode] = useState<Mode>('hos');
  const on = mode === 'hos';
  const cur = on ? ON : OFF;
  const other = on ? OFF : ON;

  const rows = ROWS.map((r, i) => ({
    ...r,
    scoreLabel: fmt(r.score),
    mult: on ? fmt(GATES[r.kind]) + 'x' : '1x',
    multCls: 'fig-mult ' + (on ? 'fig-mult-' + r.kind : 'fig-mult-off'),
    gated: cur.scores[i],
    gatedLabel: cur.scores[i] >= 0.9 ? fmt(cur.scores[i]) : '',
    weight: cur.weights[i],
    weightLabel: cur.weights[i] >= 0.005 ? cur.weights[i].toFixed(2) : '',
    ghost: Math.abs(other.weights[i] - cur.weights[i]) > 0.05 ? other.weights[i] : null,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, paddingBottom: 24, fontFamily: 'var(--font-body)', fontSize: 'var(--type-body)', color: 'var(--color-text)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '10px 20px' }}>
        <SegmentedControl ariaLabel="Gates" value={mode} onChange={(v) => setMode(v as Mode)} options={MODE_OPTIONS} />
        <span style={{ fontSize: 'var(--type-caption)', color: 'var(--color-text-muted)' }}>
          {on ? 'Dashed outlines: attention weights without HoS gates.' : 'Dashed outlines: attention weights with HoS gates.'}
        </span>
      </div>

      <FigurePanel tone="method" style={{ overflowX: 'auto', containerType: 'inline-size' }}>
        <div style={{ display: 'flex', gap: em(COL_GAP), width: 'max-content', margin: '0 auto', padding: `${em(0.4)} 0 ${em(0.3)}`, fontSize: 'max(13px, min(calc(var(--type-body) + 2px), 1.75cqi))', fontFamily: 'var(--fig-font)', color: 'var(--fig-ink)' }}>
          <div style={{ ...column, width: em(5.6), paddingTop: em(HEAD + GAP) }}>
            <div style={{ height: em(ROW) }} />
            <ChannelLabel kind="trusted" text={['Trusted', 'tokens']} />
            <div style={{ height: em(ROW) }} />
            <ChannelLabel kind="untrusted" text={['Untrusted', 'tokens']} />
          </div>

          <div style={{ ...column, alignItems: 'flex-end', paddingTop: em(HEAD + GAP) }}>
            {rows.map((r) => (
              <div key={r.tok} style={{ height: em(ROW), display: 'flex', alignItems: 'center' }}>
                <code className={'token token-fig token-' + r.kind} style={{ fontSize: em(TEXT) }}>{r.tok}</code>
              </div>
            ))}
          </div>

          <div style={{ ...column, width: em(SCORE_W) }}>
            <div style={colHead}>Attention<br />scores</div>
            {rows.map((r) => <Bar key={r.tok} kind={r.kind} value={r.score} max={10} width={SCORE_W} label={r.scoreLabel} />)}
          </div>

          <div style={{ ...column, width: em(4.8) }}>
            <div style={{ ...colHead, justifyContent: 'center', textAlign: 'center' }}>{on ? <>× HoS<br />gate</> : <>× no<br />gate</>}</div>
            <div style={{ flex: 1, margin: `-${em(0.3)} 0`, padding: `${em(0.3)} 0`, background: pill, borderRadius: 8, display: 'flex', flexDirection: 'column', gap: em(GAP), alignItems: 'center' }}>
              {rows.map((r) => (
                <div key={r.tok} style={{ height: em(ROW), display: 'flex', alignItems: 'center' }}>
                  <span className={r.multCls} style={{ fontSize: em(TEXT), transition: 'color 120ms ease' }}>{r.mult}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ ...column, width: em(SCORE_W) }}>
            <div style={colHead}>Gated<br />scores</div>
            {rows.map((r) => <Bar key={r.tok} kind={r.kind} value={r.gated} max={10} width={SCORE_W} label={r.gatedLabel} />)}
          </div>

          <div style={{ width: em(1.8), margin: `${em(HEAD + GAP - 0.3)} 0 -${em(0.3)}`, background: pill, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', writingMode: 'vertical-rl', transform: 'rotate(180deg)', color: 'var(--fig-ink-muted)' }}>softmax</div>

          <div style={{ ...column, width: em(WEIGHT_W) }}>
            <div style={colHead}>Attention<br />weights</div>
            {rows.map((r) => <Bar key={r.tok} kind={r.kind} value={r.weight} max={1} width={WEIGHT_W} label={r.weightLabel} ghost={r.ghost} />)}
          </div>
        </div>
      </FigurePanel>
    </div>
  );
}

// A token-coloured bar, `value / max` of the row's width (with a floor so it stays visible), with its
// value label inside when it fits and to its right otherwise; `ghost` outlines the other mode's value.
function Bar({ kind, value, max, width, label, ghost = null }: { kind: Channel; value: number; max: number; width: number; label: string; ghost?: number | null }) {
  const bar = value > 0 ? Math.max((value / max) * width, label ? LABEL_FLOOR : STUB) : 0;
  const inside = !!label && PAD_L + label.length * CHAR + PAD_R <= bar + 1e-6;
  const ghostW = ghost !== null ? Math.max((ghost / max) * width, 0.3) : 0;
  return (
    <div style={{ position: 'relative', width: em(width), height: em(ROW), display: 'flex', alignItems: 'center', gap: em(0.3) }}>
      <span
        className={'token token-fig token-' + kind}
        style={{ ...mono, flex: 'none', width: em(bar / TEXT), height: '90%', boxSizing: 'border-box', padding: `0 0 0 ${em(PAD_L / TEXT)}`, overflow: 'visible', transition: 'width 320ms cubic-bezier(.2,.7,.2,1)' }}
      >
        {inside ? label : ''}
      </span>
      {label && !inside ? <span style={mono}>{label}</span> : null}
      {ghost !== null ? (
        <span style={{ position: 'absolute', left: 0, top: '5%', height: '90%', width: em(ghostW), boxSizing: 'border-box', border: '1px dashed var(--fig-ink-muted)', borderRadius: 'var(--radius-md)', pointerEvents: 'none' }} />
      ) : null}
    </div>
  );
}

function ChannelLabel({ kind, text }: { kind: 'trusted' | 'untrusted'; text: [string, string] }) {
  return (
    <div style={{ height: em(2 * ROW + GAP), display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: em(0.5), lineHeight: 1.15, textAlign: 'right', color: `var(--fig-text-${kind})` }}>
      <span>{text[0]}<br />{text[1]}</span>
      <i style={{ width: em(0.3), alignSelf: 'stretch', borderRadius: 1, background: `var(--fig-bracket-${kind})` }} />
    </div>
  );
}
