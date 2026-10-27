import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { FigurePanel } from '../ds/FigurePanel';
import { SegmentedControl } from '../ds/SegmentedControl';

// Method figure, laid out as the paper's Figure 1 without the query's arrows to the tokens: tokens, attention
// scores, "×" and the gate multipliers, the gate, gated scores, the softmax, attention weights; one illustrative
// head with hand-picked scores. Every length is in em of the diagram's font, which is DIAGRAM_CQI percent of
// the panel's width (between 13px and the body text size + 2px) so that the fixed parts fit with bars of
// BAR_DEFAULT em; the three bar columns share whatever width is left, so the diagram always fills the panel.
// When the panel is too narrow for 13px it scrolls. Bars are proportional to their values, with a small floor
// so labelled bars stay visible; a value label sits inside its bar when it fits and just right of it otherwise.
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
const LABEL_FONT = 'calc(var(--type-body) + 2px)'; // the channel labels and column headers: 2px above the body text
const HEAD_H = `calc(${LABEL_FONT} * 2.4)`; // column headers: two lines at line-height 1.2
const COL_GAP = 1; // gap between columns
const TOKEN_GAP = 1.6; // extra gap after the tokens, so they read apart from the scores
const TIMES_W = 1; // the "×" between a score and its multiplier
const MULT_W = 2.6; // the gate multipliers
const ARROW_W = 1.25; // the lines into the two pills and the arrows out of them, as in the paper
const PILL_W = 2.2; // the "Apply learned HoS gates" and "softmax" pills
const BAR_DEFAULT = 8; // bar column width before the panel is measured, and at the panel width the font scales with
const BAR_MIN = 5; // narrower than this the panel scrolls instead
const DIAGRAM_CQI = 1.75; // font size as a percentage of the panel width: fixed parts (about 33em) + three bars of BAR_DEFAULT
const LABEL_GAP = 1.4; // between a channel label's bar and the pair's wider token (about a row, as in the paper)
const LABEL_RIGHT_DEFAULT = 3.9; // the label's offset from the tokens' right edge before it is measured
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

const colHead = { height: HEAD_H, fontSize: LABEL_FONT, display: 'flex', alignItems: 'flex-end', lineHeight: 1.2, color: 'var(--fig-ink-muted)', whiteSpace: 'nowrap' } as const;
const belowHead = `calc(${HEAD_H} + ${em(GAP)})`; // top padding of the columns without a header
const column = { display: 'flex', flexDirection: 'column', gap: em(GAP) } as const;
const barColumn = { ...column, flex: '1 1 0', minWidth: em(BAR_MIN) } as const;
const pill = { width: em(PILL_W), margin: `calc(${HEAD_H} + ${em(GAP - 0.3)}) 0 -${em(0.3)}`, background: 'var(--fig-pill)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', writingMode: 'vertical-rl', transform: 'rotate(180deg)', color: 'var(--fig-ink-muted)', whiteSpace: 'nowrap' } as const;
const mono = { fontFamily: 'var(--fig-font-mono)', fontSize: em(TEXT), whiteSpace: 'nowrap' } as const;

export default function GateDiagram() {
  const [mode, setMode] = useState<Mode>('hos');
  const on = mode === 'hos';
  const cur = on ? ON : OFF;
  const other = on ? OFF : ON;

  // Measured once rendered, and again whenever the panel resizes: the bar columns' width in em (they grow with
  // the panel), and how far left of the tokens' right edge each channel label sits, so that its bar is tucked
  // in next to the pair's wider token, as in the paper.
  const scoresCol = useRef<HTMLDivElement>(null);
  const tokenCol = useRef<HTMLDivElement>(null);
  const tokenEls = useRef<Record<string, HTMLElement | null>>({});
  const [barW, setBarW] = useState(BAR_DEFAULT);
  const [labelRight, setLabelRight] = useState<Record<'trusted' | 'untrusted', number>>({ trusted: LABEL_RIGHT_DEFAULT, untrusted: LABEL_RIGHT_DEFAULT });
  const [overhang, setOverhang] = useState(0); // px by which the labels reach left of the longest token; padded so they stay inside the panel
  useEffect(() => {
    const scores = scoresCol.current, tokens = tokenCol.current;
    if (!scores || !tokens) return;
    const measure = () => {
      const f = parseFloat(getComputedStyle(scores).fontSize);
      setBarW(Math.max(BAR_MIN, scores.getBoundingClientRect().width / f));
      const right = tokens.getBoundingClientRect().right;
      const extent = (names: string[]) => Math.max(...names.map((n) => right - (tokenEls.current[n]?.getBoundingClientRect().left ?? right))) / f;
      const offsets = { trusted: extent(['Count', 'R’s']) + LABEL_GAP, untrusted: extent(['Send', 'money']) + LABEL_GAP };
      setLabelRight(offsets);
      const tokensW = tokens.clientWidth - (parseFloat(getComputedStyle(tokens).paddingLeft) || 0);
      const reach = Math.max(...[...tokens.querySelectorAll<HTMLElement>('[data-label]')].map((l) => l.getBoundingClientRect().width + offsets[l.dataset.label as 'trusted' | 'untrusted'] * f));
      setOverhang(Math.max(0, Math.ceil(reach - tokensW)));
    };
    const ro = new ResizeObserver(measure);
    ro.observe(scores);
    measure();
    return () => ro.disconnect();
  }, []);

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
        <div style={{ display: 'flex', gap: em(COL_GAP), width: '100%', padding: `${em(0.4)} 0 ${em(0.3)}`, fontSize: `max(13px, min(calc(var(--type-body) + 2px), ${DIAGRAM_CQI}cqi))`, fontFamily: 'var(--fig-font)', color: 'var(--fig-ink)' }}>
          {/* the tokens, right-aligned; the channel labels sit over the column, their bars next to the short tokens */}
          <div ref={tokenCol} style={{ ...column, position: 'relative', alignItems: 'flex-end', paddingTop: belowHead, paddingLeft: overhang, marginRight: em(TOKEN_GAP) }}>
            {rows.map((r) => (
              <div key={r.tok} style={{ height: em(ROW), display: 'flex', alignItems: 'center' }}>
                <code ref={(el) => { tokenEls.current[r.tok] = el; }} className={'token token-fig token-' + r.kind} style={{ fontSize: em(TEXT) }}>{r.tok}</code>
              </div>
            ))}
            <ChannelLabel kind="trusted" text={['Trusted', 'tokens']} style={{ position: 'absolute', top: `calc(${HEAD_H} + ${em(GAP + ROW + GAP)})`, right: em(labelRight.trusted) }} />
            <ChannelLabel kind="untrusted" text={['Untrusted', 'tokens']} style={{ position: 'absolute', top: `calc(${HEAD_H} + ${em(GAP + 4 * (ROW + GAP))})`, right: em(labelRight.untrusted) }} />
          </div>

          <div ref={scoresCol} style={barColumn}>
            <div style={colHead}>Attention<br />scores</div>
            {rows.map((r) => <Bar key={r.tok} kind={r.kind} value={r.score} max={10} width={barW} label={r.scoreLabel} />)}
          </div>

          <div style={{ ...column, width: em(TIMES_W), color: 'var(--fig-ink-muted)' }} aria-hidden="true">
            <div style={colHead} />
            {rows.map((r) => <div key={r.tok} style={{ height: em(ROW), display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span style={{ fontSize: em(1.15) }}>×</span></div>)}
          </div>

          <div style={{ ...column, width: em(MULT_W) }}>
            <div style={{ ...colHead, justifyContent: 'center', textAlign: 'center' }}>{on ? <>× HoS<br />gate</> : <>× no<br />gate</>}</div>
            {rows.map((r) => (
              <div key={r.tok} style={{ height: em(ROW), display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className={r.multCls} style={{ fontSize: em(TEXT), transition: 'color 120ms ease' }}>{r.mult}</span>
              </div>
            ))}
          </div>

          <Flow>{on ? 'Apply learned HoS gates' : 'No gates applied'}</Flow>

          <div style={barColumn}>
            <div style={colHead}>Gated<br />scores</div>
            {rows.map((r) => <Bar key={r.tok} kind={r.kind} value={r.gated} max={10} width={barW} label={r.gatedLabel} />)}
          </div>

          <Flow>softmax</Flow>

          <div style={barColumn}>
            <div style={colHead}>Attention<br />weights</div>
            {rows.map((r) => <Bar key={r.tok} kind={r.kind} value={r.weight} max={1} width={barW} label={r.weightLabel} ghost={r.ghost} />)}
          </div>
        </div>
      </FigurePanel>
    </div>
  );
}

// A token-coloured bar, `value / max` of the column's width (with a floor so it stays visible), with its
// value label inside when it fits and to its right otherwise; `ghost` outlines the other mode's value.
function Bar({ kind, value, max, width, label, ghost = null }: { kind: Channel; value: number; max: number; width: number; label: string; ghost?: number | null }) {
  const bar = value > 0 ? Math.max((value / max) * width, label ? LABEL_FLOOR : STUB) : 0;
  const inside = !!label && PAD_L + label.length * CHAR + PAD_R <= bar + 1e-6;
  const ghostW = ghost !== null ? Math.max((ghost / max) * width, 0.3) : 0;
  return (
    <div style={{ position: 'relative', width: '100%', height: em(ROW), display: 'flex', alignItems: 'center', gap: em(0.3) }}>
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

// A pill with its flow lines, as in the paper's figure: on every row a plain line runs into the pill and an
// arrow leads out of it, both touching the pill.
function Flow({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'stretch' }}>
      <Lines head={false} />
      <div style={pill}>{children}</div>
      <Lines head />
    </div>
  );
}

function Lines({ head }: { head: boolean }) {
  return (
    <div style={{ ...column, width: em(ARROW_W), paddingTop: belowHead, color: 'var(--fig-ink-muted)' }} aria-hidden="true">
      {ROWS.map((r) => (
        <div key={r.tok} style={{ height: em(ROW), display: 'flex', alignItems: 'center' }}>
          <svg viewBox="0 0 24 10" style={{ display: 'block', width: em(ARROW_W), height: em(ARROW_W * 10 / 24) }}>
            <path d={head ? 'M0 5h22M18.5 1.5L22 5l-3.5 3.5' : 'M0 5h24'} fill="none" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      ))}
    </div>
  );
}

function ChannelLabel({ kind, text, style }: { kind: 'trusted' | 'untrusted'; text: [string, string]; style?: CSSProperties }) {
  return (
    <div data-label={kind} style={{ height: em(2 * ROW + GAP), display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: em(0.9), lineHeight: 1.15, textAlign: 'right', whiteSpace: 'nowrap', color: `var(--fig-text-${kind})`, ...style }}>
      <span style={{ fontSize: LABEL_FONT }}>{text[0]}<br />{text[1]}</span>
      <i style={{ width: em(0.3), alignSelf: 'stretch', borderRadius: 1, background: `var(--fig-bracket-${kind})` }} />
    </div>
  );
}
