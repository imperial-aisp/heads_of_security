import figure from '../../assets/figures/gate_heatmap_gptoss20b.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 2 (right) as exported from the paper: Δ per layer × head for GPT-OSS-20B.
// `height` px tall, unless the column is narrower than the figure: then it scales down with its aspect ratio.
const RATIO = 252.89813 / 306.48063; // the SVG's viewBox

export default function GateHeatmap({ height = 420 }: { height?: number }) {
  const w = +(Math.max(100, height) * RATIO).toFixed(1);
  return <PaperFigure src={figure} alt="Learned attention gates for GPT-OSS-20B: attention shift per layer and head" width={`min(${w}px, 100%)`} />;
}
