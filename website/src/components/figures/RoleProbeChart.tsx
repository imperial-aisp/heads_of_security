import figure from '../../assets/figures/role_confusion_layer_sweep.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 5 (Appendix B) as exported from the paper. Sized so that its legend, 9pt in the figure's 586pt
// width, is 2px larger than the body text (and never wider than the column), like the other paper figures.
const LEGEND_TO_WIDTH = 65.1;

export default function RoleProbeChart() {
  return (
    <PaperFigure
      src={figure}
      width={`calc((var(--type-body) + 2px) * ${LEGEND_TO_WIDTH})`}
      alt="Role-probe accuracy and injection-token userness across layers for GPT-OSS-20B and w/ Heads of Security"
    />
  );
}
