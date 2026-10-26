import figure from '../../assets/figures/role_confusion_layer_sweep.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 5 (Appendix B) as exported from the paper.
export default function RoleProbeChart() {
  return <PaperFigure src={figure} alt="Role-probe accuracy and injection-token userness across layers for GPT-OSS-20B and w/ Heads of Security" />;
}
