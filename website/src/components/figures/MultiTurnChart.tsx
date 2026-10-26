import figure from '../../assets/figures/multi_turn_alpacafarm_utility_by_input_turn.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 3 as exported from the paper.
export default function MultiTurnChart() {
  return <PaperFigure src={figure} alt="Per-turn instruction-completion rate for Llama-3.3-70B, w/ SecAlign and w/ Heads of Security" />;
}
