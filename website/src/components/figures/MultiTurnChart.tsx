import figure from '../../assets/figures/multi_turn_alpacafarm_utility_by_input_turn.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 3 as exported from the paper. Sized so that its legend, 11pt in the figure's 556pt width, is
// 2px larger than the body text (and never wider than the column), like the headline figure.
const LEGEND_TO_WIDTH = 50.5;

export default function MultiTurnChart() {
  return (
    <PaperFigure
      src={figure}
      width={`calc((var(--type-body) + 2px) * ${LEGEND_TO_WIDTH})`}
      alt="Per-turn instruction-completion rate for Llama-3.3-70B, w/ SecAlign and w/ Heads of Security"
    />
  );
}
