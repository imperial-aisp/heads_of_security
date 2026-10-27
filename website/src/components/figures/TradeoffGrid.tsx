import figure from '../../assets/figures/tradeoff_all_linear.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 4 as exported from the paper. Sized so that its legend, 11pt in the figure's 522pt width, is
// 2px larger than the body text (and never wider than the column), like the headline figure.
const LEGEND_TO_WIDTH = 47.4;

export default function TradeoffGrid() {
  return (
    <PaperFigure
      src={figure}
      width={`calc((var(--type-body) + 2px) * ${LEGEND_TO_WIDTH})`}
      alt="Security–utility trade-off for GPT-OSS-20B on AgentDojo, AgentDyn, LLMail and WASP"
    />
  );
}
