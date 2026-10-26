import figure from '../../assets/figures/tradeoff_all_linear.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 4 as exported from the paper.
export default function TradeoffGrid() {
  return <PaperFigure src={figure} alt="Security–utility trade-off for GPT-OSS-20B on AgentDojo, AgentDyn, LLMail and WASP" />;
}
