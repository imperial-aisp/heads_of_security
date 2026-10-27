import figure from '../../assets/figures/claudini_cumulative_asr.svg';
import { PaperFigure } from './PaperFigure';

// Paper Figure 2 (left), cut from the paper: cumulative ASR of the Claudini attack against attack compute.
// Sized so that its legend, about 8pt in the figure's 274pt width, is 2px larger than the body text.
const LEGEND_TO_WIDTH = 34;

export default function ClaudiniChart() {
  return (
    <PaperFigure
      src={figure}
      width={`calc((var(--type-body) + 2px) * ${LEGEND_TO_WIDTH})`}
      alt="Cumulative attack success rate of the Claudini attack against attack compute on AgentDojo and InjecAgent, for GPT-OSS-20B, w/ SecAlign and w/ Heads of Security"
    />
  );
}
