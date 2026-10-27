import figure from '../../assets/figures/results_overview.svg';
import { PaperFigure } from './PaperFigure';

// Headline figure, compiled from the repo's figure-src/results_overview.tex: the paper's teaser (teaser_v6.tex) without
// the method card, the Utility and Security cards side by side under one legend line.
// Sized so that the legend and the "Utility" and "Security" labels are 2px larger than the body text: they are
// set at 11.8pt in the figure's 615pt width, so the figure is 52.2 times the wanted label size wide, and never
// wider than the column (on a 1440px screen the column caps it and the labels come out at about 18.7px).
const LABEL_TO_WIDTH = 52.2;

export default function ResultsOverview() {
  return (
    <PaperFigure
      width={`calc((var(--type-body) + 2px) * ${LABEL_TO_WIDTH})`}
      src={figure}
      alt="Utility (MMLU-Pro, GSM8K, AgentDojo) and security (AgentDojo static attack and Claudini, InjecAgent RL-Hammer and Claudini) of GPT-OSS-20B, w/ SecAlign, w/ ASIDE and w/ Heads of Security"
    />
  );
}
