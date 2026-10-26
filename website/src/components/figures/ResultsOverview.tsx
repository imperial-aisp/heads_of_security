import figure from '../../assets/figures/results_overview.svg';
import { PaperFigure } from './PaperFigure';

// Headline figure, compiled from the repo's figure-src/results_overview.tex: the paper's teaser (teaser_v6.tex) with the
// Utility card stacked above the Security card and an ARC-C panel from paper Table 2.

export default function ResultsOverview() {
  return (
    <PaperFigure
      src={figure}
      alt="Utility (ARC-C, MMLU-Pro, GSM8K, AgentDojo) and security (AgentDojo static attack and Claudini, InjecAgent RL-Hammer and Claudini) of GPT-OSS-20B, w/ SecAlign, w/ ASIDE and w/ Heads of Security"
    />
  );
}
