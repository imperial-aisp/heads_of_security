# Headline figure source

`results_overview.tex` builds the Utility + Security figure at the top of the Results section. It is the
paper's teaser with the same preamble, plot styles, data and legend, except:

- no method card (the page shows the method as an interactive diagram);
- the Utility card sits above the Security card;
- Utility starts with an ARC-C (0-shot) panel, from the paper's Table 2;
- the legend is scaled down uniformly (to about 9pt from the paper's 11.8pt) so its one line fits the cards' width;

If you change the teaser's data or styles, copy the change into `results_overview.tex` too.

## Rebuild

The site build can't run LaTeX, so the SVG is committed (`../website/src/assets/figures/results_overview.svg`).
After editing the tex, regenerate it from this folder:

```bash
pdflatex results_overview.tex   # needs the standalone, pgfplots, roboto and sfmath packages
python to_svg.py                # needs PyMuPDF (pip install pymupdf); writes the SVG
```
