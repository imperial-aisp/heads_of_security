# Headline figure source

`results_overview.tex` builds the Utility + Security figure at the top of the Results section. It is the
paper's teaser with the same preamble, plot styles, data and legend, except:

- no method card (the page shows the method as an interactive diagram);
- the Utility and Security cards sit side by side at a common height, each at its own width;
- the legend is one line above both cards at 11.8pt, the size of the card names (the page sizes the figure so
  that this text is 2px larger than the body text).

If you change the teaser's data or styles, copy the change into `results_overview.tex` too.

## Rebuild

The site build can't run LaTeX, so the SVG is committed (`../website/src/assets/figures/results_overview.svg`).
After editing the tex, regenerate it from this folder:

```bash
pdflatex results_overview.tex   # needs the standalone, pgfplots, roboto and sfmath packages
python to_svg.py                # needs PyMuPDF (pip install pymupdf); writes the SVG
```
