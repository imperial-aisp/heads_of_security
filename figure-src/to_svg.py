# Convert the compiled figure into the SVG the page shows (needs PyMuPDF: pip install pymupdf).
# Run from this folder after `pdflatex results_overview.tex`.
import re
import pymupdf

svg = pymupdf.open('results_overview.pdf')[0].get_svg_image(text_as_path=True)
svg = re.sub(r'width="([\d.]+)" height="([\d.]+)"', r'width="\1pt" height="\2pt"', svg, count=1)  # sizes in pt, like the other figures
with open('../website/src/assets/figures/results_overview.svg', 'w') as f:
    f.write(svg)
