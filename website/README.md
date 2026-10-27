# Heads of Security — project page

Static project page for the paper *Heads of Security: Learned Attention Gates Improve Prompt Injection
Robustness*, built with [Astro](https://astro.build) and React islands.

## Develop

Requires Node.js 22.12 or later.

```bash
cd website
npm install
npm run dev       # http://localhost:4321/heads_of_security/
npm run check     # type-check .astro / .ts / .tsx
npm run build     # static site in dist/
npm run preview   # serve dist/
```

## Deploy

`.github/workflows/pages.yml` builds this folder and deploys it to GitHub Pages on every push to `main`
or `website` that touches `website/` (or manually via *Run workflow*). One-time setup in the repository: **Settings →
Pages → Source: GitHub Actions**. The site is served at <https://imperial-aisp.github.io/heads_of_security/>;
with a custom domain the workflow builds for the root path automatically.

## Where things are

| Path | Contents |
| --- | --- |
| `src/config.ts` | Links, arXiv id, BibTeX, code snippets, notebook URL |
| `src/pages/` | `index.astro`, the main page, and `attention.astro`, the gate analysis around the before-and-after attention widget, linked from the main page's "Inside the model" section and its section rail |
| `public/hos-header.webp`, `public/favicon.png` | The meme, shown at the end of the Cite section for those who read that far, and the dog cut from it for the icons and link previews |
| `src/components/sections/` | One Astro component per page section (static HTML) |
| `src/components/figures/` | Figures. `GateDiagram` and `AttentionCompare` are hydrated islands; `ResultsOverview` (the headline figure), `TradeoffGrid`, `MultiTurnChart`, `RoleProbeChart` and `GateHeatmap` show the paper's own figures as images |
| `src/assets/figures/` | The paper's figures as SVG (exported from the PDFs). To update one, replace its file |
| `../figure-src/` (repo root) | LaTeX source of the headline figure (the paper's teaser re-laid-out for the page) and how to rebuild its SVG |
| `../figure-src/attention/` (repo root) | Script and SLURM job that compute the attention widget's data on a GPU; outputs go to `src/assets/attention/` (imported) and `public/attention/` (fetched on demand) |
| `src/components/islands/` | Small interactive pieces: collapsible sections, copy buttons |
| `src/components/ds/` | Design-system components (ported from `design-system-components/`) |
| `src/components/Logo.astro`, `public/brands/` | The arXiv, GitHub, Hugging Face and Google Colab marks on the resource buttons: arXiv's logomark (the three paths of the primary logo served by arxiv.org, unaltered; see [arXiv's brand guidelines](https://info.arxiv.org/brand/brand-guidelines.html)), GitHub's black Invertocat from [its logo package](https://brand.github.com/foundations/logo), and the other two from [Dashboard Icons](https://dashboardicons.com). Trademarks of their owners |
| `src/styles/` | Design tokens and component classes (ported unchanged), plus `site.css` for page layout. The text column's width (`--page-max`) and the text sizes (`--type-*`) are set at the top of `site.css`, with smaller sizes for phones at the end. The sticky bar at the top holds the resource buttons; from 1360px wide the section links float left of the column (the `--rail`, `--shift` block) |

Only the interactive parts ship JavaScript.

## Before launch

1. **Attention data** (`src/assets/attention/`, `public/attention/`) is real: GPT-OSS-20B with the gates off and
   on, for the demo Space's first three scenarios, computed by `../figure-src/attention/extract_attention.py` on a
   GPU (see its README). Rerun it if the gates, the scenarios or the Space's prompt construction change.
2. **arXiv id** is the placeholder `XXXX.XXXXX` (`ARXIV_ID` in `src/config.ts`; used by the Paper links and
   the BibTeX entry).
