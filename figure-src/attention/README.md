# Attention data for "Where the next token looks"

`extract_attention.py` computes the real attention shown in the page's attention widget: for the
five scenarios of the demo Space (`scenarios.json`, copied from the Space), it runs
GPT-OSS-20B with the HoS gates on and off, generates each arm's step (greedy, as in the Space) and
records how much the first generated token, the query, attends to each prompt token in every head.

Outputs:

- `../../website/src/assets/attention/gates_delta.json` — each head's gate shift Δ (paper Figure 2) and the 50
  reddest and 50 bluest heads.
- `../../website/src/assets/attention/<id>.json` — tokens, channels, message segments; per arm the mean attention
  over all 1,536 heads, over the red heads and over the blue heads, their channel shares (including the
  attention sink), and the generated step. Imported by the page at build time.
- `../../website/public/attention/<id>.heads.bin` — the full attention row and channel shares of the 50 reddest
  and 50 bluest heads, for the "One head" view; fetched only when that view is opened. The layout is
  documented in `write_heads()` in the script.
- `--full-out` (the job writes `~/hos-attention-full/`) — the complete [layers, heads, tokens] arrays,
  for other cuts of the data without another GPU run.

## Rebuild

From the repo root, with the repo's environment set up (`uv sync`) and a GPU:

```bash
sbatch figure-src/attention/run.slurm     # on the cluster, or:
.venv/bin/python figure-src/attention/extract_attention.py --out website/src/assets/attention
```

To change the scenarios, edit `scenarios.json` (the Space's file) or `--n`.
