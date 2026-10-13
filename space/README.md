---
title: Heads of Security
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: gpt-oss-20b with attention gates against prompt injection
models:
  - openai/gpt-oss-20b
  - imperial-cpg/hos-gpt-oss-20b
---

# Heads of Security

Demo of [Heads of Security](https://github.com/imperial-aisp/heads_of_security): learned per-head
attention gates that help gpt-oss-20b use tool results without following instructions injected into them.
The demo runs the gated model (`imperial-cpg/hos-gpt-oss-20b`, lr 0.1) and the base model on the same prompt.
