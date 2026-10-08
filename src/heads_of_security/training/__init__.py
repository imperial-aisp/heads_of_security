"""Gate training: a trimmed torchtune 0.6.1 full-DPO recipe with the base model frozen.

    uv sync --extra train
    uv run tune run --nproc_per_node 1 src/heads_of_security/training/full_dpo_distributed.py \
        --config configs/training/gpt_oss_20b.yaml output_dir=runs/gptoss20b_lr0.1

See ``configs/training/`` and the README's training section. The modules here are only
imported through torchtune's ``_component_`` config mechanism, so importing the package
itself does not require torchtune.
"""
