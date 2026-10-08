# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Derived from torchtune 0.6.1 ``full_dpo_distributed.py``, reduced to what gate training
# needs: an HF gated model with ``token_channels`` threaded through the forward, the base
# frozen and only the gate tensors trained, the *same* model with ``token_channels=None`` as
# the DPO reference policy (the gated attention is a bit-exact no-op without channels, so
# the reference is the base model and no second copy is needed), and gate-only checkpoints.
#
#     tune run --nproc_per_node 8 src/heads_of_security/training/full_dpo_distributed.py \
#         --config configs/training/llama_3.3_70b.yaml output_dir=runs/llama70b_lr0.008

from __future__ import annotations

import glob
import os
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from safetensors.torch import save_file
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, rlhf, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import disable_dropout
from torchtune.training.lr_schedulers import get_lr
from torchtune.utils import get_world_size_and_rank
from tqdm import tqdm

from heads_of_security.attention_gating.gating_core import GATE_PARAM_SUFFIXES
from heads_of_security.training.collator import padded_collate_dpo_with_channels

log = utils.get_logger("DEBUG")


class GateDPORecipe(FTRecipeInterface):
    """DPO on the attention gates of a frozen base model, distributed with FSDP2.

    Features kept from the torchtune recipe: FSDP sharding (``fsdp_cpu_offload``,
    ``fsdp_reshard_after_forward``), activation checkpointing and offloading, bf16, gradient
    accumulation, gradient clipping, cosine LR schedule, WandB / disk / stdout metric loggers.

    Removed: resume, optimizer-in-backward, profiler, compile, LoRA. Checkpoints hold the
    gate tensors only (``gates/epoch_{e}.safetensors``), which is the file format every
    loader in this package reads.

    Config keys beyond torchtune's: ``freeze_base`` (default True), ``save_every_n_steps``,
    ``keep_last_n_checkpoints``, ``eval_dataset`` / ``eval_every_n_steps`` / ``eval_max_batches``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        if self._dtype == torch.float16:
            raise ValueError("fp16 is not supported; use bf16 or fp32.")

        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = (
            cfg.get("log_peak_memory_stats", False) and self._device.type == "cuda"
        )

        self.world_size, self.rank = get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._enable_activation_checkpointing = cfg.get("enable_activation_checkpointing", False)
        self._enable_activation_offloading = cfg.get("enable_activation_offloading", False)
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError("enable_activation_offloading requires CUDA")
            if not self._enable_activation_checkpointing:
                raise RuntimeError("enable_activation_offloading requires enable_activation_checkpointing")

        self.seed = training.set_seed(seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None))
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self._batch_size = cfg.batch_size

        self._freeze_base = cfg.get("freeze_base", True)
        # 0/None disables mid-epoch saving (only saves at end of epoch).
        self._save_every_n_steps = cfg.get("save_every_n_steps", 0) or 0
        # Retain only the N most recent gate snapshots. 0/None keeps all.
        self._keep_last_n_checkpoints = cfg.get("keep_last_n_checkpoints", 3) or 0
        # Eval cadence: 0/None disables periodic eval.
        self._eval_every_n_steps = cfg.get("eval_every_n_steps", 0) or 0
        self._eval_max_batches = cfg.get("eval_max_batches", 8)
        self.global_step = 0

    # ------------------------------------------------------------------ setup

    def setup(self, cfg: DictConfig) -> None:
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        # Tokenizer first: the builder reads gating_num_channels from it, so the tokenizer's
        # channel policy is the single source of truth for the gate table size.
        self._tokenizer = config.instantiate(cfg.tokenizer)
        cfg.model.gating_num_channels = self._tokenizer.num_channels

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
        )
        self._optimizer = config.instantiate(cfg.optimizer, self._model.parameters())
        self._loss_fn = config.instantiate(cfg.loss)
        utils.log_rank_zero(log, "Optimizer and loss are initialized.")

        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset, shuffle=cfg.shuffle, batch_size=cfg.batch_size
        )
        self._eval_dataloader = None
        if self._eval_every_n_steps > 0 and cfg.get("eval_dataset") is not None:
            _, self._eval_dataloader = self._setup_data(
                cfg_dataset=cfg.eval_dataset, shuffle=False, batch_size=cfg.batch_size
            )
            utils.log_rank_zero(
                log, f"Eval enabled: every {self._eval_every_n_steps} steps, max {self._eval_max_batches} batches."
            )

        self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
        self._lr_scheduler = config.instantiate(
            cfg.lr_scheduler,
            self._optimizer,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=-1,
        )
        if self._is_rank_zero:
            effective = cfg.batch_size * cfg.gradient_accumulation_steps * self.world_size
            log.info(
                f"{len(self._dataloader.dataset)} training pairs, {self._steps_per_epoch} optimizer steps/epoch, "
                f"effective batch {effective} ({cfg.batch_size} x {cfg.gradient_accumulation_steps} accumulation "
                f"x {self.world_size} ranks)"
            )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        utils.log_rank_zero(log, "Instantiating model (weights loaded via from_pretrained) ...")
        init_start = time.perf_counter()

        # The builder calls from_pretrained, so the model arrives with weights loaded on CPU.
        with training.set_default_dtype(self._dtype):
            model = config.instantiate(cfg_model)

        if enable_activation_checkpointing:
            # The models are HF ``PreTrainedModel``s, so HF's own mechanism is used (torchtune's
            # policy would match none of our decoder layers). use_reentrant=False: with a frozen
            # base the layer inputs do not require grad, and reentrant checkpointing silently
            # skips such regions.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            utils.log_rank_zero(log, "Activation checkpointing: HF gradient checkpointing enabled")

        training.shard_model(
            model=model,
            shard_conditions=[partial(training.get_shard_conditions, names_to_match=custom_sharded_layers)],
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(model, enable_activation_offloading)
        training.validate_no_params_on_meta_device(model)
        utils.log_rank_zero(log, f"Instantiating model took {time.perf_counter() - init_start:.2f} secs")
        if self._is_rank_zero:
            training.log_memory_stats(training.get_memory_stats(device=self._device))

        # Dropout off: the reference and policy log-probs must come from the same deterministic net.
        disable_dropout(model)
        # from_pretrained returns an eval-mode model and HF gradient checkpointing only runs
        # when ``self.training`` is True; without this the checkpointing flag is dead.
        model.train()

        if self._freeze_base:
            n_trainable = model.freeze_base()
            utils.log_rank_zero(log, f"Base frozen. Trainable params (gates only): {n_trainable:,}")

        torch.distributed.barrier()
        return model

    def _setup_data(
        self, cfg_dataset: DictConfig, shuffle: bool, batch_size: int
    ) -> Tuple[DistributedSampler, DataLoader]:
        ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
        sampler = DistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle, seed=0)
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=partial(
                padded_collate_dpo_with_channels,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            ),
        )
        return sampler, dataloader

    # ------------------------------------------------------------------ gates

    def _gate_params(self) -> List[Tuple[str, torch.Tensor]]:
        """``(name, full tensor)`` for every gate parameter. ``full_tensor()`` is an FSDP2
        collective and must run on every rank."""
        out = []
        for name, p in self._model.named_parameters():
            if name.endswith(GATE_PARAM_SUFFIXES):
                out.append((name, (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().float()))
        return out

    @staticmethod
    def _gate_means(gates: List[Tuple[str, torch.Tensor]]) -> Dict[str, float]:
        """Mean ``sigmoid(gate)`` per (query channel, key channel) pair over layers and heads,
        plus the sink column when present. The one summary worth watching during training:
        ``gates/pair_0_1`` is a trusted query's gate on untrusted keys."""
        head = [g for n, g in gates if n.endswith("head_gates")]
        sink = [g for n, g in gates if n.endswith("sink_gates")]
        out: Dict[str, float] = {}
        if head:
            stacked = torch.sigmoid(torch.stack(head))  # [L, C, C, H]
            for q in range(stacked.shape[1]):
                for k in range(stacked.shape[2]):
                    out[f"gates/pair_{q}_{k}"] = stacked[:, q, k, :].mean().item()
            out["gates/mean"] = stacked.mean().item()
        if sink:
            s = torch.sigmoid(torch.stack(sink))  # [L, C, H]
            for q in range(s.shape[1]):
                out[f"gates/sink_{q}"] = s[:, q, :].mean().item()
        return out

    # ------------------------------------------------------------------ checkpoints

    @staticmethod
    def _prune_checkpoint_dir(directory: str, keep: int) -> list[str]:
        """Delete all but the ``keep`` most recent ``*.safetensors`` (by mtime) in ``directory``."""
        if keep <= 0:
            return []
        paths = sorted(glob.glob(os.path.join(directory, "*.safetensors")), key=os.path.getmtime)
        removed = []
        for stale in paths[:-keep]:
            try:
                os.remove(stale)
                removed.append(os.path.basename(stale))
            except OSError:
                pass
        return removed

    def save_checkpoint(self, epoch: int, step: Optional[int] = None) -> None:
        """Write the gate tensors to ``output_dir/gates/epoch_{epoch}[_step_{step}].safetensors``
        (fp32, keys ``model.layers.{i}.self_attn.{head_gates,sink_gates}``) and the model config
        to ``output_dir/latest/config.json``. The gates file is what ``from_gates`` /
        ``load_gated_model_local`` / ``push_gates`` take."""
        if self._is_rank_zero:
            log.info("Saving checkpoint ...")
            start = time.perf_counter()
        gate_state = {name: t.cpu().contiguous() for name, t in self._gate_params()}
        if self._is_rank_zero:
            latest_dir = os.path.join(self._output_dir, "latest")
            gates_dir = os.path.join(self._output_dir, "gates")
            os.makedirs(latest_dir, exist_ok=True)
            os.makedirs(gates_dir, exist_ok=True)
            self._model.config.save_pretrained(latest_dir)
            tag = f"epoch_{epoch}" if step is None else f"epoch_{epoch}_step_{step}"
            save_file(gate_state, os.path.join(gates_dir, f"{tag}.safetensors"))
            dropped = self._prune_checkpoint_dir(gates_dir, self._keep_last_n_checkpoints)
            if dropped:
                log.info(f"Pruned {len(dropped)} old gate snapshot(s): {dropped}")
            log.info(f"Saved {len(gate_state)} gate tensors to {gates_dir}/{tag}.safetensors "
                     f"({time.perf_counter() - start:.2f} s)")
        torch.distributed.barrier()

    # ------------------------------------------------------------------ forward / loss

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        use_channels: bool = True,
        activations_handling: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward over the concatenated (chosen, rejected) batch.

        ``use_channels=False`` is the reference policy: the gated attention is a bit-exact no-op
        without ``token_channels``, so the same frozen model is the base model.
        Returns (chosen log-probs, rejected log-probs, chosen logits, rejected logits); log-probs
        are summed over the response tokens.
        """
        input_ids, labels, channels = batch
        input_ids = input_ids.to(self._device)
        labels = labels.to(self._device)
        kwargs = dict(input_ids=input_ids, attention_mask=input_ids.ne(self._tokenizer.pad_id).long())
        if use_channels:
            kwargs["token_channels"] = channels.to(self._device)
        len_chosen = input_ids.shape[0] // 2

        if activations_handling:
            with self.activations_handling_ctx:
                output = model(**kwargs)
        else:
            output = model(**kwargs)
        all_logits = output.logits if hasattr(output, "logits") else output
        if torch.isnan(all_logits).any():
            utils.log_rank_zero(log, f"NaN in logits (use_channels={use_channels}, shape={tuple(all_logits.shape)})")

        chosen_log_probs = rlhf.get_batch_log_probs(
            all_logits[:len_chosen], labels[:len_chosen], return_average_logprobs=False
        )
        rejected_log_probs = rlhf.get_batch_log_probs(
            all_logits[len_chosen:], labels[len_chosen:], return_average_logprobs=False
        )
        return chosen_log_probs, rejected_log_probs, all_logits[:len_chosen], all_logits[len_chosen:]

    def _run_eval(self) -> Dict[str, float]:
        """Forward-only DPO on at most ``eval_max_batches`` batches of the eval set (rank-local)."""
        if self._eval_dataloader is None:
            return {}
        was_training = self._model.training
        self._model.eval()
        agg = {k: 0.0 for k in ("loss", "rewards/chosen", "rewards/rejected", "rewards/accuracies")}
        n_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self._eval_dataloader):
                if batch_idx >= self._eval_max_batches:
                    break
                n_batches += 1
                p_chos, p_rej, _, _ = self.concatenated_forward(
                    self._model, batch, use_channels=True, activations_handling=False
                )
                r_chos, r_rej, _, _ = self.concatenated_forward(
                    self._model, batch, use_channels=False, activations_handling=False
                )
                loss, chos_rew, rej_rew = self._loss_fn(p_chos, p_rej, r_chos, r_rej)
                agg["loss"] += loss.mean().item()
                agg["rewards/chosen"] += chos_rew.mean().item()
                agg["rewards/rejected"] += rej_rew.mean().item()
                agg["rewards/accuracies"] += (chos_rew > rej_rew).float().mean().item()
        if was_training:
            self._model.train()
        if n_batches == 0:
            return {}
        out = {k: v / n_batches for k, v in agg.items()}
        out["rewards/margins"] = out["rewards/chosen"] - out["rewards/rejected"]
        return out

    # ------------------------------------------------------------------ train

    def train(self) -> None:
        training.cleanup_before_training()
        self._optimizer.zero_grad()
        t0 = time.perf_counter()
        running_loss = 0
        running = {k: 0 for k in ("rewards/chosen", "rewards/rejected", "rewards/accuracies",
                                  "log_probs/chosen", "log_probs/rejected",
                                  "log_probs/ref_chosen", "log_probs/ref_rejected")}
        num_tokens = 0

        init_means = self._gate_means(self._gate_params())
        if self._is_rank_zero:
            log.info("[step 0] " + " ".join(f"{k.removeprefix('gates/')}={v:.4f}" for k, v in init_means.items()))
            self._metric_logger.log_dict(init_means, step=0)

        for curr_epoch in range(self.total_epochs):
            self._sampler.set_epoch(curr_epoch)
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            for idx, batch in enumerate(self._dataloader):
                if (self.max_steps_per_epoch is not None
                        and (idx // self._gradient_accumulation_steps) == self.max_steps_per_epoch):
                    break
                num_tokens += torch.tensor(batch[0].numel())

                policy_chosen, policy_rejected, pc_logits, pr_logits = self.concatenated_forward(
                    self._model, batch, use_channels=True
                )
                del pc_logits, pr_logits  # freeing the full logits before the reference pass lowers peak memory
                with torch.no_grad():
                    ref_chosen, ref_rejected, rc_logits, rr_logits = self.concatenated_forward(
                        self._model, batch, use_channels=False, activations_handling=False
                    )
                del rc_logits, rr_logits

                loss, chosen_rewards, rejected_rewards = self._loss_fn(
                    policy_chosen, policy_rejected, ref_chosen, ref_rejected
                )
                loss = loss.mean() / self._gradient_accumulation_steps
                running_loss += loss
                s = 1 / self._gradient_accumulation_steps
                running["rewards/chosen"] += s * chosen_rewards.mean()
                running["rewards/rejected"] += s * rejected_rewards.mean()
                running["rewards/accuracies"] += s * (chosen_rewards > rejected_rewards).float().mean()
                running["log_probs/chosen"] += s * policy_chosen.detach().mean()
                running["log_probs/rejected"] += s * policy_rejected.detach().mean()
                running["log_probs/ref_chosen"] += s * ref_chosen.mean()
                running["log_probs/ref_rejected"] += s * ref_rejected.mean()

                loss.backward()

                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    torch.distributed.all_reduce(running_loss, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(num_tokens)
                    for key in running:
                        torch.distributed.all_reduce(running[key], op=torch.distributed.ReduceOp.AVG)
                    grad_norm = None
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), max_norm=float(self._clip_grad_norm)
                        ).full_tensor()

                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self._lr_scheduler.step()
                    gate_means = self._gate_means(self._gate_params())

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log:.4f}")

                    if self._save_every_n_steps > 0 and self.global_step % self._save_every_n_steps == 0:
                        self.save_checkpoint(epoch=curr_epoch, step=self.global_step)

                    if self._eval_dataloader is not None and self.global_step % self._eval_every_n_steps == 0:
                        eval_metrics = self._run_eval()
                        if self._is_rank_zero and eval_metrics:
                            self._metric_logger.log_dict(
                                {f"eval/{k}": v for k, v in eval_metrics.items()}, step=self.global_step
                            )
                            log.info(f"[step {self.global_step}] eval: loss={eval_metrics['loss']:.4f} "
                                     f"acc={eval_metrics['rewards/accuracies']:.3f} "
                                     f"margin={eval_metrics['rewards/margins']:.3f}")

                    if self.global_step % self._log_every_n_steps == 0 and self._is_rank_zero:
                        time_per_step = time.perf_counter() - t0
                        log_dict: Dict[str, Any] = {
                            "loss": loss_to_log,
                            "lr": get_lr(self._optimizer),
                            "tokens_per_second_per_gpu": num_tokens / (time_per_step * self.world_size),
                            "rewards/margins": (running["rewards/chosen"] - running["rewards/rejected"]).cpu(),
                            **{k: v.cpu() for k, v in running.items()},
                            **gate_means,
                        }
                        if grad_norm is not None:
                            log_dict["grad_norm"] = grad_norm.item()
                        if self._log_peak_memory_stats:
                            log_dict.update(training.get_memory_stats(device=self._device))
                        self._metric_logger.log_dict(log_dict, step=self.global_step)
                        log.info(
                            f"[step {self.global_step}] loss={loss_to_log:.4f} "
                            f"acc={float(running['rewards/accuracies']):.3f} "
                            f"margin={float(log_dict['rewards/margins']):.3f} "
                            f"pair_0_1={gate_means.get('gates/pair_0_1', float('nan')):.4f}"
                        )

                    running_loss = 0
                    running = {key: 0 for key in running}
                    num_tokens = 0
                    t0 = time.perf_counter()

            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point. Config values come from the YAML and are overwritten by ``key=value`` CLI args."""
    if not training.is_distributed():
        raise RuntimeError("Run through a distributed launcher: tune run --nproc_per_node <gpus> ...")
    init_process_group("cuda:nccl,cpu:gloo")
    if cfg.get("fsdp_cpu_offload", False):
        training.set_torch_num_threads()
    config.log_config(recipe_name="GateDPORecipe", cfg=cfg)
    recipe = GateDPORecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
