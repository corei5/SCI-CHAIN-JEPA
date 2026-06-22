"""
train.py -- training loop with: from_scratch pretraining OR fine-tuning,
jepa_enabled switch, loss-dropout, configurable JEPA loss, LR warmup+cosine,
checkpoint/resume, edge-pass timing (for the loss-dropout speed study).
Multi-GPU via Accelerate. Returns a dict of run stats (timing).

ANTI-COLLAPSE: the JEPA target embedding is DETACHED (stop-gradient) when
CFG.detach_target is True. This prevents the trivial collapse where the encoder
makes all embeddings identical to drive the cosine loss to zero.
"""
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm

from config import CFG
from model import SciChainJEPA
from data_stream import build_shards, ChainShardDataset


def jepa_loss(pred, true):
    if CFG.jepa_loss_type == "mse":
        return ((pred - true) ** 2).sum(-1).mean()
    return (1.0 - (pred * true).sum(-1)).mean()   # cosine


def collate(batch):
    return batch


def train():
    acc = Accelerator(mixed_precision="bf16" if CFG.use_amp and torch.cuda.is_available()
                      else "no", gradient_accumulation_steps=CFG.grad_accum_steps)
    random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.seed)

    if acc.is_main_process:
        build_shards()
    acc.wait_for_everyone()

    loader = DataLoader(ChainShardDataset("train"), batch_size=CFG.batch_size,
                        num_workers=CFG.num_workers, collate_fn=collate)
    model = SciChainJEPA()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=CFG.lr, weight_decay=CFG.weight_decay)
    total = CFG.max_train_steps if CFG.max_train_steps > 0 else 100_000
    scheduler = get_cosine_schedule_with_warmup(optimizer, CFG.warmup_steps, total)
    model, optimizer, loader, scheduler = acc.prepare(model, optimizer, loader, scheduler)

    acc.print(f"\n{'='*60}\n  MODE: from_scratch={CFG.from_scratch} | "
              f"JEPA={CFG.jepa_enabled} | lambda={CFG.lambda_jepa} | "
              f"loss_dropout={CFG.loss_dropout} | detach_target={CFG.detach_target}"
              f"\n{'='*60}\n")

    base = acc.unwrap_model(model)   # pulled out of the loop (cheap but cleaner)
    global_step, edges_run, t0 = 0, 0, time.time()
    model.train()
    stop = False
    for epoch in range(CFG.num_epochs):
        pbar = tqdm(loader, disable=not acc.is_main_process, desc=f"epoch {epoch}")
        for batch in pbar:
            if len(batch) == 0:
                continue
            with acc.accumulate(model):
                full = [" ".join(c[s] for s in CFG.stages) for c in batch]
                loss = base.ntp_loss(full)

                if CFG.jepa_enabled:
                    edges = list(CFG.chain_edges)
                    if not CFG.use_long_range_edge:
                        edges = [(i, j) for (i, j) in edges if not (i == 2 and j == 4)]
                    jl, used = 0.0, 0
                    for (i, j) in edges:
                        if random.random() < CFG.loss_dropout:    # loss dropout
                            continue
                        ctx = [" ".join(c[CFG.stages[k]] for k in range(i + 1)) for c in batch]
                        tgt = [c[CFG.stages[j]] for c in batch]
                        pe = base.predict_stage(ctx)

                        # ----- ANTI-COLLAPSE: stop-gradient on the target -----
                        if CFG.detach_target:
                            with torch.no_grad():
                                te = base.encode(tgt)
                            te = te.detach()
                            # TODO (optional, stronger): replace the line above with
                            # an EMA target encoder:
                            #   te = ema_model.encode(tgt).detach()
                            # updating ema_model after optimizer.step() each iter.
                        else:
                            te = base.encode(tgt)   # original (collapse-prone) path
                        # ------------------------------------------------------

                        edges_run += 1
                        if CFG.use_trivial_edge_filter:
                            with torch.no_grad():
                                if (pe * te).sum(-1).mean().item() > CFG.trivial_edge_cosine_threshold:
                                    continue
                        jl = jl + jepa_loss(pe, te); used += 1
                    if used > 0:
                        loss = loss + CFG.lambda_jepa * (jl / used)

                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if acc.sync_gradients:
                global_step += 1
                if global_step % CFG.log_every_steps == 0:
                    pbar.set_postfix(step=global_step, loss=float(loss))
                if global_step % CFG.save_every_steps == 0 and acc.is_main_process:
                    acc.save_state(os.path.join(CFG.output_dir, f"step{global_step}"))
                if 0 < CFG.max_train_steps <= global_step:
                    stop = True
                    break
        if stop:
            break

    elapsed = time.time() - t0
    stats = {"wall_time_s": elapsed, "steps": global_step,
             "edges_run": edges_run,
             "steps_per_sec": global_step / max(1e-9, elapsed)}

    if acc.is_main_process:
        tag = f"{'scratch' if CFG.from_scratch else 'ft'}_" \
              f"{'jepa' if CFG.jepa_enabled else 'baseline'}_seed{CFG.seed}"
        final = os.path.join(CFG.output_dir, f"final_{tag}")
        os.makedirs(final, exist_ok=True)
        base.lm.save_pretrained(final)
        base.tokenizer.save_pretrained(final)
        acc.print(f"[train] Saved -> {final} | stats={stats}")
        stats["model_path"] = final
    return stats


if __name__ == "__main__":
    train()
