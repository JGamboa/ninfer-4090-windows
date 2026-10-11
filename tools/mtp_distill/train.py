"""Evaluate and train Bonsai's MTP layer against Bonsai's own argmax (self-distillation).

  python -m tools.mtp_distill.train eval  --artifact TARGET.ninfer --data DIR --weights ARTIFACT_OR_PT [--steps 3]
  python -m tools.mtp_distill.train train --artifact TARGET.ninfer --data DIR --weights ARTIFACT_OR_PT \
      --out FILE.pt [--lr 1e-5 --epochs 1 --steps 3]

TARGET is the model whose MTP is trained (its embedding and proposal head are used, frozen);
--weights initializes the MTP layer (an artifact with an MTP component, or a saved .pt).

Accuracy of step s = fraction of columns whose step-s draft (argmax over the proposal head,
mapped to the global vocabulary, teacher-forced on the true preceding tokens) equals the
teacher's argmax for that token. Step 1 is the runtime's position-1 acceptance under greedy
verification; steps 2 and 3 are the chained drafts given correct earlier drafts.
"""

from __future__ import annotations

import argparse
import math
import random
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.mtp_distill import mtp_torch as m

STEP_WEIGHTS = (1.0, 0.6, 0.4)


def split(dataset: m.Dataset, held_mod: int = 10):
    train, held = [], []
    for i, meta in enumerate(dataset.texts):
        (held if zlib.crc32(meta["id"].encode()) % held_mod == 0 else train).append(i)
    return train, held


def load_weights(source: str) -> dict[str, torch.Tensor]:
    return torch.load(source) if source.endswith(".pt") else m.load_mtp_weights(source)


class Context:
    def __init__(self, data: str, artifact: str, embedding_cache: Path):
        self.dataset = m.Dataset(data)
        self.embedding = m.load_embedding(artifact, embedding_cache)
        head, ids = m.load_proposal(artifact)
        self.head = head.cuda()
        self.ids = ids.cuda()
        local = torch.full((self.embedding.shape[0],), -100, dtype=torch.int64)
        local[ids] = torch.arange(ids.numel())
        self.local = local.cuda()  # global id -> proposal row, -100 when outside the shortlist

    def batch(self, i: int, max_columns: int = 4096):
        hidden, nxt, target = self.dataset.sequence(i)
        hidden, nxt, target = hidden[:max_columns], nxt[:max_columns], target[:max_columns]
        emb = self.embedding[nxt]
        positions = torch.arange(hidden.shape[0])
        return hidden.cuda(), emb.cuda(), positions.cuda(), target.cuda()

    def chain(self, i: int, steps: int, max_columns: int):
        hidden, per_step = self.dataset.chain(i, steps)
        hidden = hidden[:max_columns].cuda()
        embs = [self.embedding[tok[:max_columns]].cuda() for tok, _ in per_step]
        targets = [tgt[:max_columns].cuda() for _, tgt in per_step]
        positions = torch.arange(hidden.shape[0], device="cuda")
        return hidden, embs, positions, targets


def logits(ctx: Context, mtp_hidden: torch.Tensor) -> torch.Tensor:
    return mtp_hidden.to(torch.bfloat16) @ ctx.head.T


@torch.no_grad()
def evaluate(ctx: Context, model: m.Mtp, indices: list[int], label: str, steps: int,
             max_columns: int = 4096) -> list[float]:
    model.eval()
    hits = [0] * steps
    totals = [0] * steps
    by_cat: dict[str, list[int]] = {}
    for i in indices:
        hidden, embs, pos, targets = ctx.chain(i, steps, max_columns)
        outs = model.chain(hidden, embs, pos)
        cat = ctx.dataset.texts[i]["id"].split("-")[0]
        for s, (out, target) in enumerate(zip(outs, targets)):
            draft = ctx.ids[logits(ctx, out).argmax(-1)]
            h = int((draft == target).sum())
            hits[s] += h
            totals[s] += target.numel()
            if s == 0:
                c = by_cat.setdefault(cat, [0, 0])
                c[0] += h
                c[1] += target.numel()
    accs = [h / max(t, 1) for h, t in zip(hits, totals)]
    cats = "  ".join(f"{k} {v[0] / v[1]:.3f}" for k, v in sorted(by_cat.items()))
    steps_text = " ".join(f"s{s + 1} {a:.4f}" for s, a in enumerate(accs))
    print(f"[{label}] {steps_text} over {totals[0]} columns  |  step 1 by category: {cats}", flush=True)
    return accs


def train(ctx: Context, model: m.Mtp, train_ids, held_ids, args) -> None:
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps_total = args.epochs * len(train_ids)
    warmup = max(20, steps_total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / steps_total, 1.0))))
    rng = random.Random(0)
    step = 0
    started = time.time()
    running = None
    for epoch in range(args.epochs):
        order = list(train_ids)
        rng.shuffle(order)
        model.train()
        for i in order:
            hidden, embs, pos, targets = ctx.chain(i, args.steps, args.max_columns)
            outs = model.chain(hidden, embs, pos)
            loss = 0.0
            for s, (out, target) in enumerate(zip(outs, targets)):
                loss = loss + STEP_WEIGHTS[s] * F.cross_entropy(
                    logits(ctx, out).float(), ctx.local[target], ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            value = float(loss.detach())
            running = value if running is None else 0.98 * running + 0.02 * value
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps_total} loss {running:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - started:.0f}s", flush=True)
            if args.eval_every and step % args.eval_every == 0:
                evaluate(ctx, model, held_ids, f"held-out at step {step}", args.steps)
                model.train()
        evaluate(ctx, model, held_ids, f"held-out after epoch {epoch}", args.steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval", "train"])
    parser.add_argument("--artifact", required=True, help="target .ninfer (embedding, proposal head)")
    parser.add_argument("--embedding-cache", type=Path, required=True,
                        help="file for the decoded embedding table (created on first use)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True, help="artifact or .pt whose MTP weights initialize the model")
    parser.add_argument("--out")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-columns", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--held-mod", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(0)
    ctx = Context(args.data, args.artifact, args.embedding_cache)
    train_ids, held_ids = split(ctx.dataset, args.held_mod)
    print(f"texts: {len(train_ids)} train, {len(held_ids)} held out", flush=True)
    model = m.Mtp(load_weights(args.weights)).cuda()
    evaluate(ctx, model, held_ids, "held-out, initial", args.steps)
    if args.mode == "eval":
        return
    train(ctx, model, train_ids, held_ids, args)
    torch.save({k.replace("__", "/"): v.detach().cpu() for k, v in model.p.items()}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
