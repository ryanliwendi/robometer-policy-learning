#!/usr/bin/env python3
"""Refit the real-world ThriftyDAgger gate between DAgger rounds.

The gate has two halves and they read different data, exactly as they do in sim. ``--bc-sources``
is the expert-support pool: the operator's takeover frames, plus the policy's own successful
rollouts standing in for the demos sim mixes in. That is what the novelty ensemble clones and what
both thresholds are quantiles of. ``--q-sources`` is the pool the risk critics learn from, and it
has to contain failures -- with a human expert who rescues nearly every episode, the intervention
rounds are almost all successes, so the solo eval rollouts are where the failed endings live.

Each source is ``dir[:segments=human|all,require_success=0|1]``.

Example Usage:
    uv run python scripts/refresh_thrifty_real.py \
        --bc-sources data/task1_stack/rdagger/round1_train:segments=human \
                     data/task1_stack/rdagger/round2_train:segments=human \
                     data/task1_stack/pi05_rollouts:require_success=1 \
        --q-sources  data/task1_stack/pi05_rollouts \
                     data/task1_stack/rdagger/round1_train \
                     data/task1_stack/rdagger/round2_train \
        --out results/thrifty/stack/gate_r2.pt
"""

import argparse
import pathlib

from loguru import logger

from robometer_policy_learning.utils.thrifty_real import (
    DinoFeatureActor, fit_from_npz, parse_source, save_thrifty)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc-sources", nargs="+", required=True,
                    help="expert-support sources for the novelty ensemble and the thresholds, "
                         "each dir[:segments=human|all,require_success=0|1]")
    ap.add_argument("--q-sources", nargs="+", required=True,
                    help="sources for the risk critics; must contain failed episodes")
    ap.add_argument("--out", required=True, help="checkpoint path to write")
    ap.add_argument("--target-rate", type=float, default=0.01,
                    help="fraction of calibration states the gate should fire on. Sim used 0.001 "
                         "over ~13k pooled steps; a real round's expert-support pool is ~8k, and "
                         "0.001 of that is 8 states, too few to estimate a quantile from")
    ap.add_argument("--grad-steps", type=int, default=2000)
    ap.add_argument("--num-nets", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.9999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dinov2-model", default="facebook/dinov2-base",
                    help="use facebook/dinov2-small if scoring cannot keep up at 15 Hz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    for spec in list(args.bc_sources) + list(args.q_sources):
        d = parse_source(spec)["dir"]        # also rejects a malformed spec before the model loads
        if not pathlib.Path(d).expanduser().is_dir():
            raise SystemExit(f"not a directory: {d}")
    if any(parse_source(s)["segments"] == "human" for s in args.q_sources):
        raise SystemExit("--q-sources cannot use segments=human: dropping the policy's own frames "
                         "would leave the critics with no failure to learn from")

    actor = DinoFeatureActor(dinov2_model=args.dinov2_model, device=args.device)
    gate, ac, info = fit_from_npz(
        args.bc_sources, args.q_sources, target_rate=args.target_rate,
        grad_steps=args.grad_steps, num_nets=args.num_nets, batch_size=args.batch_size,
        lr=args.lr, gamma=args.gamma, seed=args.seed, device=args.device, actor=actor)

    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_thrifty(out, gate, ac, actor, info["act_dim"],
                 meta=dict(target_rate=args.target_rate, **info))
    logger.info(f"done: {out}")


if __name__ == "__main__":
    main()
