#!/usr/bin/env python3
"""Offline grid-search over RewardGate hyperparameters.

What we want from a gate, operationally:
  * FIRE on failures -- and fire EARLY, while a correction is still worth making.
  * STAY SILENT on successes -- every fire on a healthy rollout hands control to the expert
    for no reason and pollutes the DAgger buffer.

Note the traces are not gated, so this only measures the gate's first fire faithfully 

Usage:
    uv run python scripts/sweep_gate_hparams.py [stats.json]
"""

import itertools
import json
import sys

import numpy as np

from robometer_policy_learning.utils.reward_gate import RewardGate

STATS = sys.argv[1] if len(sys.argv) > 1 else "gated_videos/student_sweep/episode_stats.json"
WARMUP = 15


def first_fire(trace, warmup=WARMUP, **gate_kwargs):
    """Replay the gate over one trace. Returns (step, trigger) of the first fire, else (None, None)."""
    gate = RewardGate(**gate_kwargs)
    for i, p in enumerate(trace):
        fired = gate.update(float(p))
        if fired and i >= warmup:
            return i, gate.last_trigger
    return None, None


def evaluate(episodes, **gate_kwargs):
    fails = [e for e in episodes if not e["success"]]
    succs = [e for e in episodes if e["success"]]

    caught, fire_steps, triggers = 0, [], []
    for e in fails:
        step, trig = first_fire(e["progress_trace"], **gate_kwargs)
        if step is not None:
            caught += 1
            fire_steps.append(step)
            triggers.append(trig)

    false_fires = 0
    for e in succs:
        step, _ = first_fire(e["progress_trace"], **gate_kwargs)
        if step is not None:
            false_fires += 1

    return dict(
        recall=caught / max(len(fails), 1),           # fraction of failures caught
        fpr=false_fires / max(len(succs), 1),         # fraction of successes disturbed
        median_fire=float(np.median(fire_steps)) if fire_steps else float("nan"),
        n_drop=sum(t == "drop" for t in triggers),
        n_plateau=sum(t == "plateau" for t in triggers),
        **gate_kwargs,
    )


def main():
    episodes = json.load(open(STATS))["episodes"]
    n_f = sum(not e["success"] for e in episodes)
    n_s = len(episodes) - n_f
    print(f"{len(episodes)} episodes: {n_s} success, {n_f} failure\n")

    grid = dict(
        method=["spearman"],
        short_window=[30, 50],  # 10, 20
        drop_threshold=[-0.7, -0.9],
        min_drop_magnitude=[0.1, 0.2], # 0.05, 0.3
        long_window=[100, 120, 150],  # 60
        plateau_threshold=[0.1], # 0.3
        smoothing=[0.0, 0.5],
    )
    keys = list(grid)
    results = [evaluate(episodes, **dict(zip(keys, combo)))
               for combo in itertools.product(*grid.values())]

    # Rank: catch every failure, disturb no success, then fire as early as possible.
    results.sort(key=lambda r: (-r["recall"], r["fpr"], r["median_fire"]))

    hdr = (f"{'recall':>6} {'fpr':>5} {'medfire':>7} {'drop/plat':>9} | {'sw':>3} {'dthr':>5} "
           f"{'mag':>4} {'lw':>4} {'pthr':>4} {'sm':>4}")
    print(hdr); print("-" * len(hdr))
    for r in results[:20]:
        print(f"{r['recall']:6.2f} {r['fpr']:5.2f} {r['median_fire']:7.0f} "
              f"{r['n_drop']:4d}/{r['n_plateau']:<4d} | {r['short_window']:3d} "
              f"{r['drop_threshold']:5.1f} {r['min_drop_magnitude']:4.2f} {r['long_window']:4d} "
              f"{r['plateau_threshold']:4.1f} {r['smoothing']:4.1f}")

    with open("gated_videos/student_sweep/gate_sweep.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n{len(results)} configs -> gated_videos/student_sweep/gate_sweep.json")


if __name__ == "__main__":
    main()
