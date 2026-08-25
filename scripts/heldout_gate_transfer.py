#!/usr/bin/env python3
"""Held-out evaluation of the simulation-tuned gate config, in both simulation and real world.

`gate_transfer` takes the best tuned configs on a single task and measured it on a different task.
This file held out some episodes from a same task and measure performance of gates tuned on the 
other episodes. 

Used primarily for testing real world gates because real world rollouts are less, 
so testing held out is more faithful because otherwise the best gates can easily overfit to noise.

Outputs:

  overfit gap        the in-sample best-of-grid minus what retuning actually reaches on unseen
                     episodes. It is small when episodes are plentiful and large when they are
                     scarce, which is why real-robot numbers cannot be read in-sample.

  fixed vs retuned   the simulation-tuned config against one retuned for best balanced accuracy on
                     each corpus, on both axes -- balanced accuracy and detection time. The retuned
                     config is chosen on accuracy alone, so it is free to fire late; reporting both
                     axes is what makes the comparison honest.

Each corpus is seeded independently, so results do not depend on the order corpora are listed in.

Usage:
    uv run python scripts/heldout_gate_transfer.py
    uv run python scripts/heldout_gate_transfer.py --repeats 4000 --out outputs/heldout.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_gate_configs import (  # noqa: E402
    DTHRS, INF, LW_GRID, MAGS, PTHRS, SMOOTH, SWS, drop_first_fires, ema,
    plateau_first_fires)

# The config chosen in simulation by leave-one-task-out, and the one deployed on the robot.
SIM_CONFIG = dict(smoothing=0.0, sw=75, dthr=-0.9, mag=0.0, lw=225, pthr=0.0)

DEFAULT_CORPORA = [
    ("LIBERO t0", "gated_videos/n200_t0_dp/episode_stats.json"),
    ("LIBERO t1", "gated_videos/n200_t1_dp/episode_stats.json"),
    ("LIBERO t5", "gated_videos/n200_t5_dp/episode_stats.json"),
    ("LIBERO t8", "gated_videos/n200_t8_dp/episode_stats.json"),
    ("LIBERO t4", "gated_videos/n200_t4/episode_stats.json"),
    ("Real robot", "outputs/robometer_labels/episode_stats_stack_live.json"),
]


def load(path):
    """Replay every config over one corpus. Returns first-fire sample index per (episode, config).

    The gate is a pure function of the progress trace, so this is exact rather than simulated.
    Window lengths are given in env steps and converted to trace samples using the corpus's own
    scoring cadence, so a config covers the same span of robot time whatever cadence it was
    recorded at.
    """
    blob = json.load(open(path))
    episodes = blob["episodes"]
    score_every = int((blob.get("meta") or {}).get("score_every")
                      or episodes[0].get("score_every") or 1)
    succ = np.array([bool(e.get("success")) for e in episodes])
    traces = [np.asarray(e["progress_trace"], dtype=np.float64) for e in episodes]
    horizon = max(len(t) for t in traces) * score_every

    sws = sorted({max(2, int(round(w / score_every))) for w in SWS})
    lws = sorted({max(2, int(round(w / score_every)))
                  for w in LW_GRID if w <= horizon})
    sw_steps = {s: s * score_every for s in sws}
    lw_steps = {s: s * score_every for s in lws}

    drop, plat = {}, {}
    for s in SMOOTH:
        for ei, v in enumerate(traces):
            vs = ema(v, s)
            drop[(s, ei)] = drop_first_fires(vs, sws, DTHRS, MAGS)
            plat[(s, ei)] = plateau_first_fires(vs, lws, PTHRS)

    keys, cols = [], []
    for s in SMOOTH:
        for sw, dthr, mag in itertools.product(sws, DTHRS, MAGS):
            for lw, pthr in itertools.product(lws, PTHRS):
                keys.append((s, sw_steps[sw], dthr, mag, lw_steps[lw], pthr))
                cols.append([min(drop[(s, ei)][(sw, dthr, mag)], plat[(s, ei)][(lw, pthr)])
                             for ei in range(len(episodes))])
    key = (SIM_CONFIG["smoothing"], SIM_CONFIG["sw"], SIM_CONFIG["dthr"],
           SIM_CONFIG["mag"], SIM_CONFIG["lw"], SIM_CONFIG["pthr"])
    if key not in keys:
        raise SystemExit(f"{path}: the sim config {key} is not on this corpus's grid")
    return np.asarray(cols).T, succ, score_every, horizon, keys.index(key)


def score(fires, succ, idx, score_every, horizon):
    """Balanced accuracy and mean detection time for every config, over the episodes in `idx`."""
    sub = fires[idx]
    fired = sub < INF
    s, f = succ[idx], ~succ[idx]
    balacc = (fired[f].mean(axis=0) + 1.0 - fired[s].mean(axis=0)) / 2.0
    ff = sub[f]
    # a failure that is never caught counts as the latest possible detection
    tdet = np.where(ff < INF, np.minimum(ff * score_every, horizon) / horizon, 1.0).mean(axis=0)
    return balacc, tdet


def evaluate(path, repeats, seed):
    fires, succ, score_every, horizon, ti = load(path)
    rng = np.random.default_rng(seed)
    S, F = np.flatnonzero(succ), np.flatnonzero(~succ)

    insample, _ = score(fires, succ, np.arange(len(succ)), score_every, horizon)
    ret_ba, ret_td, fix_ba, fix_td = [], [], [], []
    for _ in range(repeats):
        s1, f1 = rng.permutation(S), rng.permutation(F)
        ns, nf = len(S) // 2, len(F) // 2
        train = np.concatenate([s1[:ns], f1[:nf]])
        test = np.concatenate([s1[ns:], f1[nf:]])
        ba_tr, td_tr = score(fires, succ, train, score_every, horizon)
        ba_te, td_te = score(fires, succ, test, score_every, horizon)
        # Retuning: most accurate on the training half, ties broken by firing earliest. The tie
        # break matters -- on a small corpus balanced accuracy takes few distinct values, so
        # hundreds of configs can sit at the maximum and picking among them arbitrarily would be
        # a weaker baseline than anyone would actually deploy.
        tied = np.flatnonzero(ba_tr >= ba_tr.max() - 1e-12)
        best = int(tied[np.argmin(td_tr[tied])])
        ret_ba.append(ba_te[best]); ret_td.append(td_te[best])
        fix_ba.append(ba_te[ti]); fix_td.append(td_te[ti])

    ret_ba, ret_td = np.array(ret_ba), np.array(ret_td)
    fix_ba, fix_td = np.array(fix_ba), np.array(fix_td)
    d = ret_ba - fix_ba
    return dict(
        n=int(len(succ)), n_success=int(succ.sum()), n_failure=int((~succ).sum()),
        horizon=int(horizon),
        insample_best=float(insample.max()),
        retuned_balacc=float(ret_ba.mean()), fixed_balacc=float(fix_ba.mean()),
        overfit_gap=float(insample.max() - ret_ba.mean()),
        balacc_loss=float(d.mean()),
        balacc_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        retuned_tdet=float(ret_td.mean()), fixed_tdet=float(fix_td.mean()),
        retuned_step=float(ret_td.mean() * horizon), fixed_step=float(fix_td.mean() * horizon),
        pct_earlier=float(100 * (ret_td.mean() - fix_td.mean()) / ret_td.mean()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=2000, help="split-half repeats per corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpora", nargs="+", default=None, help='"Name:path" pairs')
    ap.add_argument("--out", default="outputs/rw_frontier/heldout_gate_transfer.json")
    args = ap.parse_args()

    corpora = ([tuple(c.split(":", 1)) for c in args.corpora] if args.corpora
               else DEFAULT_CORPORA)
    rows = {}
    for tag, path in corpora:
        if not os.path.exists(path):
            print(f"  skipping {tag}: {path} not found")
            continue
        rows[tag] = evaluate(path, args.repeats, args.seed)

    print(f"\nOVERFIT GAP  (in-sample best minus what retuning reaches on held-out episodes)")
    print(f"  {'corpus':<12s}{'n':>5s}{'in-sample':>11s}{'held-out':>10s}{'gap':>8s}")
    for tag, r in rows.items():
        print(f"  {tag:<12s}{r['n']:>5d}{r['insample_best']:>11.3f}"
              f"{r['retuned_balacc']:>10.3f}{r['overfit_gap']:>8.3f}")

    print(f"\nFIXED SIM CONFIG vs RETUNED FOR BEST BALANCED ACCURACY  (held out)")
    print(f"  {'corpus':<12s}{'balacc fixed/retuned':>22s}{'loss':>8s}{'95% CI':>18s}"
          f"{'steps fixed/retuned':>21s}{'earlier':>9s}")
    for tag, r in rows.items():
        ci = f"[{r['balacc_ci'][0]:+.3f},{r['balacc_ci'][1]:+.3f}]"
        print(f"  {tag:<12s}{r['fixed_balacc']:>11.3f}{r['retuned_balacc']:>11.3f}"
              f"{r['balacc_loss']:>8.3f}{ci:>18s}"
              f"{r['fixed_step']:>11.0f}{r['retuned_step']:>10.0f}"
              f"{r['pct_earlier']:>8.0f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
