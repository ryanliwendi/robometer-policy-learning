#!/usr/bin/env python3
"""Try gate configs on prerecorded offline episodes and store them. Gates include robometer 
and baselines (timeout, absolute thres).

Each config gets two scores:
  - accuracy: does it fire on the failures and stay quiet on the successes (balanced accuracy)
  - latency: how early it fires, as a fraction of the episode length

Usage:
    uv run python scripts/sweep_gate_configs.py <episode_stat_dir> \
      --out-dir <output_dir>
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

INF = np.iinfo(np.int32).max

SMOOTH = [0.0, 0.5]
SWS = [20, 30, 50, 75, 100]
DTHRS = [-0.5, -0.7, -0.9]
MAGS = [0.0, 0.05, 0.1, 0.2]
LW_GRID = [150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450]
PTHRS = [-0.2, 0.0, 0.1, 0.3, 0.5]


def _avg_ranks(W: np.ndarray) -> np.ndarray:
    """Replace the values in each row by their position in sorted order, sharing tied positions. Used by spearman correlation"""
    n, w = W.shape
    order = np.argsort(W, axis=1, kind="stable")
    Ws = np.take_along_axis(W, order, axis=1)

    neq = np.empty((n, w), dtype=bool)  # True where a new run of equal values starts
    neq[:, 0] = True
    neq[:, 1:] = Ws[:, 1:] != Ws[:, :-1]
    last = np.empty((n, w), dtype=bool)  # True where a run of equal values ends
    last[:, -1] = True
    last[:, :-1] = neq[:, 1:]

    ar = np.arange(w)
    grp_start = np.maximum.accumulate(np.where(neq, ar, -1), axis=1)
    grp_end = np.minimum.accumulate(np.where(last, ar, w)[:, ::-1], axis=1)[:, ::-1]
    avg_sorted = (grp_start + grp_end) / 2.0   # everyone in a run gets the run's middle position

    ranks = np.empty((n, w), dtype=np.float64)
    np.put_along_axis(ranks, order, avg_sorted, axis=1)
    return ranks


def rolling_spearman(v: np.ndarray, w: int) -> np.ndarray:
    """The spearman correlation over the last `w` steps."""
    n = len(v)
    out = np.full(n, np.nan)
    if n < w or w < 2:
        return out
    W = sliding_window_view(v, w)  # one row per step: that step's last w values
    flat = W.max(axis=1) == W.min(axis=1)
    R = _avg_ranks(W)

    # Correlate each row's ranks against 0,1,2,...,w-1 (the step numbers).
    x = np.arange(w, dtype=np.float64)
    x -= x.mean()
    xden = np.sqrt((x * x).sum())
    Rc = R - R.mean(axis=1, keepdims=True)
    num = Rc @ x
    den = np.sqrt((Rc * Rc).sum(axis=1)) * xden
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / den
    corr[flat] = np.nan
    out[w - 1:] = corr
    return out


def rolling_dropmag(v: np.ndarray, w: int) -> np.ndarray:
    """Min_drop_magnitude check: for every step, how far progress has fallen below its peak in the last `w` steps."""
    n = len(v)
    out = np.full(n, np.nan)
    if n < w:
        return out
    W = sliding_window_view(v, w)
    out[w - 1:] = W.max(axis=1) - W[:, -1]
    return out


def ema(v: np.ndarray, s: float) -> np.ndarray:
    """Smoothing. s is the smoothing coefficient."""
    if s == 0.0:
        return v
    out = np.empty_like(v)
    acc = v[0]
    out[0] = acc
    for i in range(1, len(v)):
        acc = s * acc + (1 - s) * v[i]
        out[i] = acc
    return out
    

def _first_true(mask: np.ndarray) -> int:
    """Position of the first True, or INF if there isn't one."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else INF


def drop_first_fires(v: np.ndarray, sws, dthrs, mags) -> dict:
    """First step the drop trigger fires, for every (short_window, threshold, magnitude). INF = never."""
    out = {}
    for sw in sws:
        corr = rolling_spearman(v, sw)
        mag_arr = rolling_dropmag(v, sw)
        valid = ~np.isnan(corr)  # a flat or too-short window is not a drop
        for dthr in dthrs:
            below = valid & (corr < dthr)
            for m in mags:
                mask = below if m <= 0 else (below & (mag_arr >= m))
                out[(sw, dthr, m)] = _first_true(mask)
    return out


def plateau_first_fires(v: np.ndarray, lws, pthrs) -> dict:
    """First step the plateau trigger fires, for every (long_window, threshold). INF = never."""
    out = {}
    for lw in lws:
        corr = rolling_spearman(v, lw)
        isnan = np.isnan(corr)
        ready = np.zeros(len(v), dtype=bool)
        ready[lw - 1:] = True   # can't call a plateau without a full window
        for p in pthrs:
            mask = ready & (isnan | (corr < p))
            out[(lw, p)] = _first_true(mask)
    return out


def metrics_from_fires(fire_s: np.ndarray, fire_f: np.ndarray, horizon: int) -> dict:
    """Score a setting from the step it fired at in each episode (INF = never fired)."""
    n_s, n_f = len(fire_s), len(fire_f)
    caught = fire_f < INF
    recall = float(caught.mean()) if n_f else float("nan")
    fpr = float((fire_s < INF).mean()) if n_s else float("nan")
    balacc = (recall + (1.0 - fpr)) / 2.0

    tdet = np.where(caught, fire_f / horizon, 1.0)   # a failure it never caught counts as the latest possible
    medfire = float(np.median(fire_f[caught])) if caught.any() else float("nan")
    return dict(
        recall=recall,
        fpr=fpr,
        balacc=balacc,
        avg_tdet=float(tdet.mean()),
        med_tdet=float(np.median(tdet)),
        medfire=medfire,
        n_caught=int(caught.sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="outputs/gate_frontier")
    ap.add_argument("--truncate", choices=["none", "min", "median_succ", "mean_succ"],
                    default="none",
                    help="Cut every episode to the same length.")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(os.path.dirname(args.stats))
    blob = json.load(open(args.stats))
    episodes = blob["episodes"]
    traces = [np.asarray(e["progress_trace"], dtype=np.float64) for e in episodes]
    succ = np.array([bool(e.get("success")) for e in episodes])

    if args.truncate != "none":
        lens = np.array([len(t) for t in traces])
        succ_lens = lens[succ]
        T = {"min": int(lens.min()),
             "median_succ": int(np.median(succ_lens)),
             "mean_succ": int(round(succ_lens.mean()))}[args.truncate]
        # Long episodes get cut down to T. Episodes SHORTER than T are dropped
        keep = np.array([len(t) >= T for t in traces])
        traces = [t[:T] for t, k in zip(traces, keep) if k]
        succ = succ[keep]
        print(f"[{tag}] truncate={args.truncate}: T={T}, kept {keep.sum()}/{len(keep)} episodes "
              f"({(~keep).sum()} shorter than T were dropped)")
        episodes = [e for e, k in zip(episodes, keep) if k]

    horizon = max(len(t) for t in traces)

    l_succ = float(np.mean([len(t) for t, s in zip(traces, succ) if s]))
    print(f"[{tag}] {len(episodes)} episodes: {succ.sum()} success, {(~succ).sum()} failure | "
          f"horizon={horizon}  L_succ={l_succ:.1f}")

    # build the list of settings to try
    LWS = [w for w in LW_GRID if w <= horizon]

    drop_keys = [None] + list(itertools.product(SWS, DTHRS, MAGS))       # None = drop trigger off
    plat_keys = [None] + list(itertools.product(LWS, PTHRS))             # None = plateau trigger off

    D = np.full((len(SMOOTH), len(episodes), len(drop_keys)), INF, dtype=np.int64)   # drop
    P = np.full((len(SMOOTH), len(episodes), len(plat_keys)), INF, dtype=np.int64)   # plateau
    for si, s in enumerate(SMOOTH):
        for ei, v in enumerate(traces):
            vs = ema(v, s)
            df = drop_first_fires(vs, SWS, DTHRS, MAGS)
            pf = plateau_first_fires(vs, LWS, PTHRS)
            for ki, k in enumerate(drop_keys):
                if k is not None:
                    D[si, ei, ki] = df[k]
            for ki, k in enumerate(plat_keys):
                if k is not None:
                    P[si, ei, ki] = pf[k]
        print(f"  smoothing={s}: replayed {len(episodes)} episodes "
              f"({len(drop_keys)} drop x {len(plat_keys)} plateau configs)")

    # score every setting
    rows = []
    for si, s in enumerate(SMOOTH):
        F = np.minimum(D[si][:, :, None], P[si][:, None, :])
        Fs, Ff = F[succ], F[~succ]
        caught = Ff < INF
        recall = caught.mean(axis=0)
        fpr = (Fs < INF).mean(axis=0)
        balacc = (recall + 1.0 - fpr) / 2.0  # balanced accuracy
        tdet = np.where(caught, Ff / horizon, 1.0).mean(axis=0)  # normalized detection time (false negatives count as 1)
        medfire = np.full(recall.shape, np.nan)
        for i in range(F.shape[1]):
            for j in range(F.shape[2]):
                c = caught[:, i, j]
                if c.any():
                    medfire[i, j] = np.median(Ff[c, i, j])

        for i, dk in enumerate(drop_keys):
            for j, pk in enumerate(plat_keys):
                if dk is None and pk is None:
                    continue
                rows.append(dict(
                    family="gate" if (dk and pk) else ("drop_only" if dk else "plateau_only"),
                    smoothing=s,
                    sw=dk[0] if dk else None, dthr=dk[1] if dk else None, mag=dk[2] if dk else None,
                    lw=pk[0] if pk else None, pthr=pk[1] if pk else None,
                    recall=float(recall[i, j]), fpr=float(fpr[i, j]),
                    balacc=float(balacc[i, j]), avg_tdet=float(tdet[i, j]),
                    medfire=float(medfire[i, j]),
                ))

    # Naive baselines
    lens = np.array([len(t) for t in traces])
    # Timeout: always fire at step T, whatever the curve is doing.
    for T in range(25, horizon + 1, 25):
        fire = np.where(lens > T, T, INF)
        rows.append(dict(family="timeout", T=T,
                         **metrics_from_fires(fire[succ], fire[~succ], horizon)))
                         
    # Absolute: fire the first time progress drops below theta, ignoring the first `delay` steps.
    for theta in np.round(np.arange(0.05, 1.0, 0.05), 2):
        for delay in [100, 150, 200, 250, 300, 400]:
            fire = np.array([
                _first_true((np.arange(len(t)) >= delay) & (t < theta)) for t in traces])
            rows.append(dict(family="absolute", theta=float(theta), delay=delay,
                             **metrics_from_fires(fire[succ], fire[~succ], horizon)))

    for r in rows:
        r.setdefault("recall", np.nan)
    out = dict(tag=tag, horizon=horizon, l_succ=l_succ,
               n_success=int(succ.sum()), n_failure=int((~succ).sum()), rows=rows)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{tag}.json")
    json.dump(out, open(path, "w"))
    print(f"  -> {len(rows)} configs written to {path}")


if __name__ == "__main__":
    main()
