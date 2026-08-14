#!/usr/bin/env python3
"""Evaluation: does a gate config tuned on one task still work on another task?

There are two separate modes here:

  no flags   Runs the source's whole frontier (~50 configs) on the target. Prints the median
             latency penalty. `--slack` is unused.
  --best     Runs ONE config per task, picked by `best_config` under `--slack`, on all four tasks.
             Prints TPR, TNR, balacc, tdet, medfire.

Penalty is measured at matched balanced accuracy: a transferred config's detection time against
the fastest native config that is at least as accurate.

Usage:
    uv run python scripts/gate_transfer.py                     # whole-frontier transfer, 4x4
    uv run python scripts/gate_transfer.py --pair t1 t8        # every config, one pair
    uv run python scripts/gate_transfer.py --best --slack 0.015  # one config per task, 4x4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_gate_configs import INF, ema, rolling_dropmag, rolling_spearman, _first_true  # noqa: E402

CORPORA = {
    "t0": "gated_videos/n200_t0_dp/episode_stats.json",
    "t1": "gated_videos/n200_t1_dp/episode_stats.json",
    "t5": "gated_videos/n200_t5_dp/episode_stats.json",
    "t8": "gated_videos/n200_t8_dp/episode_stats.json",
}
TASKS = ["t0", "t1", "t5", "t8"]
FRONT_DIR = os.environ.get("GATE_FRONT_DIR", "outputs/gate_frontier_n200")


class Corpus:
    """Stores info for one task's rollouts, plus a cache for the ema, correlation, and mag values."""

    def __init__(self, tag):
        blob = json.load(open(CORPORA[tag]))
        self.tag = tag
        self.traces = [np.asarray(e["progress_trace"], dtype=np.float64) for e in blob["episodes"]]
        self.succ = np.array([bool(e.get("success")) for e in blob["episodes"]])
        self.horizon = max(len(t) for t in self.traces)
        self.l_succ = float(np.mean([len(t) for t, s in zip(self.traces, self.succ) if s]))
        self._ema, self._corr, self._mag = {}, {}, {}

    def _series(self, ei, s):
        """Return an episode's smoothed values, with traces `ei` and smoothing coefficient `s`."""
        if (ei, s) not in self._ema:
            self._ema[(ei, s)] = ema(self.traces[ei], s)
        return self._ema[(ei, s)]

    def corr(self, ei, s, w):
        """For one episode smoothed by `s`, the correlation at every step over the last `w` steps."""
        if (ei, s, w) not in self._corr:
            self._corr[(ei, s, w)] = rolling_spearman(self._series(ei, s), w)
        return self._corr[(ei, s, w)]

    def mag(self, ei, s, w):
        """For one episode smoothed by `s`, the max value minus the current value at every step over the last `w` steps"""
        if (ei, s, w) not in self._mag:
            self._mag[(ei, s, w)] = rolling_dropmag(self._series(ei, s), w)
        return self._mag[(ei, s, w)]

    def evaluate(self, cfg):
        """Evaluate the given config with metrics like recall, fpr, etc."""
        fires = np.full(len(self.traces), INF, dtype=np.int64)
        s = cfg.get("smoothing") or 0.0
        for ei in range(len(self.traces)):
            d = INF
            if cfg.get("sw"):
                c = self.corr(ei, s, cfg["sw"])
                m = (~np.isnan(c)) & (c < cfg["dthr"])
                if cfg.get("mag"):
                    m &= self.mag(ei, s, cfg["sw"]) >= cfg["mag"]
                d = _first_true(m)
            p = INF
            if cfg.get("lw"):
                lw = cfg["lw"]
                c = self.corr(ei, s, lw)
                ready = np.zeros(len(c), dtype=bool)
                if lw - 1 < len(c):
                    ready[lw - 1:] = True
                p = _first_true(ready & (np.isnan(c) | (c < cfg["pthr"])))
            fires[ei] = min(d, p)
        fs, ff = fires[self.succ], fires[~self.succ]
        caught = ff < INF
        recall = float(caught.mean()) if len(ff) else float("nan")
        fpr = float((fs < INF).mean()) if len(fs) else float("nan")
        return dict(recall=recall, fpr=fpr, balacc=(recall + 1.0 - fpr) / 2.0,
                    avg_tdet=float(np.where(caught, ff / self.horizon, 1.0).mean()),
                    medfire=float(np.median(ff[caught])) if caught.any() else float("nan"))


def load_front(tag):
    """Native pareto front (balacc up, avg_tdet down), gate family only."""
    blob = json.load(open(os.path.join(FRONT_DIR, f"{tag}_dp.json")))
    rows = [r for r in blob["rows"]
            if r["family"] in ("gate", "drop_only", "plateau_only")
            and np.isfinite(r.get("balacc", np.nan))]  # List of configs
    rows.sort(key=lambda r: (r["avg_tdet"], -r["balacc"]))
    front, best = [], -np.inf
    for r in rows:
        if r["balacc"] > best + 1e-12:  # Beats all faster configs on balanced accuracy
            best = r["balacc"]
            front.append(r)
    return front


def best_config(tag, family="gate", slack=0.0):
    """The config on the pareto frontier, with balanced accuracy being `slack` away from the best
    balanced accuracy.
    
    Allowing a little slack matters because the most accurate configs often overoptimizes for balanced
    accuracy and transfer to other tasks with a large latency cost. Giving up
    a few points of accuracy is enough to limit latency to a <10%.
    """
    blob = json.load(open(os.path.join(FRONT_DIR, f"{tag}_dp.json")))
    rows = [r for r in blob["rows"]
            if r["family"] == family and np.isfinite(r.get("balacc", np.nan))]
    if not rows:
        raise ValueError(f"no {family} rows for {tag}")
    top = max(r["balacc"] for r in rows)
    return min([r for r in rows if r["balacc"] >= top - slack - 1e-9],
               key=lambda r: r["avg_tdet"])


def best_config_matrix(corpora, slack=0.0):
    """Build the 4x4 table: each task's chosen best config (under slack) run on all four tasks.

    Each row carries two comparisons:
      * against the target's OWN shipped config -- what you lose by not retuning
      * `penalty`, against the fastest native config that is at least as accurate as what the
        transferred config actually reached.
    """
    fronts = {t: load_front(t) for t in TASKS}
    rows = []
    for s in TASKS:
        cfg_row = best_config(s, slack=slack)
        cfg = as_cfg(cfg_row)
        for t in TASKS:
            tgt = corpora[t]
            m = tgt.evaluate(cfg)
            own = best_config(t, slack=slack)
            ok = [r["avg_tdet"] for r in fronts[t] if r["balacc"] >= m["balacc"] - 1e-9]
            nat = min(ok) if ok else float("nan")
            rows.append(dict(
                source=s, target=t, cfg=cfg_str(cfg), lw=cfg.get("lw"),
                tpr=m["recall"], tnr=1.0 - m["fpr"], balacc=m["balacc"],
                avg_tdet=m["avg_tdet"], medfire=m["medfire"],
                penalty=100 * (m["avg_tdet"] / nat - 1), native_at_balacc=nat,
                own_tpr=own["recall"], own_tnr=1.0 - own["fpr"], own_balacc=own["balacc"],
                own_tdet=own["avg_tdet"], own_medfire=own.get("medfire", float("nan")),
            ))
    return rows


def print_best_matrix(rows, label):
    """Prints the 4*4 best task transfer matrix."""
    print(f"\n=== best config transfer, {label} ===")
    hdr = (f"  {'src->tgt':<12}{'TPR':>7}{'TNR':>7}{'balacc':>8}{'tdet':>7}{'medfire':>9}"
           f"{'pen%':>7}   config")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for t in TASKS:
        for r in [x for x in rows if x["target"] == t]:
            mark = " *" if r["source"] == r["target"] else "  "
            mf = f"{r['medfire']:.0f}" if np.isfinite(r["medfire"]) else "-"
            print(f"  {r['source']+'->'+r['target']:<12}{r['tpr']:>7.3f}{r['tnr']:>7.3f}"
                  f"{r['balacc']:>8.3f}{r['avg_tdet']:>7.3f}{mf:>9}{r['penalty']:>7.1f}"
                  f"{mark} {r['cfg']}")
        print()
    off = [r for r in rows if r["source"] != r["target"]]
    print(f"  transferred (off-diagonal, n={len(off)}): "
          f"balacc {np.mean([r['balacc'] for r in off]):.3f}  "
          f"tdet {np.mean([r['avg_tdet'] for r in off]):.3f}")
    print(f"  native      (diagonal,     n={len(TASKS)}): "
          f"balacc {np.mean([r['own_balacc'] for r in rows if r['source']==r['target']]):.3f}  "
          f"tdet {np.mean([r['own_tdet'] for r in rows if r['source']==r['target']]):.3f}")
    d_balacc = np.mean([r["balacc"] - r["own_balacc"] for r in off])
    d_tdet = np.mean([r["avg_tdet"] - r["own_tdet"] for r in off])
    d_tdet_pct = 100 * np.mean([r["avg_tdet"] / r["own_tdet"] - 1 for r in off])
    pens = [r["penalty"] for r in off]
    # vs the target's own shipped config: which operating point you ended up on.
    print(f"  COST OF NOT RETUNING: balacc {d_balacc:+.3f}   "
          f"tdet {d_tdet:+.3f} ({d_tdet_pct:+.0f}%)")
    # vs the target's frontier at the accuracy you actually reached: how far off it you are.
    print(f"  LATENCY PENALTY AT MATCHED BALACC: mean {np.mean(pens):+.1f}%  "
          f"median {np.median(pens):+.1f}%  worst {max(pens):+.1f}%")


def as_cfg(r):
    return dict(smoothing=r.get("smoothing") or 0.0, sw=r.get("sw"), dthr=r.get("dthr"),
                mag=r.get("mag"), lw=r.get("lw"), pthr=r.get("pthr"))


def cfg_str(c):
    p = []
    if c.get("sw"):
        p.append(f"sw{c['sw']}/d{c['dthr']}/m{c['mag']}")
    if c.get("lw"):
        p.append(f"lw{c['lw']}/p{c['pthr']}")
    if c.get("smoothing"):
        p.append(f"s{c['smoothing']}")
    return " ".join(p) or "(none)"


def pareto(rows, ykey="balacc", xkey="avg_tdet"):
    rs = sorted(rows, key=lambda r: (r[xkey], -r[ykey]))
    front, best = [], -np.inf
    for r in rs:
        if r[ykey] > best + 1e-12:
            best = r[ykey]
            front.append(r)
    return front


def transfer(src, tgt_corpus, verbose=False):
    """Run every config on src's frontier against tgt, and return one dict per config."""
    src_front = load_front(src)
    native_front = load_front(tgt_corpus.tag)

    def native_at(b):
        """Detection time of the fastest native config that is at least as accurate as `b`."""
        ok = [r["avg_tdet"] for r in native_front if r["balacc"] >= b - 1e-9]
        return min(ok) if ok else float("nan")

    rows = []
    for r in src_front:
        c = as_cfg(r)
        m = tgt_corpus.evaluate(c)
        rows.append(dict(
            src_balacc=r["balacc"], src_tdet=r["avg_tdet"], cfg=c,
            **m, penalty=100 * (m["avg_tdet"] / native_at(m["balacc"]) - 1)))
    if verbose:
        print(f"  {'lw':>6} | {'balacc':>7}{'FPR':>6}{'tdet':>7}{'pen%':>6}   config")
        for r in rows:
            print(f"  {str(r['cfg'].get('lw')):>6} | {r['balacc']:>7.3f}{r['fpr']:>6.3f}"
                  f"{r['avg_tdet']:>7.3f}{r['penalty']:>6.0f}   {cfg_str(r['cfg'])}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--best", action="store_true",
                    help="transfer matrix for one config per task (TPR/TNR/medfire)")
    ap.add_argument("--save", default=None, help="write the --best rows to this JSON")
    ap.add_argument("--slack", type=float, default=0.0,
                    help="how much balanced accuracy to give up when picking each task's config. "
                         "Only used with --best.")
    args = ap.parse_args()

    corpora = {t: Corpus(t) for t in TASKS}

    # Mode 1: one config per task, chosen by `best_config` under `--slack`
    if args.best:
        rows = best_config_matrix(corpora, slack=args.slack)
        print_best_matrix(rows, f"slack {args.slack:g}")
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            json.dump(rows, open(args.save, "w"), indent=1)
            print(f"\n-> {args.save}")
        return

    # Mode 2: the whole frontier, about 50 configs per source.
    print("Task horizons (L_succ = mean successful-episode length)")
    for t in TASKS:
        print(f"  {t}: {corpora[t].l_succ:6.1f}   "
              f"({int(corpora[t].succ.sum())}S/{int((~corpora[t].succ).sum())}F)")
    print()

    if args.pair:
        s, t = args.pair
        print(f"=== {s} -> {t}")
        transfer(s, corpora[t], verbose=True)
        return

    M = np.full((len(TASKS), len(TASKS)), np.nan)
    for i, s in enumerate(TASKS):
        for j, t in enumerate(TASKS):
            if s != t:
                M[i, j] = np.nanmedian([r["penalty"] for r in transfer(s, corpora[t])])

    print("=== median matched-bal-acc latency penalty % ===")
    hdr = "src\\tgt"
    print(f"  {hdr:<9}" + "".join(f"{t:>8}" for t in TASKS) + f"{'row med':>10}")
    for i, s in enumerate(TASKS):
        cells = "".join("       -" if np.isnan(M[i, j]) else f"{M[i, j]:>8.0f}"
                        for j in range(len(TASKS)))
        print(f"  {s:<9}{cells}{np.nanmedian(M[i]):>10.0f}")
    print(f"  {'col med':<9}" + "".join(f"{np.nanmedian(M[:, j]):>8.0f}" for j in range(len(TASKS)))
          + f"{np.nanmedian(M):>10.0f}")

    off = ~np.eye(len(TASKS), dtype=bool)
    print(f"\nOVERALL: median {np.nanmedian(M[off]):+.0f}%  "
          f"mean {np.nanmean(M[off]):+.0f}%  worst {np.nanmax(M[off]):+.0f}%")


if __name__ == "__main__":
    main()
