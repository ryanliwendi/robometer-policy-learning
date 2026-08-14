#!/usr/bin/env python3
"""Evaluation: does a gate config tuned on one task still work on another task?

Two modes, both scored by the same number: the LATENCY PENALTY at matched balanced accuracy. A
transferred config's detection time is divided by the detection time of the fastest native config
that is at least as accurate.

  --mode 1   Tune on ONE task, test on the other three. 
  --mode 2   Tune on THREE tasks jointly, test on the fourth.

Both modes need a delta: how much balanced accuracy the chosen config may give up against the best
config on the tuning task(s). Each mode is run at several deltas and the one with the lowest mean
penalty is reported in full.

Usage:
    uv run python scripts/gate_transfer.py     # both modes, default deltas
    uv run python scripts/gate_transfer.py --mode 2
    uv run python scripts/gate_transfer.py --mode 1 --deltas 0 0.01 0.02
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
    "t4": "gated_videos/n200_t4/episode_stats.json",  # only used by --extra
}
TASKS = ["t0", "t1", "t5", "t8"]
FRONT_DIR = os.environ.get("GATE_FRONT_DIR", "outputs/gate_frontier_n200")
DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05]  # balanced accuracy `slack` levels when picking a config


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


def native_at(front, balacc):
    """Detection time of the fastest config on `front` that is at least as accurate as `balacc`."""
    ok = [r["avg_tdet"] for r in front if r["balacc"] >= balacc - 1e-9]
    return min(ok) if ok else float("nan")


def cfg_key(r):
    """The six numbers that identify a config, used to match the same config across tasks."""
    return (r.get("smoothing") or 0.0, r.get("sw"), r.get("dthr"),
            r.get("mag"), r.get("lw"), r.get("pthr"))


_ROWS = {}


def load_rows(tag, family="gate"):
    """All swept configs for one task, as a dict of config -> its measured numbers on that task."""
    if (tag, family) not in _ROWS:
        blob = json.load(open(os.path.join(FRONT_DIR, f"{tag}_dp.json")))
        _ROWS[(tag, family)] = {cfg_key(r): r for r in blob["rows"]
                                if r["family"] == family and np.isfinite(r.get("balacc", np.nan))}
    return _ROWS[(tag, family)]


def joint_best_config(tags, family="gate", slack=0.0):
    """One config that stays within `slack` of the best balanced accuracy on every task in `tags`.

    Among the configs that pass on all of them, take the one with the lowest mean detection time.
    Returns None when no single config passes on all tasks (happens at slack 0).
    """
    tabs = [load_rows(t, family) for t in tags]
    tops = [max(r["balacc"] for r in tab.values()) for tab in tabs]
    keys = set(tabs[0])
    for tab in tabs[1:]:
        keys &= set(tab)
    ok = [k for k in keys
          if all(tab[k]["balacc"] >= top - slack - 1e-9 for tab, top in zip(tabs, tops))]
    if not ok:
        return None
    best = min(ok, key=lambda k: float(np.mean([tab[k]["avg_tdet"] for tab in tabs])))
    return tabs[0][best]


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
    """MODE 1 table: each task's chosen best config (under slack) run on all four tasks.

    Each row carries two comparisons:
      * against the target's own config with the same pre-transfer BA level; measures both drop in BA and latency
      * `penalty`, against the fastest native config with the same post-transfer BA level
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
            nat = native_at(fronts[t], m["balacc"])
            rows.append(dict(
                source=s, target=t, cfg=cfg_str(cfg), lw=cfg.get("lw"),
                tpr=m["recall"], tnr=1.0 - m["fpr"], balacc=m["balacc"],
                avg_tdet=m["avg_tdet"], medfire=m["medfire"],
                penalty=100 * (m["avg_tdet"] / nat - 1), native_at_balacc=nat,
                own_tpr=own["recall"], own_tnr=1.0 - own["fpr"], own_balacc=own["balacc"],
                own_tdet=own["avg_tdet"], own_medfire=own.get("medfire", float("nan")),
            ))
    return rows


def holdout_matrix(corpora, slack=0.0):
    """MODE 2 table: tune on three tasks jointly, test on the fourth. One row per held-out task."""
    fronts = {t: load_front(t) for t in TASKS}
    rows = []
    for t in TASKS:
        calib = [x for x in TASKS if x != t]
        cfg_row = joint_best_config(calib, slack=slack)
        if cfg_row is None:
            rows.append(dict(target=t, calib=calib, cfg=None, penalty=float("nan")))
            continue
        cfg = as_cfg(cfg_row)
        m = corpora[t].evaluate(cfg)
        nat = native_at(fronts[t], m["balacc"])
        own = best_config(t, slack=slack)
        # How far the config fell below each tuning task's best balanced accuracy. All of these are
        # <= slack by construction; printing them shows how much of the budget was actually spent.
        tabs = {c: load_rows(c) for c in calib}
        gaps = {c: max(r["balacc"] for r in tabs[c].values()) - tabs[c][cfg_key(cfg_row)]["balacc"]
                for c in calib}
        rows.append(dict(
            target=t, calib=calib, cfg=cfg_str(cfg), lw=cfg.get("lw"),
            tpr=m["recall"], tnr=1.0 - m["fpr"], balacc=m["balacc"],
            avg_tdet=m["avg_tdet"], medfire=m["medfire"],
            penalty=100 * (m["avg_tdet"] / nat - 1), native_at_balacc=nat,
            calib_gap=max(gaps.values()),
            own_balacc=own["balacc"], own_tdet=own["avg_tdet"],
        ))
    return rows


def all4_matrix(corpora, slack=0.0):
    """The config tuned jointly on all four tasks, run on each of those four tasks.

    These are in-sample numbers: the config was fitted on these tasks. A penalty above zero means
    the one shared config is off a task's own frontier because it has to suit all four at once.
    """
    cfg_row = joint_best_config(TASKS, slack=slack)
    if cfg_row is None:
        return []
    cfg = as_cfg(cfg_row)
    rows = []
    for t in TASKS:
        m = corpora[t].evaluate(cfg)
        nat = native_at(load_front(t), m["balacc"])
        own = best_config(t, slack=slack)
        rows.append(dict(
            target=t, cfg=cfg_str(cfg), lw=cfg.get("lw"),
            tpr=m["recall"], tnr=1.0 - m["fpr"], balacc=m["balacc"],
            avg_tdet=m["avg_tdet"], medfire=m["medfire"],
            penalty=100 * (m["avg_tdet"] / nat - 1), native_at_balacc=nat,
            own_balacc=own["balacc"], own_tdet=own["avg_tdet"],
        ))
    return rows


def print_all4_matrix(rows, label):
    """Prints the one shared config's numbers on each of the four tasks it was tuned on."""
    if not rows:
        print("\n  no config stays within slack on all four tasks")
        return
    print(f"\n=== config tuned on all 4 tasks, run on each of them (in-sample), {label} ===")
    print(f"  config: {rows[0]['cfg']}")
    hdr = (f"  {'task':<8}{'TPR':>7}{'TNR':>7}{'balacc':>8}{'tdet':>7}{'medfire':>9}"
           f"{'native':>8}{'pen%':>7}   retuned on itself")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        mf = f"{r['medfire']:.0f}" if np.isfinite(r["medfire"]) else "-"
        print(f"  {r['target']:<8}{r['tpr']:>7.3f}{r['tnr']:>7.3f}{r['balacc']:>8.3f}"
              f"{r['avg_tdet']:>7.3f}{mf:>9}{r['native_at_balacc']:>8.3f}{r['penalty']:>7.1f}"
              f"   ba {r['own_balacc']:.3f}  tdet {r['own_tdet']:.3f}")
    pens = [r["penalty"] for r in rows]
    print(f"\n  LATENCY PENALTY AT MATCHED BALACC: mean {np.mean(pens):+.1f}%  "
          f"median {np.median(pens):+.1f}%  worst {max(pens):+.1f}%")


def external_matrix(corpora, tag, slack=0.0):
    """Test on a task that took no part in tuning, and no part in choosing delta.

    Two things are run on `tag`: the config tuned jointly on all four tasks in TASKS, and the four
    leave-one-out configs from mode 2. The first is the one you would actually ship; the other four
    show how much the answer moves when the tuning set changes.
    """
    front = load_front(tag)
    own = best_config(tag, slack=slack)
    cands = [("all 4", joint_best_config(TASKS, slack=slack))]
    for t in TASKS:
        cands.append((f"all but {t}", joint_best_config([x for x in TASKS if x != t], slack=slack)))
    rows = []
    for name, cfg_row in cands:
        if cfg_row is None:
            rows.append(dict(tuned_on=name, target=tag, cfg=None, penalty=float("nan")))
            continue
        cfg = as_cfg(cfg_row)
        m = corpora[tag].evaluate(cfg)
        nat = native_at(front, m["balacc"])
        rows.append(dict(
            tuned_on=name, target=tag, cfg=cfg_str(cfg), lw=cfg.get("lw"),
            tpr=m["recall"], tnr=1.0 - m["fpr"], balacc=m["balacc"],
            avg_tdet=m["avg_tdet"], medfire=m["medfire"],
            penalty=100 * (m["avg_tdet"] / nat - 1), native_at_balacc=nat,
            own_balacc=own["balacc"], own_tdet=own["avg_tdet"],
        ))
    return rows


def print_external_matrix(rows, tag, label):
    """Prints the --extra table: several tuning sets, all tested on the same unseen task."""
    print(f"\n=== transfer onto {tag} (never tuned on, never used to pick delta), {label} ===")
    hdr = (f"  {'tuned on':<12}{'TPR':>7}{'TNR':>7}{'balacc':>8}{'tdet':>7}{'medfire':>9}"
           f"{'native':>8}{'pen%':>7}   config")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        if r["cfg"] is None:
            print(f"  {r['tuned_on']:<12}   no config stays within slack on every tuning task")
            continue
        mf = f"{r['medfire']:.0f}" if np.isfinite(r["medfire"]) else "-"
        print(f"  {r['tuned_on']:<12}{r['tpr']:>7.3f}{r['tnr']:>7.3f}{r['balacc']:>8.3f}"
              f"{r['avg_tdet']:>7.3f}{mf:>9}{r['native_at_balacc']:>8.3f}{r['penalty']:>7.1f}"
              f"   {r['cfg']}")
    got = [r for r in rows if np.isfinite(r["penalty"])]
    if got:
        print(f"\n  {tag} retuned on itself: balacc {got[0]['own_balacc']:.3f}  "
              f"tdet {got[0]['own_tdet']:.3f}")
        d_ba = np.mean([r["balacc"] - r["own_balacc"] for r in got])
        d_td = np.mean([r["avg_tdet"] - r["own_tdet"] for r in got])
        d_td_pct = 100 * np.mean([r["avg_tdet"] / r["own_tdet"] - 1 for r in got])
        print(f"  COST OF NOT RETUNING: balacc {d_ba:+.3f}   tdet {d_td:+.3f} ({d_td_pct:+.0f}%)")
        pens = [r["penalty"] for r in got]
        print(f"  LATENCY PENALTY AT MATCHED BALACC: mean {np.mean(pens):+.1f}%  "
              f"median {np.median(pens):+.1f}%  worst {max(pens):+.1f}%")


def print_holdout_matrix(rows, label):
    """Prints the mode 2 table: one row per held-out task."""
    print(f"\n=== leave-one-task-out transfer, {label} ===")
    hdr = (f"  {'held out':<10}{'TPR':>7}{'TNR':>7}{'balacc':>8}{'tdet':>7}{'medfire':>9}"
           f"{'native':>8}{'pen%':>7}{'gap':>7}   config (tuned on the other 3)")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        if r["cfg"] is None:
            print(f"  {r['target']:<10}   no config stays within slack on all of "
                  f"{', '.join(r['calib'])}")
            continue
        mf = f"{r['medfire']:.0f}" if np.isfinite(r["medfire"]) else "-"
        print(f"  {r['target']:<10}{r['tpr']:>7.3f}{r['tnr']:>7.3f}{r['balacc']:>8.3f}"
              f"{r['avg_tdet']:>7.3f}{mf:>9}{r['native_at_balacc']:>8.3f}{r['penalty']:>7.1f}"
              f"{r['calib_gap']:>7.3f}   {r['cfg']}")
    got = [r for r in rows if np.isfinite(r["penalty"])]
    if got:
        d_ba = np.mean([r["balacc"] - r["own_balacc"] for r in got])
        d_td = np.mean([r["avg_tdet"] - r["own_tdet"] for r in got])
        d_td_pct = 100 * np.mean([r["avg_tdet"] / r["own_tdet"] - 1 for r in got])
        # vs the held-out task's own shipped config: which operating point you ended up on.
        print(f"\n  COST OF NOT RETUNING: balacc {d_ba:+.3f}   tdet {d_td:+.3f} ({d_td_pct:+.0f}%)")
        pens = [r["penalty"] for r in got]
        print(f"  LATENCY PENALTY AT MATCHED BALACC: mean {np.mean(pens):+.1f}%  "
              f"median {np.median(pens):+.1f}%  worst {max(pens):+.1f}%")


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


def transferred_rows(rows, mode):
    """The rows that were actually transferred: mode 1 drops the diagonal, and every mode drops the
    tasks where no config was feasible."""
    return [r for r in rows
            if (mode != 1 or r["source"] != r["target"]) and np.isfinite(r["penalty"])]


def sweep_deltas(corpora, deltas, mode, extra=None):
    """Run one mode at several deltas. Prints one line per delta and returns the rows of each.

    pen% is the latency penalty at matched balanced accuracy. dBA and dtdet are the cost of not
    retuning: the transferred config minus the target's own config, before transfer.
    """
    per_delta = {}
    name = f"mode {mode}" if mode != 3 else f"transfer onto {extra}"
    print(f"\n=== {name} delta sweep ===")
    print(f"  {'delta':>7}{'mean%':>9}{'median%':>9}{'worst%':>9}{'n':>5}{'dBA':>9}{'dtdet':>9}")
    for d in deltas:
        if mode == 1:
            rows = best_config_matrix(corpora, slack=d)
        elif mode == 2:
            rows = holdout_matrix(corpora, slack=d)
        else:
            rows = external_matrix(corpora, extra, slack=d)
        per_delta[d] = rows
        got = transferred_rows(rows, mode)
        if not got:
            print(f"  {d:>7.3f}{'-':>9}{'-':>9}{'-':>9}{0:>5}{'-':>9}{'-':>9}   no feasible config")
            continue
        pens = [r["penalty"] for r in got]
        d_ba = np.mean([r["balacc"] - r["own_balacc"] for r in got])
        d_td = np.mean([r["avg_tdet"] - r["own_tdet"] for r in got])
        print(f"  {d:>7.3f}{np.mean(pens):>9.1f}{np.median(pens):>9.1f}"
              f"{max(pens):>9.1f}{len(pens):>5}{d_ba:>+9.3f}{d_td:>+9.3f}")
    return per_delta


def mean_penalty(rows, mode):
    got = transferred_rows(rows, mode)
    return float(np.mean([r["penalty"] for r in got])) if got else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["1", "2", "both"],
                    help="1 = tune on one task, test on the other three. "
                         "2 = tune on three tasks jointly, test on the fourth.")
    ap.add_argument("--deltas", nargs="*", type=float, default=DELTAS,
                    help="balanced accuracy budgets to try when picking a config")
    ap.add_argument("--extra", default=None,
                    help="also test on this task, which is not in TASKS. It took no part in "
                         "tuning and no part in picking delta, so it checks the whole recipe.")
    ap.add_argument("--fix-delta", type=float, default=None,
                    help="print the detail table at this delta instead of the lowest-penalty one")
    ap.add_argument("--save", default=None, help="write the winning delta's rows to this JSON")
    args = ap.parse_args()

    tags = TASKS + ([args.extra] if args.extra else [])
    corpora = {t: Corpus(t) for t in tags}
    print("Task horizons (L_succ = mean successful-episode length)")
    for t in tags:
        print(f"  {t}: {corpora[t].l_succ:6.1f}   "
              f"({int(corpora[t].succ.sum())}S/{int((~corpora[t].succ).sum())}F)")

    modes = [1, 2] if args.mode == "both" else [int(args.mode)]
    if args.extra:
        modes.append(3)
    out = {}
    for mode in modes:
        per_delta = sweep_deltas(corpora, args.deltas, mode, extra=args.extra)
        best_d = (args.fix_delta if args.fix_delta is not None
                  else min(per_delta, key=lambda d: mean_penalty(per_delta[d], mode)))
        rows = per_delta.get(best_d)
        if rows is None:  # --fix-delta asked for a delta that was not swept
            rows = per_delta[min(per_delta, key=lambda d: abs(d - best_d))]
        label = (f"delta {best_d:g}" if args.fix_delta is not None
                 else f"delta {best_d:g} (lowest mean penalty)")
        if mode == 1:
            print_best_matrix(rows, label)
        elif mode == 2:
            print_holdout_matrix(rows, label)
        else:
            print_external_matrix(rows, args.extra, label)
            in_sample = all4_matrix(corpora, slack=best_d)
            print_all4_matrix(in_sample, label)
            out["all4_in_sample"] = dict(delta=best_d, rows=in_sample)
        out[f"mode{mode}"] = dict(delta=best_d, rows=rows)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        json.dump(out, open(args.save, "w"), indent=1)
        print(f"\n-> {args.save}")


if __name__ == "__main__":
    main()
