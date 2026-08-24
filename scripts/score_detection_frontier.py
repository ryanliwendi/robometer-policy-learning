#!/usr/bin/env python3
"""Turn recorded signal traces into the threshold-by-threshold rows the frontier figure is drawn
from. This is the compute half of the pair; `plot_detection_frontier.py` reads what it writes.

Every gate has one knob that trades firing early against firing on episodes that were going to
succeed anyway. ThriftyDAgger, Diff-DAgger and UCF call it alpha, roughly "what fraction of steps
should the gate fire on": none of them sets a threshold directly, they look at the scores from
earlier episodes and pick the cutoff that only the top alpha fraction beat. Careful with UCF: its
own config states the same knob the other way round, as the QUANTILE it keeps (0.95), so its
`ucf_alpha: 0.95` is alpha = 0.05 here. LogpZO's alpha is the width of a
per-timestep band instead of a single cutoff. Sweeping alpha gives a curve -- low alpha fires
rarely and late, high alpha fires often and early -- and this script measures every point on it,
for every family, so the curves can be put on one pair of axes.

Nothing is re-run on the robot. score_signals_offline.py already recorded, for every step of every
episode, our progress score and each baseline's score. Because they came from the same episodes,
the curves can be compared episode by episode. How a curve is scored (balanced accuracy, how early
it fires, which points are best) is imported from sweep_gate_configs.py so every script here
measures the same way.

Each setting is scored two ways, because the real question is which threshold to actually use:

  insample  pick the cutoff using all the episodes, then test on those same episodes. Too
            optimistic, since the cutoff already saw the steps it is being tested on.
  cv        split the episodes into K groups. Pick the cutoff from K-1 of them, test on the one
            left out, repeat. This is what really happens on the robot -- the threshold is set
            from the episodes collected so far and used on the next one -- so quote these numbers
            and choose alpha from them.

One asymmetry worth knowing: the baselines pick their cutoff without ever looking at whether an
episode succeeded, but our gate's settings are chosen by comparing against success labels. To keep
that fair, our gate's cv rows re-pick their settings inside each split, under a cap on how often
they're allowed to fire on a successful episode.

Usage:
    uv run python scripts/score_detection_frontier.py gated_videos/sigq_t1/signal_traces.json
    uv run python scripts/score_detection_frontier.py gated_videos/sigq_t*/signal_traces.json \
        --truncate min --out-dir outputs/thrifty_frontier
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep_gate_configs import (  # noqa: E402
    DTHRS, INF, LW_GRID, MAGS, PTHRS, SMOOTH, SWS, drop_first_fires, ema,
    metrics_from_fires, plateau_first_fires)
from gate_transfer import pareto  # noqa: E402
from robometer_policy_learning.utils.logpzo_gate import conformal_band, pad_to  # noqa: E402

# For our gate, the knob that plays the same role as alpha is "how often am I willing to fire on a
# successful episode". These are the caps we try. The list goes all the way to 1.0 on purpose: if
# we stopped at 0.5 our gate would never get to try the fire-early-and-often settings that
# Thrifty's high alphas cover, and the two curves would not span the same range.
FPR_BUDGETS = sorted({0.02} | {round(0.05 * i, 2) for i in range(21)})

# score_signals_offline.py marks episodes it built LogpZO from with this group number, so they can
# be dropped from every method's numbers at once.
HELD_IN = -2


def alpha_grid() -> np.ndarray:
    """The alphas we try, spread evenly on a log scale, plus the two the configs really use."""
    g = list(np.geomspace(1e-4, 0.5, 41))
    return np.unique(np.round(sorted(g + [0.001, 0.01]), 8))


# ---------------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------------


class SignalCorpus:
    """The episodes from one signal_traces.json file, after cutting them all to the same length.

    Cutting them to the same length matters. In LIBERO a success stops as soon as the task is
    done, while a failure runs until the 800-step cap. So if you leave the lengths alone, a
    detector that does nothing but count steps looks almost perfect. Trimming every episode to the
    same number of steps takes that shortcut away.
    """

    def __init__(self, path: str, truncate: str = "min"):
        blob = json.load(open(path))
        self.path = path
        self.meta = blob["meta"]
        self.tag = os.path.basename(os.path.dirname(os.path.abspath(path)))
        self.names = list(self.meta["signals"])

        eps = blob["episodes"]
        succ = np.array([bool(e["success"]) for e in eps])
        lens = np.array([len(e["traces"][self.names[0]]) for e in eps])
        fold = np.array([e.get("logpzo_fold", -1) for e in eps], dtype=np.int64)

        # -2 marks episodes LogpZO was built from under a rollout budget. Every method loses them,
        # so all the curves are still measured on exactly the same episodes.
        keep = fold != HELD_IN
        self.n_held_in = int((~keep).sum())
        T = int(lens[keep].max())
        if truncate != "none":
            T = {"min": int(lens[keep].min()),
                 "median_succ": int(np.median(lens[keep & succ])),
                 "mean_succ": int(round(lens[keep & succ].mean()))}[truncate]
            # Cut long episodes down to T, and throw away ones shorter than T. We don't pad them,
            # because padding adds a fake flat tail and the plateau trigger would read that as the
            # robot being stuck. Same rule as sweep_gate_configs.py.
            keep &= lens >= T

        self.truncate = truncate
        self.horizon = T
        self.succ = succ[keep]
        self.n_dropped = int((~keep).sum()) - self.n_held_in
        # Which group each episode was in when LogpZO was fit. Its band has to be built from the
        # other groups' episodes, the same ones its flow was not trained on.
        self.logpzo_fold = fold[keep]
        self.traces: Dict[str, List[np.ndarray]] = {
            n: [np.asarray(e["traces"][n], dtype=np.float64)[:T]
                for e, k in zip(eps, keep) if k]
            for n in self.names
        }
        self.n_ep = len(self.succ)
        # How long a successful episode usually runs. The figure draws this as a vertical line:
        # past it, an episode is already taking suspiciously long, so even a stopwatch starts to
        # look accurate. Once we trim all episodes to the same length this number just equals that
        # length, so the plot skips the line rather than drawing a useless one.
        self.raw_lens = lens[keep]
        self.l_succ = float(np.minimum(self.raw_lens, T)[self.succ].mean())

    @property
    def thrifty_variants(self) -> List[str]:
        return [n[len("novelty_"):] for n in self.names if n.startswith("novelty_")]

    def describe(self) -> str:
        held = (f", {self.n_held_in} held in for LogpZO" if self.n_held_in else "")
        return (f"[{self.tag}] task {self.meta.get('task_id')} | {self.n_ep} episodes "
                f"({int(self.succ.sum())} success / {int((~self.succ).sum())} failure) | "
                f"truncate={self.truncate} horizon={self.horizon} "
                f"(dropped {self.n_dropped} shorter than T{held})")


# ---------------------------------------------------------------------------------------------
# Quantile gates (Thrifty)
# ---------------------------------------------------------------------------------------------


def recalibrate_thresholds(pool: np.ndarray, alphas: np.ndarray, higher_fires: bool) -> np.ndarray:
    """Find the cutoff that only the top (or bottom) alpha fraction of `pool` passes.

    This is a copy of what ThriftyGate.recalibrate does, just computed for every alpha at once.
    Sort the scores; if we want the gate to fire on 1% of steps, take the value sitting 99% of the
    way up the sorted list. `higher_fires` picks which end: the novelty score fires when it goes
    ABOVE its cutoff, the risk score fires when it drops BELOW its cutoff.
    """
    n = len(pool)
    idx = np.minimum(((1.0 - alphas) * n).astype(np.int64), n - 1)
    order = np.sort(pool) if higher_fires else np.sort(pool)[::-1]
    return order[idx]


def first_fire_over_thresholds(trace: np.ndarray, thr: np.ndarray,
                               higher_fires: bool) -> np.ndarray:
    """The first step where this episode's score crosses each cutoff. INF means it never did.

    Trick for speed: instead of scanning once per cutoff, take the running maximum of the trace
    (the highest value seen so far at each step). That only ever goes up, so a binary search finds
    where it passes each cutoff, and we get all the cutoffs in one shot.
    """
    if higher_fires:
        run = np.maximum.accumulate(trace)
        pos = np.searchsorted(run, thr, side="right")          # first t with run[t] > thr
    else:
        run = -np.minimum.accumulate(trace)
        pos = np.searchsorted(run, -thr, side="right")         # first t with trace[t] < thr
    return np.where(pos < len(trace), pos, INF).astype(np.int64)


def thrifty_fires(corpus: SignalCorpus, variant: str, alphas: np.ndarray,
                  calib_idx: np.ndarray) -> Dict[str, np.ndarray]:
    """For each episode and each alpha, the step the gate would first fire on.

    Returned three ways: novelty alone, risk alone, and either one (the rule the robot actually
    runs). The cutoff is computed from every step of the calibration episodes, successes and
    failures mixed together, because that is what recalibrate_online does on the robot -- it never
    looks at whether an episode succeeded. Not needing labels is the whole appeal of the method.
    """
    nov = corpus.traces[f"novelty_{variant}"]
    saf = corpus.traces[f"safety_{variant}"]
    delta = recalibrate_thresholds(np.concatenate([nov[i] for i in calib_idx]), alphas, True)
    beta = recalibrate_thresholds(np.concatenate([saf[i] for i in calib_idx]), alphas, False)

    fn = np.stack([first_fire_over_thresholds(t, delta, True) for t in nov])
    fr = np.stack([first_fire_over_thresholds(t, beta, False) for t in saf])
    return dict(novelty=fn, risk=fr, union=np.minimum(fn, fr),
                delta_h=delta, beta_h=beta)


# ---------------------------------------------------------------------------------------------
# RewardGate grid on the same episodes
# ---------------------------------------------------------------------------------------------


def robometer_fire_table(corpus: SignalCorpus, signal: str = "robometer_progress"):
    """For each episode and each of our gate's settings, the step it would first fire on."""
    lws = [w for w in LW_GRID if w <= corpus.horizon]
    drop_keys = [None] + list(itertools.product(SWS, DTHRS, MAGS))
    plat_keys = [None] + list(itertools.product(lws, PTHRS))
    traces = corpus.traces[signal]

    cols, cfgs = [], []
    for s in SMOOTH:
        D = np.full((corpus.n_ep, len(drop_keys)), INF, dtype=np.int64)
        P = np.full((corpus.n_ep, len(plat_keys)), INF, dtype=np.int64)
        for ei, v in enumerate(traces):
            vs = ema(v, s)
            df = drop_first_fires(vs, SWS, DTHRS, MAGS)
            pf = plateau_first_fires(vs, lws, PTHRS)
            for ki, k in enumerate(drop_keys):
                if k is not None:
                    D[ei, ki] = df[k]
            for ki, k in enumerate(plat_keys):
                if k is not None:
                    P[ei, ki] = pf[k]
        F = np.minimum(D[:, :, None], P[:, None, :])           # (n_ep, n_drop, n_plat)
        cols.append(F.reshape(corpus.n_ep, -1))
        cfgs += [dict(smoothing=s,
                      sw=dk[0] if dk else None, dthr=dk[1] if dk else None,
                      mag=dk[2] if dk else None,
                      lw=pk[0] if pk else None, pthr=pk[1] if pk else None)
                 for dk in drop_keys for pk in plat_keys]

    F = np.concatenate(cols, axis=1)
    valid = np.array([not (c["sw"] is None and c["lw"] is None) for c in cfgs])
    return F[:, valid], [c for c, v in zip(cfgs, valid) if v]


def grid_metrics(F: np.ndarray, succ: np.ndarray, horizon: int):
    """Score every setting at once: how many failures it catches, how often it cries wolf, how
    early it fires. Same maths as metrics_from_fires, just done for all settings in one pass."""
    Fs, Ff = F[succ], F[~succ]
    caught = Ff < INF
    recall = caught.mean(axis=0)
    fpr = (Fs < INF).mean(axis=0)
    tdet = np.where(caught, Ff / horizon, 1.0).mean(axis=0)
    return recall, fpr, (recall + 1.0 - fpr) / 2.0, tdet


# ---------------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------------


def stratified_folds(succ: np.ndarray, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Split the episodes into k groups, dealing successes and failures out one at a time like
    cards, so no group ends up with only successes or only failures."""
    folds = [[] for _ in range(k)]
    for cls in (True, False):
        idx = np.flatnonzero(succ == cls)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def mean_std(rows: List[dict], keys) -> dict:
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v))
    return out


METRIC_KEYS = ["recall", "fpr", "balacc", "avg_tdet", "med_tdet", "fire_rate"]


def _metrics(fire: np.ndarray, succ: np.ndarray, horizon: int) -> dict:
    m = metrics_from_fires(fire[succ], fire[~succ], horizon)
    m["fire_rate"] = float((fire < INF).mean())      # episode-level intervention rate
    return m


def thrifty_rows(corpus: SignalCorpus, alphas: np.ndarray, folds: int, repeats: int,
                 seed: int) -> List[dict]:
    rows = []
    all_idx = np.arange(corpus.n_ep)
    for variant in corpus.thrifty_variants:
        tabs = thrifty_fires(corpus, variant, alphas, all_idx)
        for fam in ("novelty", "risk", "union"):
            for ai, a in enumerate(alphas):
                rows.append(dict(
                    method="thrifty", variant=variant, family=fam, alpha=float(a),
                    protocol="insample", delta_h=float(tabs["delta_h"][ai]),
                    beta_h=float(tabs["beta_h"][ai]),
                    **_metrics(tabs[fam][:, ai], corpus.succ, corpus.horizon)))

        # ---- Now the honest version: the cutoff only ever sees the other groups' episodes ----
        per_repeat = {fam: [] for fam in ("novelty", "risk", "union")}
        thr_acc = {"delta_h": [], "beta_h": []}
        for rep in range(repeats):
            rng = np.random.default_rng(seed + rep)
            fold_idx = stratified_folds(corpus.succ, folds, rng)
            held = {fam: np.full((corpus.n_ep, len(alphas)), INF, dtype=np.int64)
                    for fam in per_repeat}
            for f in fold_idx:
                calib = np.setdiff1d(all_idx, f)
                t = thrifty_fires(corpus, variant, alphas, calib)
                for fam in per_repeat:
                    held[fam][f] = t[fam][f]
                thr_acc["delta_h"].append(t["delta_h"])
                thr_acc["beta_h"].append(t["beta_h"])
            for fam in per_repeat:
                per_repeat[fam].append([_metrics(held[fam][:, ai], corpus.succ, corpus.horizon)
                                        for ai in range(len(alphas))])
        d_mean = np.mean(thr_acc["delta_h"], axis=0)
        b_mean = np.mean(thr_acc["beta_h"], axis=0)
        for fam, reps in per_repeat.items():
            for ai, a in enumerate(alphas):
                rows.append(dict(
                    method="thrifty", variant=variant, family=fam, alpha=float(a),
                    protocol="cv", delta_h=float(d_mean[ai]), beta_h=float(b_mean[ai]),
                    **mean_std([r[ai] for r in reps], METRIC_KEYS)))
    return rows


# ---------------------------------------------------------------------------------------------
# Single-score quantile gates (Diff-DAgger's diffusion loss, LogpZO's density score)
# ---------------------------------------------------------------------------------------------


def quantile_variants(corpus: SignalCorpus, prefix: str) -> List[str]:
    """`dd_loss_nb500` -> `nb500`. One variant per hyperparameter setting that was recorded."""
    return [n[len(prefix) + 1:] for n in corpus.names if n.startswith(prefix + "_")]


def quantile_fires(corpus: SignalCorpus, signal: str, alphas: np.ndarray,
                   calib_idx: np.ndarray, calib_on: str = "demos") -> Dict[str, np.ndarray]:
    """First firing step per episode, for a signal that fires when it rises above a cutoff.

    Where the cutoff comes from is the whole point.

    demos     what Diff-DAgger publishes: the cutoff is a quantile of the losses the student gets
              on its own training demos, and alpha is that quantile.
    rollouts  the cutoff is a quantile of the rollout scores instead, so alpha is "what fraction
              of rollout steps am I willing to fire on". This reaches cutoffs the demos cannot
              produce, which the method could not do at deploy time without collecting rollouts
              first, so it reads as an upper bound rather than as the method.
    """
    tr = corpus.traces[signal]
    demo = corpus.meta.get("calib_scores", {}).get(signal)
    if calib_on == "demos" and demo:
        # alpha is the fraction of demo states allowed to exceed the cutoff.
        thr = np.quantile(np.asarray(demo, dtype=np.float64), 1.0 - alphas)
    else:
        thr = recalibrate_thresholds(np.concatenate([tr[i] for i in calib_idx]), alphas, True)
    return dict(fire=np.stack([first_fire_over_thresholds(t, thr, True) for t in tr]), thr=thr)


def quantile_rows(corpus: SignalCorpus, method: str, prefix: str, alphas: np.ndarray,
                  folds: int, repeats: int, seed: int, calib_on: str = "demos") -> List[dict]:
    """Rows for one single-score method, in-sample and cross-validated, like `thrifty_rows`."""
    rows, all_idx = [], np.arange(corpus.n_ep)
    for variant in quantile_variants(corpus, prefix):
        signal = f"{prefix}_{variant}"
        tab = quantile_fires(corpus, signal, alphas, all_idx, calib_on)
        for ai, a in enumerate(alphas):
            rows.append(dict(
                method=method, variant=variant, family=method, alpha=float(a),
                protocol="insample", threshold=float(tab["thr"][ai]),
                **_metrics(tab["fire"][:, ai], corpus.succ, corpus.horizon)))

        # The cutoff only ever sees the other folds' episodes.
        per_repeat, thr_acc = [], []
        for rep in range(repeats):
            rng = np.random.default_rng(seed + rep)
            held = np.full((corpus.n_ep, len(alphas)), INF, dtype=np.int64)
            for f in stratified_folds(corpus.succ, folds, rng):
                t = quantile_fires(corpus, signal, alphas, np.setdiff1d(all_idx, f), calib_on)
                held[f] = t["fire"][f]
                thr_acc.append(t["thr"])
            per_repeat.append([_metrics(held[:, ai], corpus.succ, corpus.horizon)
                               for ai in range(len(alphas))])
        thr_mean = np.mean(thr_acc, axis=0)
        for ai, a in enumerate(alphas):
            rows.append(dict(
                method=method, variant=variant, family=method, alpha=float(a),
                protocol="cv", threshold=float(thr_mean[ai]),
                **mean_std([r[ai] for r in per_repeat], METRIC_KEYS)))
    return rows


# ---------------------------------------------------------------------------------------------
# LogpZO's band gate: one cutoff per timestep, not one cutoff for the whole episode
# ---------------------------------------------------------------------------------------------


def band_calib_traces(corpus: SignalCorpus, variant: str, fold: str) -> np.ndarray:
    """The held-out successful traces LogpZO's band is built from, for one fold.

    score_signals_offline.py scored these with the same flow that scored the fold's episodes, but
    never trained the flow on them.
    """
    stored = corpus.meta.get("logpzo", {}).get(variant, {}).get("calib_traces", {})
    T = corpus.horizon
    out = [pad_to(t, T) for t in stored.get(fold, [])]
    return np.asarray(out) if out else np.empty((0, T))


def build_band(traces: np.ndarray, alphas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Split the successful traces the way SAFE does -- 30% sets the average curve and the spread,
    70% sets how wide the band has to be -- and return one band per alpha."""
    idx = rng.permutation(len(traces))
    n_first = max(1, int(len(traces) * 0.3))
    return conformal_band(traces[idx[:n_first]], traces[idx[n_first:]], alphas)


def first_fire_over_band(trace: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """First step where the score rises above the band, one answer per band. INF means never.

    Only the episode's real steps are read. The band may be longer, because it was built from
    episodes stretched to a common length, but an episode cannot fire after it has ended.
    """
    over = np.asarray(trace, dtype=np.float64)[None, :] > bands[:, :len(trace)]
    return np.where(over.any(axis=1), over.argmax(axis=1), INF).astype(np.int64)


def band_rows(corpus: SignalCorpus, alphas: np.ndarray, repeats: int, seed: int) -> List[dict]:
    """Rows for LogpZO, scored two ways like every other method here.

    cv        the band comes from held-out successful episodes of the same fold, so neither the
              flow nor the band ever saw the episode being judged.
    insample  the band comes from every successful episode in the corpus, including the ones it is
              then read on. Too optimistic, and kept only so the figure can show both protocols.
    """
    rows = []
    for variant in quantile_variants(corpus, "logpzo"):
        traces = corpus.traces[f"logpzo_{variant}"]
        folds = sorted({int(f) for f in corpus.logpzo_fold})

        per_repeat = []
        for rep in range(repeats):
            rng = np.random.default_rng(seed + rep)
            held = np.full((corpus.n_ep, len(alphas)), INF, dtype=np.int64)
            for f in folds:
                cal = band_calib_traces(corpus, variant, "0" if f < 0 else str(f))
                if len(cal) < 2:
                    held = None
                    break
                bands = build_band(cal, alphas, rng)
                for i in np.flatnonzero(corpus.logpzo_fold == f):
                    held[i] = first_fire_over_band(traces[i], bands)
            if held is None:
                break
            per_repeat.append([_metrics(held[:, ai], corpus.succ, corpus.horizon)
                               for ai in range(len(alphas))])
        if not per_repeat:
            print(f"  logpzo {variant}: no held-out calibration traces stored, skipping cv rows")
        else:
            for ai, a in enumerate(alphas):
                rows.append(dict(method="logpzo", variant=variant, family="logpzo", alpha=float(a),
                                 protocol="cv",
                                 **mean_std([r[ai] for r in per_repeat], METRIC_KEYS)))

        succ_traces = np.asarray([pad_to(traces[i], corpus.horizon)
                                  for i in np.flatnonzero(corpus.succ)])
        if len(succ_traces) < 2:
            continue
        per_repeat = []
        for rep in range(repeats):
            bands = build_band(succ_traces, alphas, np.random.default_rng(seed + 100 + rep))
            fire = np.stack([first_fire_over_band(t, bands) for t in traces])
            per_repeat.append([_metrics(fire[:, ai], corpus.succ, corpus.horizon)
                               for ai in range(len(alphas))])
        for ai, a in enumerate(alphas):
            rows.append(dict(method="logpzo", variant=variant, family="logpzo", alpha=float(a),
                             protocol="insample",
                             **mean_std([r[ai] for r in per_repeat], METRIC_KEYS)))
    return rows


def calibration_curve(corpus: SignalCorpus, variant: str, alpha: float, sizes: List[int],
                      repeats: int, seed: int) -> List[dict]:
    """How many episodes do you need before the cutoff settles down? The practical question.

    Take N episodes at random, set the cutoff from just those, then see how often the gate fires
    on the episodes left over. Repeat with different draws. If the answers swing wildly, N is too
    small to trust for a real deployment.
    """
    alphas = np.array([alpha])
    out = []
    for n in sizes:
        if n >= corpus.n_ep:
            continue
        per = {fam: [] for fam in ("novelty", "risk", "union")}
        thr = []
        for rep in range(repeats):
            rng = np.random.default_rng(seed + 1000 * rep + n)
            calib = rng.choice(corpus.n_ep, size=n, replace=False)
            held = np.setdiff1d(np.arange(corpus.n_ep), calib)
            t = thrifty_fires(corpus, variant, alphas, calib)
            thr.append((t["delta_h"][0], t["beta_h"][0]))
            for fam in per:
                per[fam].append(_metrics(t[fam][held, 0], corpus.succ[held], corpus.horizon))
        for fam, reps in per.items():
            out.append(dict(variant=variant, family=fam, alpha=alpha, n_calib=n,
                            delta_h=float(np.mean([a for a, _ in thr])),
                            delta_h_std=float(np.std([a for a, _ in thr])),
                            beta_h=float(np.mean([b for _, b in thr])),
                            beta_h_std=float(np.std([b for _, b in thr])),
                            **mean_std(reps, METRIC_KEYS)))
    return out


def trivial_baseline_rows(corpus: SignalCorpus) -> List[dict]:
    """The two dumb baselines: fire at a fixed step, and fire when progress drops below a fixed
    value. Same definitions as sweep_gate_configs.py.

    Neither has anything to tune beyond one number, so there is nothing for cross-validation to
    correct. They are on the figure for context, not as claims.
    """
    rows = []
    traces = corpus.traces["robometer_progress"]
    lens = np.array([len(t) for t in traces])
    for T in range(25, corpus.horizon + 1, 25):
        fire = np.where(lens > T, T, INF)             # only fires if the episode reaches step T
        rows.append(dict(method="robometer", protocol="insample", family="timeout", T=T,
                         **_metrics(fire, corpus.succ, corpus.horizon)))
    for theta in np.round(np.arange(0.05, 1.0, 0.05), 2):
        for delay in [0, 25, 50, 100, 150, 200, 250, 300, 400]:
            if delay >= corpus.horizon:
                continue
            fire = np.array([_first_fire_mask((np.arange(len(t)) >= delay) & (t < theta))
                             for t in traces])
            rows.append(dict(method="robometer", protocol="insample", family="absolute",
                             theta=float(theta), delay=delay,
                             **_metrics(fire, corpus.succ, corpus.horizon)))
    return rows


def _first_fire_mask(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else INF


def robometer_rows(corpus: SignalCorpus, folds: int, repeats: int, seed: int) -> List[dict]:
    F, cfgs = robometer_fire_table(corpus)
    rows = []
    for i, c in enumerate(cfgs):
        fam = ("gate" if (c["sw"] and c["lw"]) else ("drop_only" if c["sw"] else "plateau_only"))
        rows.append(dict(method="robometer", protocol="insample", family=fam, cfg=c,
                         **_metrics(F[:, i], corpus.succ, corpus.horizon)))
    rows += trivial_baseline_rows(corpus)

    # Here we re-pick our gate's settings inside every split. The rule: among settings that stay
    # under the cap on firing during successful episodes, take the one that fires earliest. We need
    # a rule like this because the grid has thousands of settings, and picking the best one on the
    # same episodes we then report would flatter it.
    all_idx = np.arange(corpus.n_ep)
    per_budget = {b: [] for b in FPR_BUDGETS}
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        fold_idx = stratified_folds(corpus.succ, folds, rng)
        held = {b: np.full(corpus.n_ep, INF, dtype=np.int64) for b in FPR_BUDGETS}
        for f in fold_idx:
            calib = np.setdiff1d(all_idx, f)
            r_c, f_c, b_c, t_c = grid_metrics(F[calib], corpus.succ[calib], corpus.horizon)
            for b in FPR_BUDGETS:
                ok = np.flatnonzero(f_c <= b + 1e-9)
                if not ok.size:
                    ok = np.flatnonzero(f_c <= f_c.min() + 1e-9)
                pick = ok[np.lexsort((-b_c[ok], t_c[ok]))[0]]
                held[b][f] = F[f, pick]
        for b in FPR_BUDGETS:
            per_budget[b].append(_metrics(held[b], corpus.succ, corpus.horizon))
    for b, reps in per_budget.items():
        rows.append(dict(method="robometer", protocol="cv", family="gate", fpr_budget=b,
                         **mean_std(reps, METRIC_KEYS)))
    return rows


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------


def print_alpha_table(rows: List[dict], variant: str, protocol: str, alphas):
    sel = [r for r in rows if r["method"] == "thrifty" and r["variant"] == variant
           and r["protocol"] == protocol]
    by = {(r["family"], round(r["alpha"], 8)): r for r in sel}
    print(f"\n  ThriftyDAgger alpha sweep [{variant}, {protocol}]")
    print("    alpha    | " + " | ".join(f"{f:^33}" for f in ("novelty", "risk", "union")))
    print("             | " + " | ".join(" balacc  recall   fpr   t_det  fire" for _ in range(3)))
    for a in alphas:
        cells = []
        for fam in ("novelty", "risk", "union"):
            r = by.get((fam, round(float(a), 8)))
            cells.append(f" {r['balacc']:.3f}  {r['recall']:.3f}  {r['fpr']:.3f}  "
                         f"{r['avg_tdet']:.3f}  {r['fire_rate']:.2f}")
        print(f"    {a:<8.5f} | " + " | ".join(cells))


def print_front(rows: List[dict], label: str, n: int = 10):
    front = pareto(rows, ykey="balacc", xkey="avg_tdet")
    print(f"\n  Pareto front -- {label}  ({len(front)} points)")
    print("    t_det  balacc  recall   fpr   fire | config")
    step = max(1, len(front) // n)
    for r in front[::step]:
        if r["method"] == "thrifty":
            cfg = f"thrifty {r['variant']} {r['family']} alpha={r['alpha']:.5f}"
        elif "fpr_budget" in r:
            cfg = f"robometer budget={r['fpr_budget']}"
        else:
            c = r["cfg"]
            cfg = (f"robometer s={c['smoothing']} sw={c['sw']} d={c['dthr']} m={c['mag']} "
                   f"lw={c['lw']} p={c['pthr']}")
        print(f"    {r['avg_tdet']:.3f}  {r['balacc']:.3f}  {r['recall']:.3f}  {r['fpr']:.3f}  "
              f"{r['fire_rate']:.2f} | {cfg}")


def hypervolume(rows: List[dict]) -> float:
    """One number for a whole curve: the area underneath it. Higher is better. Accuracy is
    measured above 0.5, since 0.5 is what coin-flipping gets you."""
    front = sorted(pareto(rows, ykey="balacc", xkey="avg_tdet"), key=lambda r: r["avg_tdet"])
    if not front:
        return float("nan")
    hv, prev = 0.0, 0.0
    best = 0.5
    for r in front:
        hv += (r["avg_tdet"] - prev) * max(0.0, best - 0.5)
        prev, best = r["avg_tdet"], max(best, r["balacc"])
    hv += (1.0 - prev) * max(0.0, best - 0.5)
    return hv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="signal_traces.json from collect_signal_traces.py")
    ap.add_argument("--truncate", choices=["none", "min", "median_succ", "mean_succ"],
                    default="min",
                    help="SAFE length-confound control. 'min' (default) is the honest setting: "
                         "LIBERO failures otherwise run to the 800-step cap while successes stop "
                         "early, so a stopwatch scores near-perfect balanced accuracy.")
    ap.add_argument("--dd-calib", choices=["demos", "rollouts"], default="demos",
                    help="where Diff-DAgger's threshold comes from. 'demos' is the published "
                         "method and collapses here, because the student's loss on its own "
                         "training demos is far below its loss on any rollout. 'rollouts' lets "
                         "the threshold be any value the rollout scores span, which shows what "
                         "the signal could do with a threshold it cannot actually obtain.")
    ap.add_argument("--ucf-calib", choices=["demos", "rollouts"], default="demos",
                    help="where UCF's threshold comes from. Same question as --dd-calib, asked of "
                         "the vector-field uncertainty: 'demos' is the published method, "
                         "'rollouts' is the upper bound it could reach with a threshold it cannot "
                         "actually obtain at deploy time.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5, help="repeated stratified K-fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-failures", type=int, default=10,
                    help="skip corpora with fewer failures than this -- the frontier is noise")
    ap.add_argument("--calib-alpha", type=float, default=0.003,
                    help="alpha at which to report the calibration-set-size curve; 0 disables")
    ap.add_argument("--no-robometer", action="store_true",
                    help="skip the RewardGate overlay (the slow part)")
    ap.add_argument("--out-dir", default="outputs/thrifty_frontier")
    args = ap.parse_args()

    alphas = alpha_grid()
    # Only print a handful of alphas so the table fits on screen. The JSON keeps all of them.
    show = [alphas[np.argmin(np.abs(alphas - a))] for a in
            (1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)]
    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.traces:
        corpus = SignalCorpus(path, truncate=args.truncate)
        print("\n" + "=" * 96)
        print(corpus.describe())
        n_fail = int((~corpus.succ).sum())
        if n_fail < args.min_failures:
            print(f"  SKIP: only {n_fail} failures (< --min-failures {args.min_failures})")
            continue

        rows = thrifty_rows(corpus, alphas, args.folds, args.repeats, args.seed)
        # Diff-DAgger fires when its loss passes one number, so alpha is a quantile of the demo
        # losses. LogpZO fires when its score passes a curve, so alpha is the width of that curve.
        if quantile_variants(corpus, "dd_loss"):
            rows += quantile_rows(corpus, "diffdagger", "dd_loss", alphas, args.folds,
                                  args.repeats, args.seed, args.dd_calib)
        # UCF fires on the same rule as Diff-DAgger -- one score over one cutoff, the cutoff a
        # quantile of the demo scores -- so it goes through the same code.
        if quantile_variants(corpus, "ucf"):
            rows += quantile_rows(corpus, "ucf", "ucf", alphas, args.folds,
                                  args.repeats, args.seed, args.ucf_calib)
        rows += band_rows(corpus, alphas, args.repeats, args.seed)
        for variant in corpus.thrifty_variants:
            print_alpha_table(rows, variant, "cv", show)

        calib = []
        if args.calib_alpha > 0:
            sizes = [n for n in (5, 10, 15, 20, 30, 40) if n < corpus.n_ep]
            print(f"\n  Calibration-set size, alpha={args.calib_alpha} (threshold fit on N "
                  f"episodes, read on the rest; +- is spread over {4 * args.repeats} draws)")
            print("    variant  family  |  N | delta_h            fpr          fire_rate")
            for variant in corpus.thrifty_variants:
                calib += calibration_curve(corpus, variant, args.calib_alpha, sizes,
                                           4 * args.repeats, args.seed)
            for r in calib:
                if r["family"] != "risk":
                    print(f"    {r['variant']:<8} {r['family']:<7} | {r['n_calib']:>2} | "
                          f"{r['delta_h']:.2e}+-{r['delta_h_std']:.0e}  "
                          f"{r['fpr']:.3f}+-{r['fpr_std']:.3f}  {r['fire_rate']:.3f}+-"
                          f"{r['fire_rate_std']:.3f}")

        if not args.no_robometer:
            rows += robometer_rows(corpus, args.folds, args.repeats, args.seed)

        thr_cv = [r for r in rows if r["method"] == "thrifty" and r["protocol"] == "cv"]
        thr_is = [r for r in rows if r["method"] == "thrifty" and r["protocol"] == "insample"]

        # Break it down by ensemble size and by which of the two scores is used. The last row
        # pools all nine of those, which quietly picks the best one using the very episodes we are
        # reporting on, so it reads better than it should. `union` is the rule the robot actually
        # runs, so that is the row to put in a baseline table.
        print("\n  Frontier hypervolume (area under balacc-vs-t_det front, ref balacc=0.5)")
        print("    variant  family  |    cv   insample")
        for variant in corpus.thrifty_variants:
            for fam in ("novelty", "risk", "union"):
                sub_cv = [r for r in thr_cv if r["variant"] == variant and r["family"] == fam]
                sub_is = [r for r in thr_is if r["variant"] == variant and r["family"] == fam]
                mark = "  <- deployed rule" if fam == "union" else ""
                print(f"    {variant:<8} {fam:<7} | {hypervolume(sub_cv):.4f}  "
                      f"{hypervolume(sub_is):.4f}{mark}")
        print(f"    {'POOLED':<8} {'any':<7} | {hypervolume(thr_cv):.4f}  "
              f"{hypervolume(thr_is):.4f}   (pooling picks variant+family on the eval data)")
        print_front(thr_cv, "ThriftyDAgger, all variants, cross-validated")

        if not args.no_robometer:
            rm_cv = [r for r in rows if r["method"] == "robometer" and r["protocol"] == "cv"]
            rm_is = [r for r in rows if r["method"] == "robometer"
                     and r["protocol"] == "insample"
                     and r["family"] in ("gate", "drop_only", "plateau_only")]
            print_front(rm_is, "Robometer RewardGate, in-sample grid")
            print_front(rm_cv, "Robometer RewardGate, per-fold reselected")
            print(f"\n  HEADLINE  hypervolume, cross-validated:"
                  f"  robometer={hypervolume(rm_cv):.4f}   "
                  f"thrifty[union]={hypervolume([r for r in thr_cv if r['family'] == 'union']):.4f}"
                  f"   thrifty[pooled]={hypervolume(thr_cv):.4f}")
            print(f"            in-sample:  robometer={hypervolume(rm_is):.4f}   "
                  f"thrifty[union]={hypervolume([r for r in thr_is if r['family'] == 'union']):.4f}"
                  f"   thrifty[pooled]={hypervolume(thr_is):.4f}")

        out = dict(tag=corpus.tag, source=os.path.abspath(path), truncate=args.truncate,
                   horizon=corpus.horizon, l_succ=corpus.l_succ, n_ep=corpus.n_ep,
                   n_success=int(corpus.succ.sum()), n_failure=n_fail,
                   folds=args.folds, repeats=args.repeats, seed=args.seed,
                   meta=corpus.meta, rows=rows, calibration_curve=calib)
        dest = os.path.join(args.out_dir, f"{corpus.tag}.json")
        json.dump(out, open(dest, "w"))
        print(f"\n  -> {len(rows)} rows written to {dest}")


if __name__ == "__main__":
    main()
