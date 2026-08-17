#!/usr/bin/env python3
"""§3 experiment 2: Return On Human Effort (ROHE) over the deploy rounds.

ROHE = CPP / (1 + H/T), where T is every episode the arm collected in a round, H is how many of
them the gate handed to the expert, and CPP is the success rate of the human-plus-policy team.

The expert is a real person, so an episode the gate hands over is counted as a success even when
the pi0 stand-in we run in simulation failed it. CPP therefore only drops when the gate stays quiet
through an episode that then fails, and it sits near 1 for every arm; the metric is mostly "how
rarely does the gate ask for help". `--cpp observed` uses the raw simulated outcome instead, which
is the pessimistic reading and is only there as a check.

A round is one DAgger iteration: the policy is frozen while it collects, so the episodes inside a
round are repeated draws from one policy and can be treated as a sample.

Everything is read from the training log. Each collection line reports the outcome and whether the
gate fired:

    123/1500 expert transitions (attempt 7, success=True, interventions=1, stored=88)

The band around each point is a percentile bootstrap over the episodes of that round: resample the
round's episodes with replacement, recompute ROHE, and take the 2.5th and 97.5th percentiles. It
shows how much of the gap between two arms is just the small number of episodes per round.

Usage:
    uv run python scripts/plot_rohe_curve.py \
        --run "Robometer (ours)=outputs/2026-08-13/t1_sys31_robometer_57478" \
        --run "ThriftyDAgger=outputs/2026-08-13/t1_sys31_thrifty_57479" \
        --run "Diff-DAgger=outputs/2026-08-13/t1_sys31_diffdagger_57480" \
        --title "Task 1, 10 rounds x 1500 expert transitions" --out rohe_t1_sys31.png
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ITER_RE = re.compile(r"DAgger iteration (\d+)/(\d+)")
ATTEMPT_RE = re.compile(r"success=(True|False), interventions=(\d+), stored=(\d+)")
EVAL_RE = re.compile(r"Success Rate:\s*([0-9.]+)%")
EVAL_N_RE = re.compile(r"Evaluation over (\d+) episodes")

# One colour per arm, in the order the arms are usually named.
PALETTE = ["#2a78d6", "#9b4dd6", "#d64550", "#0f8f8f", "#eb6834", "#1baf7a"]
MARKERS = ["o", "P", "v", "D", "s", "^"]
INK = "#1a1a1a"
GRID = "#d9d7d2"


def find_log(path):
    """Accept a run directory, a directory plus a log name, or the log file itself."""
    if os.path.isfile(path):
        return path
    for name in ("training.log", "train_dagger.log"):
        cand = os.path.join(path, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"no training log under {path}")


def parse_log(path):
    """Return one dict per round, plus the eval success rates in the order they were printed.

    Lines before the first "DAgger iteration" marker belong to no round; they are the eval of the
    untouched student, so their episodes are dropped but the eval number is kept.
    """
    rounds, evals = [], []
    cur = None
    eval_n = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if "expert transitions (attempt" not in line and "Success Rate" not in line \
                    and "DAgger iteration" not in line and "Evaluation over" not in line:
                continue
            m = ITER_RE.search(line)
            if m:
                cur = dict(round=int(m.group(1)), success=[], fired=[])
                rounds.append(cur)
                continue
            m = EVAL_N_RE.search(line)
            if m:
                eval_n = int(m.group(1))
                continue
            m = EVAL_RE.search(line)
            if m:
                # The round this eval measures: the eval printed before iteration 1 is round 0,
                # the untouched student.
                evals.append(dict(round=len(rounds), rate=float(m.group(1)) / 100.0, n=eval_n))
                continue
            m = ATTEMPT_RE.search(line)
            if m and cur is not None:
                cur["success"].append(m.group(1) == "True")
                cur["fired"].append(int(m.group(2)) > 0)
    return rounds, evals


def rohe(success, fired):
    """ROHE for one set of episodes. Returns nan if the set is empty."""
    t = len(success)
    if t == 0:
        return float("nan")
    cpp = float(np.mean(success))
    h = float(np.sum(fired))
    return cpp / (1.0 + h / t)


def bootstrap_ci(success, fired, n_boot=4000, level=0.95, seed=0):
    """Percentile bootstrap over the episodes, resampling (success, fired) pairs together."""
    t = len(success)
    if t < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    s = np.asarray(success, dtype=float)
    f = np.asarray(fired, dtype=float)
    idx = rng.integers(0, t, size=(n_boot, t))
    cpp = s[idx].mean(axis=1)
    hrate = f[idx].mean(axis=1)
    vals = cpp / (1.0 + hrate)
    lo = (1.0 - level) / 2 * 100
    return float(np.percentile(vals, lo)), float(np.percentile(vals, 100 - lo))


def wilson(k, n, z=1.96):
    """Wilson interval for a success rate out of n episodes.

    The eval is n independent episodes, so the honest error bar is binomial. At n=50 it is roughly
    +/- 13 points near 50% success, which is worth showing before reading anything into a two-point
    difference between arms.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def eval_rows(evals):
    """Turn the parsed evals into per-round rows with a binomial band."""
    out = []
    for e in evals:
        n = e["n"]
        lo, hi = wilson(round(e["rate"] * n), n)
        out.append(dict(round=e["round"], rate=e["rate"], n=n, lo=lo, hi=hi))
    return out


def team_success(success, fired, cpp_mode):
    """Per-episode outcome that goes into CPP.

    Under the perfect-human reading, any episode the gate handed over is a success, because a real
    person would have finished it. Only a silent failure counts against the arm.
    """
    if cpp_mode == "observed":
        return list(success)
    return [bool(s) or bool(f) for s, f in zip(success, fired)]


def summarize(path, n_boot, seed, cpp_mode="perfect"):
    rounds, evals = parse_log(find_log(path))
    out = []
    all_s, all_f = [], []
    for r in rounds:
        s = team_success(r["success"], r["fired"], cpp_mode)
        f = r["fired"]
        if not s:
            continue
        all_s += s
        all_f += f
        lo, hi = bootstrap_ci(s, f, n_boot, seed=seed + r["round"])
        # The cumulative value pools every episode up to and including this round, which is what a
        # deployment would actually have paid so far.
        clo, chi = bootstrap_ci(all_s, all_f, n_boot, seed=seed + 1000 + r["round"])
        out.append(dict(
            round=r["round"], T=len(s), H=int(np.sum(f)), CPP=float(np.mean(s)),
            rohe=rohe(s, f), lo=lo, hi=hi,
            cum_T=len(all_s), cum_H=int(np.sum(all_f)), cum_rohe=rohe(all_s, all_f),
            cum_lo=clo, cum_hi=chi,
        ))
    overall = dict(T=len(all_s), H=int(np.sum(all_f)),
                   CPP=float(np.mean(all_s)) if all_s else float("nan"),
                   rohe=rohe(all_s, all_f))
    return out, overall, eval_rows(evals)


def print_table(label, rows, overall, evals):
    by_round = {e["round"]: e for e in evals}
    print(f"\n=== {label} ===")
    print("  round     T     H    H/T     CPP    ROHE      95% CI       cumROHE    eval%  "
          "     95% CI")
    for r in rows:
        e = by_round.get(r["round"])
        ev = (f"{100 * e['rate']:6.1f}  [{100 * e['lo']:4.1f}, {100 * e['hi']:4.1f}]"
              if e else "     -")
        print(f"  {r['round']:5d} {r['T']:5d} {r['H']:5d}  {r['H'] / r['T']:5.3f}  "
              f"{r['CPP']:6.3f}  {r['rohe']:6.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]   "
              f"{r['cum_rohe']:7.3f}  {ev}")
    print(f"  TOTAL {overall['T']:5d} {overall['H']:5d}  {overall['H'] / overall['T']:5.3f}  "
          f"{overall['CPP']:6.3f}  {overall['rohe']:6.3f}")
    if evals:
        seq = " -> ".join("%.0f" % (100 * e["rate"]) for e in evals)
        note = "   [round 0 = untouched student]" if 0 in by_round else ""
        print(f"  evals (n={evals[0]['n']}): {seq}{note}")


def draw(ax, panels, key, lo_key, hi_key, ylabel, title):
    """panels is a list of (label, rows); every row needs `round`, `key`, `lo_key`, `hi_key`."""
    ticks = set()
    for i, (label, rows) in enumerate(panels):
        x = [r["round"] for r in rows]
        ticks.update(x)
        c = PALETTE[i % len(PALETTE)]
        ax.fill_between(x, [r[lo_key] for r in rows], [r[hi_key] for r in rows],
                        color=c, alpha=0.15, lw=0, zorder=2 + i)
        ax.plot(x, [r[key] for r in rows], color=c, lw=2.4,
                marker=MARKERS[i % len(MARKERS)], ms=6.0, label=label, zorder=5 + i)
    ax.set_xlabel("Deploy round", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, color=INK, pad=8)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(sorted(ticks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help='"Label=path/to/run_dir" (or a log file). Repeat once per arm.')
    ap.add_argument("--out", default="rohe_curve.png")
    ap.add_argument("--title", default="Return on human effort over deploy rounds")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-round-only", action="store_true",
                    help="Draw only the per-round panel, without the cumulative one.")
    ap.add_argument("--cpp", choices=["perfect", "observed"], default="perfect",
                    help="perfect: a handed-over episode counts as a success (the paper's "
                         "assumption). observed: use the simulated expert's raw outcome.")
    ap.add_argument("--metric", choices=["rohe", "success", "both"], default="rohe",
                    help="rohe: per-round and cumulative ROHE. success: the eval success rate "
                         "after each round, with a binomial band. both: all three panels.")
    args = ap.parse_args()

    series = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        rows, overall, evals = summarize(path, args.n_boot, args.seed, args.cpp)
        if not rows:
            raise SystemExit(f"no collection rounds parsed from {path}")
        print_table(label, rows, overall, evals)
        series.append((label, rows, overall, evals))

    want_rohe = args.metric in ("rohe", "both")
    want_success = args.metric in ("success", "both")
    panels = []
    if want_rohe:
        panels.append(("rohe", "lo", "hi", "ROHE", "ROHE per round",
                       [(lab, r) for lab, r, _o, _e in series]))
        if not args.per_round_only:
            panels.append(("cum_rohe", "cum_lo", "cum_hi", "ROHE",
                           "ROHE, cumulative (all rounds so far)",
                           [(lab, r) for lab, r, _o, _e in series]))
    if want_success:
        panels.append(("rate", "lo", "hi", "Eval success rate",
                       "Autonomous success after each round",
                       [(lab, e) for lab, _r, _o, e in series]))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.6), squeeze=False)
    for ax, (key, lo, hi, ylab, title, data) in zip(axes[0], panels):
        draw(ax, data, key, lo, hi, ylab, title)

    handles, _ = axes[0][0].get_legend_handles_labels()
    if args.metric == "success":
        names = [f"{lab}  (final: {100 * e[-1]['rate']:.0f}%)" for lab, _r, _o, e in series]
    else:
        names = [f"{lab}  (all rounds: {ov['rohe']:.3f})" for lab, _r, ov, _e in series]
    fig.legend(handles, names, loc="lower center", ncol=len(series), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(args.title, fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(args.out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
