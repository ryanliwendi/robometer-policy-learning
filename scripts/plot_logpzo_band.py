#!/usr/bin/env python3
"""Show what LogpZO's conformal band actually looks like, next to real episode scores.

The frontier figure reduces LogpZO to a handful of dots. This one opens it up, so you can see
where the threshold comes from and why the curve is short.

Left panel: how the band is built. The calibration episodes -- successful rollouts the flow was
never trained on -- are drawn faintly, their per-step mean is the dark line, and the bands at a few
alphas sit above it. Right panel: how the band fires. A few held-out episodes are drawn against one
band, with a marker at the first step each one crosses.

The band itself comes from `conformal_band` in the gate module, the same function the frontier and
the live gate call, so nothing here is a re-implementation.

Usage:
    uv run python scripts/plot_logpzo_band.py gated_videos/sigq_t0_b50/signal_traces.json \
        --out outputs/logpzo_band_t0_b50.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robometer_policy_learning.utils.logpzo_gate import conformal_band  # noqa: E402

# Categorical slots 1 and 8 of the house palette, validated for colour-blind separation as a pair.
SUCCESS = "#2a78d6"
FAILURE = "#e34948"
BAND = "#4a3aa7"
INK = "#1a1a1a"
INK2 = "#52514e"
GRID = "#d9d7d2"
FAINT = "#b9b7b2"


def pad_to(trace, T):
    """Stretch a trace to length T by repeating its last value, the way the frontier does."""
    a = np.asarray(trace, dtype=np.float64)
    if len(a) >= T:
        return a[:T]
    return np.concatenate([a, np.full(T - len(a), a[-1] if len(a) else 0.0)])


def first_cross(trace, band):
    """First step where the score rises above the band, or None if it never does."""
    a = np.asarray(trace, dtype=np.float64)
    over = a > band[:len(a)]
    return int(np.argmax(over)) if over.any() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", help="signal_traces.json from score_signals_offline.py")
    ap.add_argument("--variant", default=None, help="which logpzo variant; default the only one")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.03, 0.003],
                    help="band widths to draw, loosest first")
    ap.add_argument("--fire-alpha", type=float, default=0.03,
                    help="which of those bands the right panel fires on")
    ap.add_argument("--n-show", type=int, default=3,
                    help="how many successes and how many failures to draw on the right")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logpzo_band.png")
    args = ap.parse_args()

    blob = json.load(open(args.traces))
    eps = blob["episodes"]
    meta = blob["meta"]["logpzo"]
    variant = args.variant or sorted(meta)[0]
    key = f"logpzo_{variant}"

    # The calibration episodes, stored by score_signals_offline.py at fit time.
    calib = meta[variant]["calib_traces"]
    calib_traces = [t for group in sorted(calib) for t in calib[group]]
    if len(calib_traces) < 2:
        raise SystemExit(f"variant {variant} stored only {len(calib_traces)} calibration traces")

    # Everything is cut to the shortest calibration episode, so the band and the traces line up.
    T = min(len(t) for t in calib_traces)
    cal = np.asarray([pad_to(t, T) for t in calib_traces])

    # Same 30/70 split the frontier uses: a first slice sets the shape, the rest sets the width.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(cal))
    n_first = max(1, int(len(cal) * 0.3))
    shape, width = cal[idx[:n_first]], cal[idx[n_first:]]
    alphas = np.array(sorted(args.alphas, reverse=True))
    bands = conformal_band(shape, width, alphas)

    held = [e for e in eps if e.get("logpzo_fold") != -2]
    succ = [e for e in held if e["success"]][: args.n_show]
    fail = [e for e in held if not e["success"]][: args.n_show]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    # ---- left: where the band comes from ----
    for i, t in enumerate(cal):
        axl.plot(t, color=FAINT, lw=0.9, alpha=0.8, zorder=2,
                 label="Calibration episodes (successes)" if i == 0 else None)
    axl.plot(cal.mean(axis=0), color=INK, lw=2.0, zorder=5, label="Per-step mean")
    # The bands land almost on top of each other, so they are separated by dash pattern and named
    # in the legend rather than labelled on the curve, where the text would overlap.
    dashes = [(0, (1, 2)), (0, (6, 3)), (0, ())]
    for a, b, d in zip(alphas, bands, dashes[-len(alphas):]):
        axl.plot(b, color=BAND, lw=2.0, ls=d, alpha=0.9, zorder=6, label=f"Band, α={a:g}")
    axl.set_title(f"Band from {len(cal)} calibration episodes "
                  f"({len(shape)} shape / {len(width)} width)",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    # The headline of this panel is how little the band moves across two orders of magnitude of
    # alpha, so state it as a number instead of asking the reader to measure the gap.
    spread = float(np.max(bands[-1] / np.maximum(bands[0], 1e-9)))
    axl.text(0.98, 0.03, f"widest α ({alphas[0]:g}) to narrowest ({alphas[-1]:g}):\n"
                         f"at most {100 * (spread - 1):.0f}% apart",
             transform=axl.transAxes, ha="right", va="bottom", fontsize=9, color=BAND)

    # ---- right: what it does to real episodes ----
    fire_band = bands[int(np.argmin(np.abs(alphas - args.fire_alpha)))]
    axr.plot(fire_band, color=BAND, lw=2.2, ls="--", zorder=6,
             label=f"Band, α={args.fire_alpha:g}")
    for group, color, name in ((succ, SUCCESS, "success"), (fail, FAILURE, "failure")):
        for i, e in enumerate(group):
            t = np.asarray(e["traces"][key], dtype=np.float64)[:T]
            axr.plot(t, color=color, lw=1.8, alpha=0.9, zorder=5,
                     label=f"Held-out {name}" if i == 0 else None)
            step = first_cross(t, fire_band)
            if step is not None:
                axr.plot([step], [t[step]], marker="o", ms=9, mfc=color, mec="white", mew=1.6,
                         zorder=8, ls="none")
    axr.set_title("Firing on held-out episodes (dot = first crossing)",
                  fontsize=10.5, color=INK, loc="left", pad=8)

    for ax in (axl, axr):
        ax.set_xlabel("Step", fontsize=10, color=INK2)
        ax.set_ylabel("LogpZO score", fontsize=10, color=INK2)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

    task = blob.get("meta", {}).get("task_id", "?")
    fig.suptitle(f"LogpZO conformal band, task {task}, variant {variant}",
                 fontsize=13, color=INK, x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=190, facecolor="white", bbox_inches="tight")
    print(f"-> {args.out}")

    # The numbers behind the picture, so the figure can be checked rather than trusted.
    print(f"\ncalibration episodes: {len(cal)}  (shape {len(shape)} / width {len(width)}), T={T}")
    print("  alpha   band@t=0   band@mid   band@end   fires on held-out succ / fail")
    for a, b in zip(alphas, bands):
        ns = sum(first_cross(np.asarray(e["traces"][key])[:T], b) is not None
                 for e in held if e["success"])
        nf = sum(first_cross(np.asarray(e["traces"][key])[:T], b) is not None
                 for e in held if not e["success"])
        n_s = sum(1 for e in held if e["success"])
        n_f = len(held) - n_s
        print(f"  {a:<7g} {b[0]:>9.1f}  {b[T // 2]:>9.1f}  {b[-1]:>9.1f}   "
              f"{ns:>3}/{n_s}  {nf:>3}/{n_f}")


if __name__ == "__main__":
    main()
