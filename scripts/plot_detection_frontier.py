#!/usr/bin/env python3
"""§2.2 figure: balanced accuracy vs average detection time, per task, per detector family.

Same axes as the SAFE failure-detection plots. Each family's threshold grid is reduced to its
Pareto front and subsampled to ~12 points, so a curve is "the best this family can do", not a
tangle of every config. The dashed vertical rule marks the mean successful-episode length: to its
right, an episode running that long is already anomalous and a stopwatch starts to work, so only
the part of a curve LEFT of the rule is detection rather than waiting. Under `--truncate min` the
rule collapses onto the horizon by construction and is omitted.

`--data-dir` picks which recorded results to draw:

  outputs/gate_frontier_n200   what sweep_gate_configs.py produced from the 200-episode runs. Our gate
                               only -- those runs never recorded the baselines' scores. The default.
  outputs/thrifty_frontier     what score_detection_frontier.py produced from the sigq_* runs, which
                               recorded our progress score AND ThriftyDAgger's two scores on the
                               same episodes. That is the only place the Thrifty curve can be drawn
                               next to ours without comparing different episodes.

Usage:
    uv run python scripts/plot_detection_frontier.py
    uv run python scripts/plot_detection_frontier.py --data-dir outputs/thrifty_frontier \
        --panels sigq_t0:"Task 0" sigq_t1:"Task 1" --out detection_frontier_thrifty.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_PANELS = ["t1_dp:Task 1", "t0_dp:Task 0", "t5_dp:Task 5", "t8_dp:Task 8"]
NPTS = 12

# Colours: three distinct ones for our gate and its two ablations, purple for ThriftyDAgger, and
# greys for the dumb baselines so they sit in the background instead of competing for attention.
STYLE = {
    "gate":         dict(label="Ours: drop + plateau", color="#2a78d6", lw=2.6, ls="-",
                         marker="o", ms=6.5, z=6, alpha=1.0),
    "drop_only":    dict(label="Drop only (ablation)", color="#eb6834", lw=1.9, ls="--",
                         marker="s", ms=5.0, z=5, alpha=0.95),
    "plateau_only": dict(label="Plateau only (ablation)", color="#1baf7a", lw=1.9, ls="--",
                         marker="^", ms=5.5, z=5, alpha=0.95),
    "thrifty":      dict(label="ThriftyDAgger (novelty + Q-risk)", color="#9b4dd6", lw=2.2,
                         ls="-", marker="P", ms=6.0, z=6, alpha=1.0),
    "thrifty_nov":  dict(label="Thrifty novelty only (ablation)", color="#c48ae8", lw=1.7,
                         ls="--", marker="X", ms=5.5, z=5, alpha=0.95),
    "thrifty_risk": dict(label="Thrifty Q-risk only (ablation)", color="#d6b3ee", lw=1.5,
                         ls=":", marker="*", ms=6.5, z=4, alpha=0.95),
    "diffdagger":   dict(label="Diff-DAgger (diffusion loss)", color="#d64550", lw=2.2,
                         ls="-", marker="v", ms=6.0, z=6, alpha=1.0),
    "logpzo":       dict(label="LogpZO (density)", color="#0f8f8f", lw=2.2,
                         ls="-", marker="D", ms=5.5, z=6, alpha=1.0),
    "absolute":     dict(label="Absolute threshold", color="#8a8985", lw=1.5, ls="-",
                         marker="D", ms=4.0, z=3, alpha=0.9),
    "timeout":      dict(label="Timeout / periodic", color="#52514e", lw=1.5, ls=":",
                         marker="v", ms=4.5, z=3, alpha=0.9),
}
# ThriftyDAgger's novelty-only and risk-only ablations are left off: with two more baselines on
# the panel it was too crowded to read, and the union curve is the rule the robot actually runs.
# Put "thrifty_nov" / "thrifty_risk" back in this list to draw them again.
ORDER = ["gate", "drop_only", "plateau_only", "thrifty",
         "diffdagger", "logpzo", "absolute", "timeout"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2df"

# The Thrifty results are stored per ensemble size and per rule, so we have to say which ones to
# draw. We draw the ensemble size the training runs actually use (thrifty_train_steps: 200), not
# whichever of the three happened to score best -- picking that per task would flatter the baseline.
THRIFTY_FAMILY = {"union": "thrifty", "novelty": "thrifty_nov", "risk": "thrifty_risk"}


def select(rows, fam, protocol, variant, quant_variants=None):
    """Pull out just the rows for one curve on the plot."""
    quant_variants = quant_variants or {}
    out = []
    for r in rows:
        if r.get("protocol", "insample") != protocol:
            continue
        if r.get("method") == "thrifty":
            if THRIFTY_FAMILY.get(r["family"]) == fam and r.get("variant") == variant:
                out.append(r)
        elif r.get("method") in quant_variants:
            # One curve per method, at the hyperparameter setting the caller asked for.
            if r["family"] == fam and r.get("variant") == quant_variants[r["method"]]:
                out.append(r)
        elif r["family"] == fam:
            out.append(r)
    return out


def front(rows):
    pts = [(r["balacc"], r["avg_tdet"]) for r in rows
           if np.isfinite(r.get("balacc", np.nan))]
    pts.sort(key=lambda p: (p[1], -p[0]))
    out, best = [], -np.inf
    for b, t in pts:
        if b > best + 1e-12:
            best = b
            out.append((b, t))
    return out


def thin(pts, n=NPTS):
    """Keep the endpoints, evenly sample the middle."""
    if len(pts) <= n:
        return pts
    idx = sorted(set(np.linspace(0, len(pts) - 1, n).round().astype(int)))
    return [pts[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("GATE_FRONT_DIR",
                                                         "outputs/gate_frontier_n200"))
    ap.add_argument("--panels", nargs="+", default=DEFAULT_PANELS,
                    help='"tag:Title" per panel, in reading order')
    ap.add_argument("--protocol", choices=["insample", "cv"], default="insample",
                    help="insample matches the RewardGate-only figure; cv reads the "
                         "cross-validated rows `score_detection_frontier.py` also writes")
    ap.add_argument("--dd-variant", default="nb500",
                    help="which Diff-DAgger N_b setting to draw")
    ap.add_argument("--logpzo-variant", default="s2000",
                    help="which LogpZO training-length setting to draw")
    ap.add_argument("--thrifty-variant", default="s200",
                    help="ensemble training budget to plot; s200 is what the arm configs deploy")
    ap.add_argument("--xmax", type=float, default=1.0,
                    help="right edge of the x axis. Without truncation the curves all finish well "
                         "before 1.0, so cutting the axis at about 0.6 fills the panel instead of "
                         "leaving half of it empty. Points past the cut are clipped, not dropped.")
    ap.add_argument("--out", default="detection_frontier.png")
    args = ap.parse_args()
    QV = {"diffdagger": args.dd_variant, "logpzo": args.logpzo_variant}

    panels = [(p.split(":", 1)[0], p.split(":", 1)[1]) for p in args.panels]
    ncol = 2 if len(panels) > 1 else 1
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.8 * nrow), squeeze=False)
    drawn = set()

    for ax, (tag, title) in zip(axes.ravel(), panels):
        blob = json.load(open(os.path.join(args.data_dir, f"{tag}.json")))
        rows, H = blob["rows"], blob["horizon"]
        nS, nF = blob["n_success"], blob["n_failure"]

        # Only worth drawing when episodes still have different lengths. If they were all trimmed
        # to the same length, this line would sit on top of the right-hand edge and mean nothing.
        x_succ = blob["l_succ"] / H
        if x_succ < 0.98:
            drawn.add("_lsucc")
            ax.axvline(x_succ, color="#4a3aa7", lw=1.5, ls=(0, (5, 4)), zorder=2, alpha=0.85)
            ax.annotate("mean success\nlength", (x_succ, 0.515), textcoords="offset points",
                        xytext=(5, 0), ha="left", va="bottom", fontsize=7.8, color="#4a3aa7")

        for fam in ORDER:
            sub = select(rows, fam, args.protocol, args.thrifty_variant, QV)
            if not sub and fam in ("absolute", "timeout"):
                # The dumb baselines have nothing to tune, so there are no cross-validated rows for
                # them. Fall back to the plain ones rather than dropping the curve.
                sub = select(rows, fam, "insample", args.thrifty_variant, QV)
            pts = thin(front(sub))
            if not pts:
                continue
            drawn.add(fam)
            s = STYLE[fam]
            ax.plot([p[1] for p in pts], [p[0] for p in pts], color=s["color"], lw=s["lw"],
                    ls=s["ls"], marker=s["marker"], ms=s["ms"], alpha=s["alpha"],
                    zorder=s["z"], mec="white", mew=0.9, clip_on=True)

        ax.set_title(f"{title}   ({nS} success / {nF} failure)", fontsize=11.5, color=INK,
                     loc="left", pad=8)
        ax.set_xlim(0.0, args.xmax)
        ax.set_ylim(0.48, 1.03)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced accuracy", fontsize=10, color=INK2)
    for ax in axes[nrow - 1, :]:
        ax.set_xlabel("Average detection time (normalized)", fontsize=10, color=INK2)

    handles = [plt.Line2D([], [], color=STYLE[f]["color"], lw=STYLE[f]["lw"], ls=STYLE[f]["ls"],
                          marker=STYLE[f]["marker"], ms=STYLE[f]["ms"], mec="white", mew=0.9,
                          label=STYLE[f]["label"]) for f in ORDER if f in drawn]
    if "_lsucc" in drawn:
        handles.append(plt.Line2D([], [], color="#4a3aa7", lw=1.5, ls=(0, (5, 4)),
                                  label="Mean successful-episode length"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.005), labelcolor=INK2, handlelength=2.6,
               columnspacing=2.0)
    fig.suptitle("Failure detection: earlier is better at equal accuracy",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.02 + 0.032 * len(handles) / ncol, 1, 0.96))
    out = args.out if os.path.isabs(args.out) else os.path.join(args.data_dir, args.out)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
