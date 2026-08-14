#!/usr/bin/env python3
"""Evaluation: score frontier transfer as one number per task pair, and draw the figure.

Run the source's whole frontier on the target, draw both curves in (balanced accuracy, detection
time), measure the area under each and divide:
    ratio = area(transferred) / area(native)

Usage:
    uv run python scripts/gate_transfer_hv.py --source t1 --target t8
    uv run python scripts/gate_transfer_hv.py --all
    uv run python scripts/gate_transfer_hv.py --plot     # all + plotting the 4x4 grid of curves
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_transfer import FRONT_DIR, TASKS, Corpus, as_cfg, cfg_str, load_front  # noqa: E402

# The worst possible detector metrics; used for calculating hypervolume
REF_BALACC = 0.5
REF_TDET = 1.0


def nondominated(pts):
    """From a list of (balacc, tdet), keep the configs nothing else beats on both at once."""
    pts = sorted(pts, key=lambda p: (p[1], -p[0]))
    out, best = [], -np.inf
    for b, t in pts:
        if b > best + 1e-12:
            best = b
            out.append((b, t))
    return out


def common_cap(*point_sets):
    """The latest detection time any of these curves reaches. Used instead of 1.0 as the right edge.

    At 1.0 both curves get credit for a wide empty strip from their last config out to the horizon.
    To ensure faithfulness we do not count that area under the hypervolume ratio.
    """
    ts = [t for pts in point_sets for _, t in pts if np.isfinite(t)]
    return max(ts) if ts else REF_TDET


def hypervolume(pts, ref_b=REF_BALACC, ref_t=REF_TDET):
    front = [(b, t) for b, t in nondominated(pts) if b > ref_b and t < ref_t]
    if not front:
        return 0.0
    hv = 0.0
    for i, (b, t) in enumerate(front):
        t_next = front[i + 1][1] if i + 1 < len(front) else ref_t
        hv += (b - ref_b) * (t_next - t)
    return hv


def transfer_points(source, target_corpus):
    """Score every config on the source's frontier against the target task's episodes."""
    src_front = load_front(source)
    rows = []
    for r in src_front:
        c = as_cfg(r)
        rows.append(dict(cfg=c, src_balacc=r["balacc"], src_tdet=r["avg_tdet"],
                         **target_corpus.evaluate(c)))
    return rows


def run(source, target, corpora, verbose=True):
    """Area ratio for one source -> target pair."""
    tgt = corpora[target]
    native_front = load_front(target)
    native_pts = [(r["balacc"], r["avg_tdet"]) for r in native_front]
    rows = transfer_points(source, tgt)
    trans_pts = [(r["balacc"], r["avg_tdet"]) for r in rows]

    cap = common_cap(nondominated(native_pts), nondominated(trans_pts))
    hv_n = hypervolume(native_pts, ref_t=cap)
    hv_t = hypervolume(trans_pts, ref_t=cap)
    ratio = hv_t / hv_n if hv_n > 0 else float("nan")

    if verbose:
        print("=" * 96)
        print(f"{source} -> {target}")
        print(f"  target corpus: {int(tgt.succ.sum())} success / {int((~tgt.succ).sum())} failure")
        print("=" * 96)
        print(f"  native frontier   : {len(native_front):3d} configs, "
              f"{len(nondominated(native_pts)):3d} non-dominated,  HV = {hv_n:.4f}")
        print(f"  transferred       : {len(rows):3d} configs, "
              f"{len(nondominated(trans_pts)):3d} non-dominated,  HV = {hv_t:.4f}")
        print(f"\n  HYPERVOLUME RATIO = {ratio:.3f}   "
              f"{'PASS (>=0.85)' if ratio >= 0.85 else 'FAIL (<0.85)'}")

        nd_t = nondominated(trans_pts)
        # Each transferred config against the fastest native config that is at least as accurate.
        print(f"\n  transferred ({len(nd_t)} configs) vs native at the same balanced accuracy:")
        print(f"    {'balacc':>8}{'tdet':>8}   {'native tdet':>12}{'gap':>8}")
        for b, t in nd_t:
            ok = [r["avg_tdet"] for r in native_front if r["balacc"] >= b - 1e-9]
            nt = min(ok) if ok else float("nan")
            gap = f"{100*(t/nt-1):+.0f}%" if np.isfinite(nt) else "n/a"
            ns = f"{nt:.3f}" if np.isfinite(nt) else "  --"
            print(f"    {b:>8.3f}{t:>8.3f}   {ns:>12}{gap:>8}")
        print(f"\n  best bal-acc: native {max(p[0] for p in native_pts):.3f}  "
              f"transferred {max(p[0] for p in trans_pts):.3f}")
    return dict(source=source, target=target, hv_native=hv_n, hv_transferred=hv_t, ratio=ratio,
                cap=cap, rows=rows, native_front=native_front,
                native_pts=nondominated(native_pts), trans_pts=nondominated(trans_pts))


# --- plotting ---
NATIVE, TRANSFER = "#2a78d6", "#eb6834"
RULE = "#8f8bb5"
DIAG = "#a3a19c"
DIAG_BG = "#faf9f8"
INK, INK2, INK3 = "#14140f", "#5c5b57", "#8a8985"
GRID = "#eeedea"
NPTS = 9


def thin(pts, n=NPTS):
    """Keep about `n` evenly spaced configs, so a panel is readable."""
    if len(pts) <= n:
        return pts
    idx = sorted(set(np.linspace(0, len(pts) - 1, n).round().astype(int)))
    return [pts[i] for i in idx]


def _draw(ax, pts, color, ls, mk, lw=2.0):
    ax.plot([p[1] for p in pts], [p[0] for p in pts], color=color, lw=lw, ls=ls,
            marker=mk, ms=4.2, mec="white", mew=0.9, zorder=6, clip_on=False)


def plot_grid(corpora, results, out_path):
    """Draw every source -> target pair. Ratios come from the full curves, not the thinned ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_pair = {(r["source"], r["target"]): r for r in results}
    n = len(TASKS)
    fig, axes = plt.subplots(n, n, figsize=(2.45 * n + 1.0, 2.15 * n + 1.5),
                             sharex=True, sharey=True)

    # Crop the x-axis to the data. Running it out to 1.0 squashed every curve into the left third.
    diag_pts = {t: nondominated([(r["balacc"], r["avg_tdet"]) for r in load_front(t)])
                for t in TASKS}
    xmax = max([r["cap"] for r in results] + [p[-1][1] for p in diag_pts.values()]) + 0.04

    ratios = []
    for i, src in enumerate(TASKS):
        for j, tgt in enumerate(TASKS):
            ax = axes[i, j]
            diag = src == tgt
            if diag:
                ax.set_facecolor(DIAG_BG)

            x_succ = corpora[tgt].l_succ / corpora[tgt].horizon
            ax.axvline(x_succ, color=RULE, lw=1.0, ls=(0, (3, 3)), alpha=0.9, zorder=2)

            if diag:
                _draw(ax, thin(diag_pts[tgt]), DIAG, "-", "o", lw=1.8)
                ax.text(0.5, 0.06, "reference", transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=8.5, color=INK3, style="italic")
            else:
                r = by_pair[(src, tgt)]
                ratios.append(r["ratio"])
                # No area fills: the two curves and the gap between them carry the comparison,
                # and shading would compete with them.
                _draw(ax, thin(r["native_pts"]), NATIVE, "-", "o")
                _draw(ax, thin(r["trans_pts"]), TRANSFER, "--", "s", lw=1.8)

            ax.set_xlim(0.0, xmax)
            ax.set_ylim(0.48, 1.04)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.grid(True, axis="y", color=GRID, lw=0.8)     # horizontal only: y is the comparison
            ax.grid(False, axis="x")
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=INK3, labelsize=8.5, length=0)
            if i == 0:
                ax.set_title(tgt, fontsize=12, color=INK, pad=10, fontweight="medium")
            if j == 0:
                ax.set_ylabel(src, fontsize=12, color=INK, labelpad=10, rotation=0,
                              va="center", ha="right")

    fig.text(0.5, 0.075, "Average detection time (normalized)  →  later",
             fontsize=10, color=INK2, ha="center")
    fig.text(0.012, 0.5, "Balanced accuracy", fontsize=10, color=INK2, va="center", rotation=90)
    fig.text(0.5, 0.945, "columns: task the gate is evaluated on      "
                         "rows: task the gate was tuned on",
             fontsize=9.5, color=INK3, ha="center")

    handles = [
        Line2D([], [], color=NATIVE, lw=2.0, marker="o", ms=4.2, mec="white", mew=0.9,
               label="Tuned on this task"),
        Line2D([], [], color=TRANSFER, lw=1.8, ls="--", marker="s", ms=4.2, mec="white",
               mew=0.9, label="Transferred from the row's task"),
        Line2D([], [], color=RULE, lw=1.0, ls=(0, (3, 3)), label="Mean successful-episode length"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.005), labelcolor=INK2, handlelength=2.4, columnspacing=2.4)
    fig.suptitle("Gate calibration transfers between tasks without retuning",
                 fontsize=14, color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0.035, 0.10, 1, 0.935))
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"-> {out_path}   (median ratio {np.median(ratios):.3f}, min {min(ratios):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="t1")
    ap.add_argument("--target", default="t8")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--save", default="outputs/gate_frontier/transfer_hv.json")
    ap.add_argument("--plot", action="store_true", help="also draw the 4x4 grid of curves")
    ap.add_argument("--out", default=None, help="where to write the figure")
    args = ap.parse_args()

    all_pairs = args.all or args.plot
    tasks = TASKS if all_pairs else sorted({args.source, args.target})
    corpora = {t: Corpus(t) for t in tasks}

    results = []
    if all_pairs:
        for s in TASKS:
            for t in TASKS:
                if s != t:
                    results.append(run(s, t, corpora, verbose=False))
        print("=== area ratio, source -> target ===")
        hdr = "src\\tgt"
        print(f"  {hdr:<9}" + "".join(f"{t:>8}" for t in TASKS))
        for s in TASKS:
            cells = ""
            for t in TASKS:
                if s == t:
                    cells += f"{'-':>8}"
                else:
                    r = next(x for x in results if x["source"] == s and x["target"] == t)
                    cells += f"{r['ratio']:>8.3f}"
            print(f"  {s:<9}{cells}")
        vals = [r["ratio"] for r in results]
        print(f"\n  median {np.median(vals):.3f}   mean {np.mean(vals):.3f}   "
              f"pass rate (>=0.85) {sum(v >= 0.85 for v in vals)}/{len(vals)}")
    else:
        results.append(run(args.source, args.target, corpora))

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    json.dump([{k: v for k, v in r.items()
                if k not in ("rows", "native_front", "native_pts", "trans_pts")} | dict(
        transferred=[{kk: vv for kk, vv in row.items() if kk != "cfg"} | {"cfg": cfg_str(row["cfg"])}
                     for row in r["rows"]],
        native=[{"balacc": x["balacc"], "avg_tdet": x["avg_tdet"]} for x in r["native_front"]])
        for r in results], open(args.save, "w"), indent=1)
    print(f"\n-> {args.save}")

    if args.plot:
        plot_grid(corpora, results,
                  args.out or os.path.join(FRONT_DIR, "transfer_frontier_grid.png"))


if __name__ == "__main__":
    main()
