#!/usr/bin/env python3
"""Report per-obs-key statistics from a robomimic-style HDF5 to check normalization.

Usage: python scripts/inspect_obs_stats.py /path/to/converted.hdf5 [max_demos]

Heuristic:
  - z-score normalized  -> mean ~ 0, std ~ 1
  - min-max [-1, 1]      -> min ~ -1, max ~ 1
  - min-max [0, 1]       -> min ~ 0,  max ~ 1
  - raw                  -> arbitrary, key-dependent scales
"""
import sys
import numpy as np
import h5py

path = sys.argv[1]
max_demos = int(sys.argv[2]) if len(sys.argv) > 2 else 25

with h5py.File(path, "r") as f:
    demos = list(f["data"].keys())[:max_demos]
    # Collect samples per obs key across the sampled demos.
    acc = {}
    for d in demos:
        obs = f["data"][d]["obs"]
        for k in obs.keys():
            arr = np.asarray(obs[k])
            # skip image/embedding tensors (high-dim); focus on low-dim proprio/state vectors
            if arr.ndim >= 3:
                continue
            acc.setdefault(k, []).append(arr.reshape(arr.shape[0], -1))

    print(f"File: {path}")
    print(f"Demos sampled: {len(demos)} (of {len(list(f['data'].keys()))})\n")
    print(f"{'key':<22}{'dim':>5}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}"
          f"{'per-dim std':>22}{'per-dim mean':>22}")
    print("-" * 119)
    for k in sorted(acc.keys()):
        x = np.concatenate(acc[k], axis=0).astype(np.float64)  # (N, D)
        d_mean = x.mean(axis=0)  # per-dimension mean
        d_std = x.std(axis=0)    # per-dimension std
        print(
            f"{k:<22}{x.shape[1]:>5}{x.mean():>9.3f}{x.std():>9.3f}{x.min():>9.3f}{x.max():>9.3f}"
            f"{f'[{d_std.min():.3f}, {d_std.max():.3f}]':>22}"
            f"{f'[{d_mean.min():.3f}, {d_mean.max():.3f}]':>22}"
        )
    print(
        "\nIf per-dim std spans a wide range (e.g. 0.01 .. >1) and per-dim mean is far from 0,\n"
        "the observations are RAW (unnormalized). z-score => per-dim mean~0, std~1."
    )
