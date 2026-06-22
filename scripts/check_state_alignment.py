#!/usr/bin/env python3
"""Verify the dataset's obs/state matches the online env's composed `state`.

Reconstructs the robosuite env from the dataset metadata, replays stored sim states, and
compares the env-composed `state` (robot0_proprio-state [+ object-state]) against the
dataset's stored obs/state[t]. This checks the FULL composed vector incl. the object part,
which convert_robomimic_to_aligned.py only partially verifies (proprio only).
"""
import sys
import numpy as np
import h5py

from robometer_policy_learning.envs.robosuite_wrappers import load_robomimic_env_metadata
from scripts.convert_robomimic_to_aligned import _build_lowdim_env

PROPRIO_KEY = "robot0_proprio-state"
OBJECT_KEY = "object-state"

path = sys.argv[1]
n_demos = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ts_per_demo = 5

env = _build_lowdim_env(load_robomimic_env_metadata(path))

with h5py.File(path, "r") as f:
    demos = list(f["data"].keys())[:n_demos]
    ds_state_dim = int(np.asarray(f["data"][demos[0]]["obs"]["state"]).shape[-1])

    # Probe env dims to infer whether `state` includes the object.
    env.reset()
    env.sim.set_state_from_flattened(np.asarray(f["data"][demos[0]]["states"])[0])
    env.sim.forward()
    di = env._get_observations(force_update=True)
    proprio_dim = int(np.asarray(di[PROPRIO_KEY]).reshape(-1).shape[0])
    obj_dim = int(np.asarray(di[OBJECT_KEY]).reshape(-1).shape[0]) if OBJECT_KEY in di else 0
    includes_object = ds_state_dim == proprio_dim + obj_dim

    state_keys = [PROPRIO_KEY] + ([OBJECT_KEY] if includes_object else [])
    print(f"File: {path}")
    print(f"  dataset state_dim = {ds_state_dim} | env proprio = {proprio_dim}, object = {obj_dim}")
    print(f"  inferred composition: {state_keys}  (includes_object={includes_object})")
    if ds_state_dim not in (proprio_dim, proprio_dim + obj_dim):
        print("  ✗ dataset state_dim matches neither proprio nor proprio+object — MISALIGNED dim")
        sys.exit(1)

    max_err_all = 0.0
    for demo in demos:
        states = np.asarray(f["data"][demo]["states"])
        ds_state = np.asarray(f["data"][demo]["obs"]["state"])
        T = min(len(states), len(ds_state))
        idxs = np.linspace(0, T - 1, min(ts_per_demo, T)).astype(int)
        for t in idxs:
            env.reset()
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            di = env._get_observations(force_update=True)
            env_state = np.concatenate(
                [np.asarray(di[k], dtype=np.float32).reshape(-1) for k in state_keys]
            )
            err = float(np.max(np.abs(env_state - ds_state[t].astype(np.float32))))
            max_err_all = max(max_err_all, err)
    aligned = max_err_all < 1e-4
    print(f"  max abs err (env vs dataset state) over {len(demos)} demos: {max_err_all:.3e}")
    print("  ✓ ALIGNED" if aligned else "  ✗ MISALIGNED (values differ)")
    sys.exit(0 if aligned else 1)
