#!/usr/bin/env python3
"""
Convert a robomimic dataset into the observation format used by the online robosuite env
(:func:`robometer_policy_learning.envs.robosuite_wrappers.setup_robomimic_env`), so the
dataset can be used for offline-to-online RL.

What it does per demo:
  * Builds ``obs/state`` (and ``next_obs/state``) = robosuite's 32-d ``robot0_proprio-state``
    vector, reconstructed by concatenating the dataset's stored proprio component obs
    (``robot0_joint_pos_cos``, ``robot0_eef_pos``, ...) in the exact order robosuite uses.
    With ``--include-object`` the ``object`` features are appended (matching the env's
    ``use_full_state=True``).
  * Copies the camera image obs (``agentview_image`` / ``robot0_eye_in_hand_image``)
    through unchanged. The online env is configured to emit these same image keys, so no
    image renaming is needed (see ``setup_robomimic_env``).
  * Copies ``actions`` / ``rewards`` / ``dones`` and preserves the ``data.attrs['env_args']``
    metadata (so the env can still be reconstructed from the converted file) and the
    ``mask`` group (train/valid splits) when present.

The proprio ordering is derived from the reconstructed env and *verified* to reproduce
``robot0_proprio-state`` exactly before any data is written.

Usage:
    uv run python scripts/convert_robomimic_to_aligned.py \
        --input /path/to/image.hdf5 --output /path/to/image_aligned.hdf5
    # state including object pose (matches env use_full_state=True):
    uv run python scripts/convert_robomimic_to_aligned.py -i in.hdf5 -o out.hdf5 --include-object
"""

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import h5py
import numpy as np
from loguru import logger

from robometer_policy_learning.envs.robosuite_wrappers import load_robomimic_env_metadata

PROPRIO_STATE_KEY = "robot0_proprio-state"
OBJECT_KEY = "object"
IMAGE_KEYS_DEFAULT = ["agentview_image", "robot0_eye_in_hand_image"]


def _build_lowdim_env(env_meta: dict):
    """Reconstruct the robosuite env from metadata with rendering disabled (low-dim only)."""
    import robosuite as suite

    env_kwargs = deepcopy(env_meta.get("env_kwargs", {}))
    env_name = env_meta["env_name"]
    env_kwargs.pop("env_name", None)
    # We only need proprioceptive (+ object) state; disable all rendering for speed.
    env_kwargs.update(
        dict(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=False,
            ignore_done=True,
            hard_reset=False,
        )
    )
    env_kwargs.pop("camera_names", None)
    env_kwargs.pop("camera_heights", None)
    env_kwargs.pop("camera_widths", None)
    env_kwargs.pop("camera_depths", None)
    return suite.make(env_name, **env_kwargs)


def _determine_proprio_order(base_env, reference_state: np.ndarray) -> List[str]:
    """
    Determine the ordered list of proprio component obs keys whose concatenation equals
    robosuite's ``robot0_proprio-state``, evaluated at a real (non-reset) sim state so the
    component values are distinct.
    """
    base_env.reset()
    base_env.sim.set_state_from_flattened(reference_state)
    base_env.sim.forward()
    di = base_env._get_observations(force_update=True)

    if PROPRIO_STATE_KEY not in di:
        raise KeyError(
            f"'{PROPRIO_STATE_KEY}' not found in env observations (have: {sorted(di.keys())})."
        )
    proprio = np.asarray(di[PROPRIO_STATE_KEY]).reshape(-1).astype(np.float64)

    # Candidate component keys: robot proprio observables (exclude the concatenation itself).
    pool: Dict[str, np.ndarray] = {}
    for k, v in di.items():
        if k == PROPRIO_STATE_KEY or not k.startswith("robot0_"):
            continue
        arr = np.asarray(v).reshape(-1).astype(np.float64)
        if arr.size > 0:
            pool[k] = arr

    order: List[str] = []
    ptr = 0
    while ptr < proprio.size:
        match = None
        for k, v in pool.items():
            L = v.size
            if ptr + L <= proprio.size and np.allclose(proprio[ptr : ptr + L], v, atol=1e-6, rtol=0):
                match = (k, L)
                break
        if match is None:
            raise RuntimeError(
                f"Could not match proprio component at offset {ptr}/{proprio.size}. "
                f"Remaining candidates: {sorted(pool.keys())}"
            )
        k, L = match
        order.append(k)
        ptr += L
        del pool[k]

    logger.info(f"Determined robot0_proprio-state composition ({proprio.size}-d): {order}")
    return order


def _verify_against_dataset(
    f: h5py.File, demo: str, proprio_order: List[str], base_env, t: int = 0
) -> int:
    """
    Verify that concatenating the dataset's stored proprio components (in ``proprio_order``)
    at timestep ``t`` reproduces the env's ``robot0_proprio-state`` for the same sim state.
    Returns the proprio state dimension.
    """
    grp = f["data"][demo]
    obs = grp["obs"]
    states = np.asarray(grp["states"])

    base_env.reset()
    base_env.sim.set_state_from_flattened(states[t])
    base_env.sim.forward()
    env_proprio = np.asarray(
        base_env._get_observations(force_update=True)[PROPRIO_STATE_KEY]
    ).reshape(-1).astype(np.float32)

    dataset_proprio = np.concatenate(
        [np.asarray(obs[k][t], dtype=np.float32).reshape(-1) for k in proprio_order]
    )

    if dataset_proprio.shape != env_proprio.shape:
        raise RuntimeError(
            f"Proprio dim mismatch: dataset {dataset_proprio.shape} vs env {env_proprio.shape}"
        )
    max_err = float(np.max(np.abs(dataset_proprio - env_proprio)))
    if not np.allclose(dataset_proprio, env_proprio, atol=1e-4, rtol=0):
        raise RuntimeError(
            f"Dataset proprio components do not reproduce env robot0_proprio-state "
            f"(max abs err {max_err:.3e}). Cannot guarantee offline/online alignment."
        )
    logger.info(f"Verified state alignment on {demo} t={t} (max abs err {max_err:.3e}).")
    return int(env_proprio.size)


def _compose_state(obs_group: h5py.Group, proprio_order: List[str], include_object: bool) -> np.ndarray:
    parts = [np.asarray(obs_group[k], dtype=np.float32).reshape(len(obs_group[k]), -1) for k in proprio_order]
    if include_object:
        if OBJECT_KEY not in obs_group:
            raise KeyError(f"--include-object set but '{OBJECT_KEY}' not in obs group.")
        parts.append(np.asarray(obs_group[OBJECT_KEY], dtype=np.float32).reshape(len(obs_group[OBJECT_KEY]), -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def convert(
    input_path: str,
    output_path: str,
    include_object: bool = False,
    image_keys: List[str] = IMAGE_KEYS_DEFAULT,
    max_demos: int = None,
):
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("output must differ from input")

    env_meta = load_robomimic_env_metadata(input_path)
    base_env = _build_lowdim_env(env_meta)

    with h5py.File(input_path, "r") as fin:
        demos = sorted(fin["data"].keys(), key=lambda d: int(d.split("_")[-1]) if d.split("_")[-1].isdigit() else d)
        if not demos:
            raise ValueError("No demos found under /data")
        first = demos[0]

        # Determine proprio composition from the env (using a real dataset state), then verify.
        ref_state = np.asarray(fin["data"][first]["states"])[0]
        proprio_order = _determine_proprio_order(base_env, ref_state)
        state_dim = _verify_against_dataset(fin, first, proprio_order, base_env, t=0)
        if include_object:
            obj_dim = int(np.asarray(fin["data"][first]["obs"][OBJECT_KEY]).shape[-1])
            logger.info(f"Including object ({obj_dim}-d): final state dim = {state_dim + obj_dim}")

        if max_demos is not None:
            demos = demos[:max_demos]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            data_out = fout.create_group("data")
            # Preserve top-level data attrs (env_args, etc.) so the env can be reconstructed.
            for ak, av in fin["data"].attrs.items():
                data_out.attrs[ak] = av

            total = 0
            for di, demo in enumerate(demos):
                gin = fin["data"][demo]
                gout = data_out.create_group(demo)
                for ak, av in gin.attrs.items():
                    gout.attrs[ak] = av

                n = int(gin["actions"].shape[0])
                total += n

                # Copy actions / rewards / dones / states verbatim.
                for key in ["actions", "rewards", "dones", "states"]:
                    if key in gin:
                        gin.copy(key, gout)

                # obs / next_obs: synthesized state + passthrough images.
                for obs_name in ["obs", "next_obs"]:
                    if obs_name not in gin:
                        continue
                    obs_in = gin[obs_name]
                    obs_out = gout.create_group(obs_name)
                    obs_out.create_dataset(
                        "state",
                        data=_compose_state(obs_in, proprio_order, include_object),
                        compression="gzip",
                    )
                    for img in image_keys:
                        if img in obs_in:
                            obs_in.copy(img, obs_out)

                if (di + 1) % 20 == 0 or di == len(demos) - 1:
                    logger.info(f"  converted {di + 1}/{len(demos)} demos")

            data_out.attrs["total"] = total

            # Preserve train/valid split masks if present.
            if "mask" in fin:
                fin.copy("mask", fout)

    logger.info(
        f"✓ Wrote aligned dataset to {output_path} "
        f"({len(demos)} demos, state_dim={state_dim + (obj_dim if include_object else 0)}, "
        f"images={image_keys})"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="Path to the source robomimic .h5 dataset.")
    p.add_argument("-o", "--output", required=True, help="Path to write the aligned .h5 dataset.")
    p.add_argument(
        "--include-object",
        action="store_true",
        help="Append object pose features to `state` (matches env use_full_state=True).",
    )
    p.add_argument(
        "--image-keys",
        nargs="*",
        default=IMAGE_KEYS_DEFAULT,
        help="Camera image obs keys to copy through unchanged.",
    )
    p.add_argument("--max-demos", type=int, default=None, help="Convert only the first N demos (debug).")
    args = p.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        include_object=args.include_object,
        image_keys=args.image_keys,
        max_demos=args.max_demos,
    )


if __name__ == "__main__":
    main()
