"""
Inspect a robomimic ``.h5`` dataset: print the stored environment metadata and,
optionally, reconstruct the environment and roll out a few random steps to verify the
observation format used by this repo (``state`` / ``image`` / ``wrist_image``).

Usage:
    uv run python scripts/check_robomimic_dataset.py /path/to/dataset.h5
    uv run python scripts/check_robomimic_dataset.py /path/to/dataset.h5 --build
"""

import argparse
import json

import numpy as np

from robometer_policy_learning.envs.robosuite_wrappers import (
    load_robomimic_env_metadata,
    setup_robomimic_env,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", help="Path to the robomimic .h5 demonstration file.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also reconstruct the env from metadata and roll out a few random steps.",
    )
    parser.add_argument("--image-size", type=int, default=84, help="Rendered camera size (square).")
    parser.add_argument("--steps", type=int, default=5, help="Random steps to take when --build is set.")
    args = parser.parse_args()

    env_meta = load_robomimic_env_metadata(args.dataset_path)
    # INSERT_YOUR_CODE
    import h5py

    print("\nLoading structure and shapes for the first demo in data/demo_0...\n")
    with h5py.File(args.dataset_path, "r") as f:
        data_group = f["data"]
        if "demo_0" in data_group:
            demo_group = data_group["demo_0"]
            def print_h5_structure(h5obj, indent=0):
                for key in h5obj:
                    item = h5obj[key]
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        dtype = item.dtype
                        print("  " * indent + f"{key}: Dataset, shape={shape}, dtype={dtype}")
                    elif isinstance(item, h5py.Group):
                        print("  " * indent + f"{key}/: Group")
                        print_h5_structure(item, indent + 1)
            print("data/demo_0/: Group")
            print_h5_structure(demo_group, 2)

            # Calculate average demo length
            demo_lengths = []
            for demo_key in data_group:
                demo = data_group[demo_key]
                # Assume 'actions' exists and its first dimension is the length
                if "actions" in demo:
                    demo_lengths.append(demo["actions"].shape[0])
            if demo_lengths:
                avg_length = sum(demo_lengths) / len(demo_lengths)
                print(f"\nNumber of demos: {len(demo_lengths)}")
                print(f"Average demo length: {avg_length:.2f} steps")
            else:
                print("No 'actions' datasets found for demo length calculation.")
        else:
            print("The dataset does not contain data/demo_0.")
    print("env_name:", env_meta.get("env_name"))
    print("type:    ", env_meta.get("type"))
    print("env_kwargs:")
    print(json.dumps(env_meta.get("env_kwargs", {}), indent=2, default=str))

    if not args.build:
        return

    env, remove_obs_keys = setup_robomimic_env(
        dataset_path=args.dataset_path,
        n_envs=1,
        image_size=args.image_size,
        max_episode_steps=max(args.steps + 1, 50),
    )
    print("\nremove_obs_keys:", remove_obs_keys)

    obs, info = env.reset(seed=0)
    print("reset observation:")
    for k in sorted(obs.keys()):
        v = np.asarray(obs[k])
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    print("language_instruction:", info.get("language_instruction"))

    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"\nafter {args.steps} random steps -> reward={np.asarray(reward)} success={info.get('success')}")
    env.close()


if __name__ == "__main__":
    main()
