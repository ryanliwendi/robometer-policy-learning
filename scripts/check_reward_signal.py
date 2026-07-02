#! /usr/bin/env python3

"""Feasibility check: does Robometer's progress/success signal real task
progress on LIBERO? Rolls out policies of varying quality and plots progress vs step"""

import os
# MuJoCo must render headless on the cluster; set this BEFORE importing the env/torch.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # no display on the cluster -> write PNGs instead of showing
import matplotlib.pyplot as plt
import h5py

# --- make the Robometer reward wrapper and LIBERO importable ---
# The wrapper lives in robometer's scripts/ dir, which is NOT an installed package, so we add it
# to sys.path. Use the SAME robometer copy that's importable as a package (the vendored one under
# robometer_policy_learning/robometer) so the wrapper's internal `from robometer...` imports resolve
# against the same codebase.
sys.path.insert(0, "/scr/liryan/robometer_policy_learning/robometer/scripts")
sys.path.insert(0, "/scr/liryan/LIBERO")

from example_libero_robometer_wrapper import LiberoRobometerRewardWrapper
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import Task
from libero.libero.envs import OffScreenRenderEnv

# --- constants ---
MODEL_PATH  = "robometer/Robometer-4B"
TASK_SUITE  = "libero_10"
TASK_ID     = 3
REWARD_KEYS = ["agentview_image"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DEMO_H5 = "/scr/liryan/LIBERO/libero/datasets/libero_10_only_successful/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5"


def build_reward_env():
    """Build the LIBERO env for TASK_ID, wrapped so Robometer scores progress/success each step.
    Returns (env, language_instruction)."""
    # 1. look up the task and its scene-definition (bddl) file
    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()  # a LIBERO_10 object
    task: Task = suite.get_task(TASK_ID)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    # 2. build the headless LIBERO sim (256x256 to match the training render)
    base_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    base_env.seed(0)

    # 3. wrap it: every step() now also runs Robometer on the accumulated frames and puts
    #    predicted_reward (progress) + success_prob into info.
    env = LiberoRobometerRewardWrapper(
        base_env,
        model_path=MODEL_PATH,
        device=DEVICE,
        reward_relabeling_keys=REWARD_KEYS,
        add_estimated_reward=False,  # return the pure predicted progress, without the environment reward
    )

    return env, task.language


def load_demo(h5_path=DEMO_H5, demo_key=None):
    with h5py.File(h5_path, "r") as f:
        demo_key = demo_key or list(f["data"].keys())[0]
        g = f["data"]["demo_key"]
        return np.array(g["states"][0]), np.array(g["actions"])


def random_action(obs):
    """Trivial policy: uniform random 7-D action (6 DoF + gripper). A guaranteed-failing rollout."""
    return np.random.uniform(-1.0, 1.0, size=7)


def run_rollout(env, action_fn, max_steps=50, label=""):
    """Step the env with actions from action_fn(obs), recording Robometer's per-step signal.
    Returns {progress[], success_prob[], true_success, label}."""
    obs, info = env.reset()
    progress, success_prob = [], []
    true_success = False
    for t in range(max_steps):
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        progress.append(float(info["predicted_reward"]))
        success_prob.append(float(info["success_prob"]))
        if info.get("success", False):
            true_success = True
        if terminated or truncated: 
            break
    return {
        "progress": progress,
        "success_prob": success_prob,
        "true_success": true_success,
        "label": label,
    }


def run_demo_rollout(env, init_state, actions, label="demo"):
    env.reset()
    env.set_init_state(init_state)
    env._frames = {k: [] for k in env.reward_relabeling_keys}
    progress, success_prob, true_success = [], [], False
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        progress.append(float(info["predicted_reward"]))
        success_prob.append(float(info["success_prob"]))
        if info.get("success", False):
            true_success = True
        if terminated or truncated:
            break
        return {
            "progress": progress,
            "success_prob": success_prob,
            "true_success": true_success,
            "label": label,
        }


def plot_results(results, out_path="reward_signal_check.png"):
    """Plot Robometer progress and success_prob vs step for each rollout"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for r in results: 
        steps = range(len(r["progress"]))
        ax1.plot(steps, r["progress"], label=f'{r["label"]} (true_success={r["true_success"]})')
        ax2.plot(steps, r["success_prob"], label=r["label"])
    ax1.set(title="Robometer progress vs step", xlabel="step", ylabel="predicted progress")
    ax2.set(title="Robometer P(success) vs step", xlabel="step", ylabel="success_prob")
    ax1.legend(); ax2.legend()
    fig.tight_layout()
    fig.save_fig(out_path, dpi=120)
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    env, language = build_reward_env()
    print("Task:", language)
    
    # failure baseline
    rand = run_rollout(env, random_action, max_steps=30, label="random")

    # success trajectory: replay a successful demo
    init_states, actions = load_demo()
    demo = run_demo_rollout(env, init_state, actions, label="demo (success)")
    print("demo true_success:", demo["true_success"])
    
    env.close()
    plot_results([demo, rand])
