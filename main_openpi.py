# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import math
import os
import pickle
import signal
import sys
import threading
import time
import urllib.request
from collections import deque
from copy import deepcopy
from pathlib import Path

from moviepy import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.misc.time import time_ms
from droid.robot_env import RobotEnv
import tqdm
import tyro

TELEOP_ROOT = Path("/home/franca_glamor/Documents/Harshitha_gesture_teleop")
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from gripper_utils import open_local_gripper_if_configured, send_gripper_command_openpi  # noqa: E402

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "33790348"  # e.g., "24259877"
    right_camera_id: str = "37998989"  # e.g., "24514023"
    wrist_camera_id: str = "14064085"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 900
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # When the NUC already runs scripts/server/run_server.py (with Polymetis up), leave this False.
    # True relaunches Polymetis from the workstation and can break an existing session on re-run.
    launch_robot: bool = False

    # ===== Reward-DAgger (added) =====
    # Robometer server (on the lab server, reached over the network).
    reward_host: str = "localhost"
    reward_port: int = 8900
    reward_url: str | None = None       # full URL; overrides host/port. Use for a Pinggy HTTP(S) tunnel.
    reward_timeout: float = 20.0
    score_every: int = 5                # env steps between Robometer calls (VLM can't keep up with 15 Hz)
    frame_width: int = 640              # resize the scored frame to match the offline traces (640x360)
    frame_height: int = 360
    # Gate (windows in SCORER TICKS = score_every env steps). Placeholders -- set from sweep_gate_real_world.py.
    method: str = "spearman"            # "spearman" | "pearson" | "naive"
    short_window: int = 10
    long_window: int = 40
    drop_threshold: float = -0.8
    plateau_threshold: float = 0.1
    min_drop_magnitude: float = 0.15
    smoothing: float = 0.5
    warmup: int = 10                    # env steps before the gate may fire
    # What happens when the gate fires:
    #   "teleop" = hand to the Quest human (needs Quest + the two VERIFY spots in human_takeover);
    #   "stop"   = end the episode at the fire (NO Quest) -- validates the full detect pipeline;
    #   "shadow" = log the fire and let pi0 keep driving (NO Quest) -- see ALL fire points per episode.
    handoff_mode: str = "teleop"
    teleop_action_space: str = "cartesian_velocity"


def _ensure_nuc_robot_ready(robot) -> None:
    """Ensure the NUC run_server FrankaRobot stub is connected to Polymetis."""
    try:
        robot.get_ee_pose()
        return
    except Exception:
        print("[robot] NUC server not connected to Polymetis; calling launch_robot ...")

    try:
        robot.launch_robot()
        time.sleep(2.0)
        robot.get_ee_pose()
        print("[robot] Connected to Polymetis.")
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to the Franka arm on the NUC.\n\n"
            "On the NUC, start Polymetis first, then the RPC server (in separate terminals):\n"
            "  conda activate polymetis-local\n"
            "  launch_robot.py robot_client=franka_hardware\n"
            "  cd ~/Documents/droid && python scripts/server/run_server.py\n\n"
            "Then re-run this script from the workstation.\n"
            f"Underlying error: {exc}"
        ) from exc


def _open_gripper(env: RobotEnv, local_gripper, *, blocking: bool = True) -> None:
    if local_gripper is not None:
        send_gripper_command_openpi(None, 0.0, local_gripper=local_gripper, blocking=blocking)
        print("[gripper] Opened local Robotiq gripper.")
        return
    env._robot.update_gripper(0, velocity=False, blocking=blocking)
    print("[gripper] Opened gripper via robot server.")


def _reset_env(env: RobotEnv, local_gripper, *, randomize: bool = False) -> None:
    """DROID joint reset; gripper opens locally when Robotiq USB is on the workstation."""
    _open_gripper(env, local_gripper)
    if randomize:
        noise = np.random.uniform(low=env.randomize_low, high=env.randomize_high)
    else:
        noise = None
    env._robot.update_joints(env.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)


def _get_env_observation(env: RobotEnv, local_gripper):
    """Build env observation without querying a dead Polymetis gripper on the NUC."""
    if local_gripper is None:
        return env.get_observation()

    obs_dict = {"timestamp": {}}
    read_start = time_ms()
    ee_pose = np.asarray(env._robot.get_ee_pose())
    joint_positions = np.asarray(env._robot.get_joint_positions())
    state_dict = {
        "cartesian_position": ee_pose.tolist(),
        "gripper_position": float(local_gripper.get_position()),
        "joint_positions": joint_positions.tolist(),
        "joint_velocities": [0.0] * len(joint_positions),
    }
    obs_dict["robot_state"] = state_dict
    obs_dict["timestamp"]["robot_state"] = {"read_start": read_start, "read_end": time_ms()}

    camera_obs, camera_timestamp = env.read_cameras()
    obs_dict.update(camera_obs)
    obs_dict["timestamp"]["cameras"] = camera_timestamp
    obs_dict["camera_type"] = deepcopy(env.camera_type_dict)
    obs_dict["camera_extrinsics"] = env.get_camera_extrinsics(state_dict)

    intrinsics = {}
    for cam in env.camera_reader.camera_dict.values():
        cam_intr_info = cam.get_intrinsics()
        for full_cam_id, info in cam_intr_info.items():
            intrinsics[full_cam_id] = info["cameraMatrix"]
    obs_dict["camera_intrinsics"] = intrinsics
    return obs_dict


def _step_env(env: RobotEnv, action: np.ndarray, local_gripper) -> None:
    """Execute policy action. Local Robotiq USB skips NUC Polymetis gripper entirely."""
    if local_gripper is None:
        env.step(action)
        return

    send_gripper_command_openpi(None, action[-1], local_gripper=local_gripper, blocking=False)
    # Joint velocity integration runs on the NUC (already has RobotIKSolver); avoid loading MuJoCo here.
    env._robot.update_joints(action[:7], velocity=True, blocking=False)


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


# ===========================================================================
# Reward-DAgger: gate + async Robometer client + human handoff (added)
# ===========================================================================
def _pearson(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    return float((xm * ym).sum() / denom) if denom > 0 else float("nan")


def _rankdata(a):
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def _corr(values, method) -> float:
    n = len(values)
    if n < 2 or min(values) == max(values):
        return float("nan")
    y = _rankdata(values) if method == "spearman" else np.asarray(values, float)
    return _pearson(np.arange(n), y)


class RewardGate:
    """Fires on a sharp progress drop (short window) or a long plateau. Numpy-only (no scipy).
    Identical behavior to robometer_policy_learning/utils/reward_gate.py. reset() after a fire."""

    def __init__(self, short_window=10, drop_threshold=-0.8, long_window=40, plateau_threshold=0.1,
                 method="spearman", smoothing=0.0, min_drop_magnitude=0.0):
        self.short_window = short_window; self.drop_threshold = drop_threshold
        self.long_window = long_window; self.plateau_threshold = plateau_threshold
        self.method = method; self.smoothing = smoothing; self.min_drop_magnitude = min_drop_magnitude
        self.history = deque(maxlen=long_window); self._ema = None; self.last_trigger = None

    def update(self, progress: float) -> bool:
        self._ema = progress if self._ema is None else self.smoothing * self._ema + (1 - self.smoothing) * progress
        self.history.append(self._ema)
        if len(self.history) < self.short_window:
            return False
        if self.method == "naive":
            recent = list(self.history)[-self.short_window:]
            drop = max(recent) - recent[-1] >= self.drop_threshold
            plateau = (len(self.history) >= self.long_window
                       and (list(self.history)[-1] - list(self.history)[0]) <= self.plateau_threshold)
        else:
            drop = self._check_drop()
            plateau = self._check_plateau() if len(self.history) >= self.long_window else False
        fired = drop or plateau
        self.last_trigger = ("drop" if drop else "plateau") if fired else None
        return fired

    def _check_drop(self) -> bool:
        recent = list(self.history)[-self.short_window:]
        corr = _corr(recent, self.method)
        if math.isnan(corr) or corr >= self.drop_threshold:
            return False
        if self.min_drop_magnitude > 0 and (max(recent) - recent[-1]) < self.min_drop_magnitude:
            return False
        return True

    def _check_plateau(self) -> bool:
        corr = _corr(list(self.history)[-self.long_window:], self.method)
        return True if math.isnan(corr) else corr < self.plateau_threshold

    def reset(self):
        self.history.clear(); self._ema = None; self.last_trigger = None


class AsyncRobometerClient:
    """Scores off the control thread so the ~15 Hz loop never blocks on the VLM. The loop appends
    a frame each step and calls request() on its cadence; a daemon thread POSTs a subsampled prefix
    to the Robometer server and publishes (progress, version). Consume each new version once."""

    def __init__(self, url, max_frames=8, timeout=20.0):
        self.url = url; self.max_frames = int(max_frames); self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._frames, self._prompt, self._episode = [], "", 0
        self._progress, self._version = 0.0, 0
        self._req = threading.Event(); self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()

    def reset(self, prompt):
        with self._lock:
            self._frames = []; self._prompt = str(prompt); self._episode += 1; self._progress = 0.0

    def append(self, frame):
        with self._lock:
            self._frames.append(np.asarray(frame, dtype=np.uint8))

    def request(self):
        self._req.set()

    @property
    def latest(self):
        with self._lock:
            return self._progress, self._version

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url, timeout=5) as r:
                return r.read() == b"ok"
        except Exception:
            return False

    def _subsample(self, frames):
        n = len(frames)
        idx = range(n) if n <= self.max_frames else np.linspace(0, n - 1, self.max_frames).astype(int)
        return np.stack([frames[i] for i in idx], axis=0)

    def _post(self, frames, prompt) -> float:
        data = pickle.dumps({"frames": frames, "prompt": prompt})
        req = urllib.request.Request(self.url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return float(pickle.loads(r.read())["progress"])

    def _loop(self):
        while not self._stop:
            if not self._req.wait(timeout=0.2):
                continue
            self._req.clear()
            with self._lock:
                if not self._frames:
                    continue
                snap = list(self._frames); prompt, eid = self._prompt, self._episode
            try:
                progress = self._post(self._subsample(snap), prompt)
            except Exception as e:
                print(f"[robometer] request failed: {e}")
                continue
            with self._lock:
                if eid != self._episode:
                    continue
                self._progress, self._version = progress, self._version + 1

    def stop(self):
        self._stop = True; self._req.set()


def human_takeover(env, local_gripper, args, video=None, scorer=None, progress_trace=None, start_step=0) -> bool:
    """Operator finishes the episode by teleoperating with the Quest; returns success bool.

    ⚠️ VERIFY ON-RIG (1): the VRPolicy import + forward() signature, and which controller-info keys
        carry the operator's SUCCESS/FAILURE button presses.
    ⚠️ VERIFY ON-RIG (2): the action-space switch (eval runs joint_velocity, VRPolicy emits
        Cartesian) AND how the action is applied -- below uses env.step (DROID default), but your
        rig drives joints + local gripper via _step_env(); route it the same way if needed.
    Only reached when --handoff_mode teleop; the no-Quest modes never call this.
    """
    from droid.controllers.oculus_controller import VRPolicy

    controller = VRPolicy(right_controller=True)  # match your working teleop (scripts/main.py)
    # DROID computes self.DoF (7 cartesian / 8 joint) ONCE in __init__, but step() reads
    # self.action_space dynamically -- so switching the space requires syncing DoF too, or
    # step()'s `assert len(action) == self.DoF` fails (VRPolicy emits a 7-dim cartesian action).
    prev_space, prev_dof = env.action_space, env.DoF
    env.action_space = args.teleop_action_space
    env.DoF = 7 if "cartesian" in env.action_space else 8
    print("\n🙋 HUMAN TAKEOVER — teleoperate to finish. Press the Quest SUCCESS/FAILURE button "
          "(or Ctrl+C = failure) to end.")
    success = None
    tstep = int(start_step)
    _, last_version = scorer.latest if scorer is not None else (0.0, 0)
    try:
        while success is None:
            start = time.time()
            obs = _get_env_observation(env, local_gripper)
            action, info = controller.forward(obs, include_info=True)
            _btns = {k: v for k, v in info.items() if isinstance(v, (bool, np.bool_)) and v}
            if _btns:  # only prints on a button press -> quiet, and reveals the A/B key names
                print("[teleop] buttons:", _btns)
            # Record + score the exterior view through the takeover, so the graph shows the human
            # completing the task after the handoff (continuing the same causal history).
            ext = _extract_observation(args, obs, local_gripper=local_gripper)[f"{args.external_camera}_image"]
            if video is not None:
                video.append(ext)
            if scorer is not None:
                scorer.append(np.asarray(
                    Image.fromarray(ext).resize((args.frame_width, args.frame_height)), dtype=np.uint8))
                if tstep % args.score_every == 0:
                    scorer.request()
                progress, version = scorer.latest
                if version != last_version:
                    last_version = version
                    if progress_trace is not None:
                        progress_trace.append((tstep, float(progress)))
            tstep += 1
            action = np.asarray(action, dtype=np.float32)
            # Do NOT use env.step here: on this rig it reads the NUC gripper (dead -- the Robotiq
            # is on the workstation) and gRPC-crashes. Mirror _step_env instead: gripper -> local,
            # arm -> _robot directly (cartesian velocity for a 7-dim VRPolicy action = 6 + gripper).
            if local_gripper is not None:
                send_gripper_command_openpi(None, action[-1], local_gripper=local_gripper, blocking=False)
                env._robot.update_pose(action[:6], velocity=True, blocking=False)  # ⚠️ VERIFY method name
            else:
                env.step(action)
            if info.get("success"):
                success = True
            elif info.get("failure"):
                success = False
            elapsed = time.time() - start
            if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)
    except KeyboardInterrupt:
        success = False  # Ctrl+C ends the takeover
    finally:
        env.action_space, env.DoF = prev_space, prev_dof  # restore joint_velocity for pi0
    print(f"↩️  takeover ended (success={success})")
    return bool(success)


def save_progress_trace(progress_trace, fire_steps, meta, save_base, handoff_step=None):
    """Store one rollout's Robometer progress series: JSON (raw) + PNG (graph with gate fires).

    The trace continues through a human takeover, so the graph shows pi0's progress dropping,
    the handoff (green line), then the human recovering it. JSON matches the offline-trace shape
    (progress_trace + gate_fires) so the existing plot/analysis scripts can consume it.
    """
    import json

    payload = dict(meta, gate_fires=[int(s) for s in fire_steps], handoff_step=handoff_step,
                   progress_trace=[[int(s), float(p)] for s, p in progress_trace])
    with open(save_base + "_progress.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved progress data  -> {save_base}_progress.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [s for s, _ in progress_trace]
        prog = [p for _, p in progress_trace]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, prog, color="#1f77b4", lw=1.5, marker=".", ms=4)
        for f in fire_steps:
            ax.axvline(f, color="#d62728", lw=1.2, ls="--")
        if handoff_step is not None:  # mark where control passed to the human
            ax.axvline(handoff_step, color="#2ca02c", lw=2.0)
            ax.text(handoff_step, 0.98, " → human teleop", color="#2ca02c", fontsize=8, ha="left", va="top")
        ax.set_xlabel("env step"); ax.set_ylabel("Robometer progress"); ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{meta.get('instruction', '')}  success={meta.get('success')}  fires={fire_steps}")
        fig.tight_layout(); fig.savefig(save_base + "_progress.png", dpi=120); plt.close(fig)
        print(f"saved progress graph -> {save_base}_progress.png")
    except Exception as e:  # noqa: BLE001
        print(f"(progress plot skipped: {e}; JSON was still saved)")


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(
        action_space="joint_velocity",
        gripper_action_space="position",
        do_reset=False,
        launch_robot=args.launch_robot,
    )
    local_gripper = open_local_gripper_if_configured()
    if args.launch_robot:
        print("Created the droid env (workstation triggered Polymetis launch).")
    else:
        print("Created the droid env (using existing NUC robot server; no relaunch).")
        _ensure_nuc_robot_ready(env._robot)
    _reset_env(env, local_gripper)

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    # Reward-DAgger: Robometer client (over the network) + gate
    reward_url = args.reward_url or f"http://{args.reward_host}:{args.reward_port}/"
    scorer = AsyncRobometerClient(reward_url, timeout=args.reward_timeout)
    if not scorer.health():
        print(f"⚠️ Robometer server not reachable at {reward_url} -- the gate won't fire until it is.")
    else:
        print(f"✓ Robometer reachable at {reward_url}")
    gate = RewardGate(short_window=args.short_window, drop_threshold=args.drop_threshold,
                      long_window=args.long_window, plateau_threshold=args.plateau_threshold,
                      method=args.method, smoothing=args.smoothing, min_drop_magnitude=args.min_drop_magnitude)

    results = []  # Collect results as list of dicts, convert to DataFrame at end

    while True:
        instruction = input("Enter instruction: ")
        # instruction = "pick up the cube"
        # instruction = "pick up the cube and place it in the bowl"
        # instruction = "pour the content in the cup into the bowl"
        # instruction = "put the pen into the cup"
        # instruction = "Reach the bowl"
        # instruction  = "Reach the bowl"

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Reward-DAgger per-episode state
        scorer.reset(instruction)
        gate.reset()
        _, last_version = scorer.latest      # ignore any scores completing during warmup
        fire_steps = []
        progress_trace = []                  # (env_step, robometer_progress) per completed score
        gate_success = None                  # set by human_takeover in teleop mode
        intervened = False
        handoff_step = None                  # env step where control passed to the human (teleop)

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    _get_env_observation(env, local_gripper),
                    save_to_disk=t_step == 0,
                    local_gripper=local_gripper,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                # ---- Robometer: append the scored (exterior) frame; request a score on cadence ----
                _ext = curr_obs[f"{args.external_camera}_image"]
                scorer.append(np.asarray(
                    Image.fromarray(_ext).resize((args.frame_width, args.frame_height)), dtype=np.uint8))
                if t_step % args.score_every == 0:
                    scorer.request()

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8) or pred_action_chunk.shape == (15, 8)    ## 10 is for pi0, 15 is for pi05

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)
                _step_env(env, action, local_gripper)

                # ---- Gate: consume each completed score once; record the progress trace ----
                progress, version = scorer.latest
                if version != last_version:
                    last_version = version
                    progress_trace.append((t_step, float(progress)))
                    if t_step >= args.warmup and gate.update(progress):
                        fire_steps.append(t_step)
                        print(f"\n[gate] fired at step {t_step} (progress={progress:.3f}, "
                              f"trigger={gate.last_trigger}) [handoff_mode={args.handoff_mode}]")
                        if args.handoff_mode == "teleop":
                            handoff_step = t_step
                            gate_success = human_takeover(env, local_gripper, args, video=video,
                                                          scorer=scorer, progress_trace=progress_trace,
                                                          start_step=t_step + 1)
                            intervened = True
                            break
                        elif args.handoff_mode == "stop":
                            print("  no-Quest: ending episode here (a human WOULD take over).")
                            intervened = True
                            break
                        else:  # "shadow": log the fire and let pi0 keep driving; reset so it can fire again
                            gate.reset()

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        os.makedirs("results/vd/", exist_ok=True)
        save_filename = os.path.join("results", "video_" + timestamp)
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        if fire_steps:
            print(f"[gate] fired at steps: {fire_steps}")
        # In teleop mode the human's end button already labeled success; otherwise ask as usual.
        success: str | float | None = float(gate_success) if gate_success is not None else None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        # df = df.append(
        #     {
        #         "success": success,
        #         "duration": t_step,
        #         "video_filename": save_filename,
        #     },
        #     ignore_index=True,
        # )
        results.append({
            "success": success,
            "duration": t_step,
            "video_filename": save_filename,
            "intervened": intervened,
            "gate_fires": fire_steps,
        })
        save_progress_trace(
            progress_trace, fire_steps,
            dict(instruction=instruction, success=success, intervened=intervened, duration=int(t_step)),
            save_filename, handoff_step=handoff_step,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        for _ in range(2):
            _reset_env(env, local_gripper)
            time.sleep(1)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False, local_gripper=None):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]
        # import ipdb; ipdb.set_trace()
        # for k in key:
        #     if "left" in k:
        #         left_image = image_observations[k]
        #     elif "right" in k:
        #         right_image = image_observations[k]
        #     elif "wrist" in k:
        #         wrist_image = image_observations[k]


    # Drop the alpha dimension
    left_image = left_image[..., :3] if left_image is not None else right_image[..., :3]
    right_image = right_image[..., :3] if right_image is not None else left_image
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    if local_gripper is not None:
        gripper_position = np.array([local_gripper.get_position()])
    else:
        gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
