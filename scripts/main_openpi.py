# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import pickle
import signal
import sys
import threading
import time
import urllib.request
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

REPO_ROOT = Path(__file__).resolve().parents[1]  # robometer_policy_learning directory
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gripper_utils import open_local_gripper_if_configured, send_gripper_command_openpi  # noqa: E402
from robometer_policy_learning.utils.reward_gate import RewardGate  # noqa: E402

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "33790348"
    right_camera_id: str = "37998989"
    wrist_camera_id: str = "14064085"

    # Policy parameters
    external_camera: str | None = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 900
    # Task instruction. If unset, the loop asks for it before every episode
    prompt: str | None = None
    open_loop_horizon: int = 8  # actions to execute from action chunk before requerying

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # points to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = 8000  # points to the port of the policy server

    # When the NUC already runs scripts/server/run_server.py (with Polymetis up), leave this False.
    # True relaunches Polymetis from the workstation and can break an existing session on re-run.
    launch_robot: bool = False

    # Robometer scoring server (scripts/robometer_http_server.py)
    reward_host: str = "localhost"
    reward_port: int = 8900
    reward_url: str | None = None       # full URL; overrides host/port
    
    reward_timeout: float = 20.0
    score_every: int = 3                # env steps between Robometer calls (benchmarked: Robometer keeps up)
    frame_width: int = 640              # resize the scored frame to match the offline traces (640x360)
    frame_height: int = 360
 
    # Reward gate hyperparams, using the LIBERO fine-tuned config
    method: str = "spearman" 
    short_window: int = 25              # 75 env steps / score_every 3
    long_window: int = 75               # 225 env steps / score_every 3
    drop_threshold: float = -0.9
    plateau_threshold: float = 0.0
    min_drop_magnitude: float = 0.0
    smoothing: float = 0.0
    
    # What happens when the gate fires:
    #   "teleop" = hand to the Quest human; needs quest setup
    #   "stop"   = end the episode at the fire
    #   "shadow" = log the fire and let pi0 keep driving
    handoff_mode: str = "teleop"
    teleop_action_space: str = "cartesian_velocity"
    
    # Save each episode's transitions for DAgger training.
    save_episodes: bool = True
    episode_dir: str = "results/rdagger/round1"


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
    """Build env observation without querying a Polymetis gripper on the NUC."""
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
        "joint_velocities": [0.0] * len(joint_positions),  # not used
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


# We are using Ctrl+C to terminate rollouts early. However, if we press Ctrl+C while the policy server is
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


class AsyncRobometerClient:
    """ Sends subsampled frames to the server every `score_every` steps."""

    def __init__(self, url, max_frames=8, timeout=20.0):
        self.url = url; self.max_frames = int(max_frames); self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._frames, self._prompt, self._episode = [], "", 0
        self._progress, self._version = 0.0, 0
        self._latencies = []          # per-call wall clock, to check the VLM keeps up with score_every
        self._req = threading.Event(); self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()

    def reset(self, prompt):
        with self._lock:
            self._frames = []; self._prompt = str(prompt); self._episode += 1; self._progress = 0.0
            self._latencies = []

    def latency_report(self, score_every, control_hz) -> str:
        """Whether Robometer kept up with the requested `score_every`."""
        with self._lock:
            lat = list(self._latencies)
        if not lat:
            return "[robometer] no completed scores this episode"
        lat = np.asarray(lat)
        budget = score_every / float(control_hz)
        over = float((lat > budget).mean())
        return (f"[robometer] {len(lat)} scores | latency p50={np.percentile(lat, 50):.2f}s "
                f"p95={np.percentile(lat, 95):.2f}s | budget={budget:.2f}s "
                f"({over:.0%} over budget -> ticks spaced ~{max(np.median(lat), budget) * control_hz:.0f} "
                f"env steps, not {score_every})")

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
                _t0 = time.time()
                progress = self._post(self._subsample(snap), prompt)
                _lat = time.time() - _t0
            except Exception as e:
                print(f"[robometer] request failed: {e}")
                continue
            with self._lock:
                if eid != self._episode:
                    continue
                self._progress, self._version = progress, self._version + 1
                self._latencies.append(_lat)

    def stop(self):
        self._stop = True; self._req.set()


_VR_CONTROLLER = None


def _controller_tracking(controller) -> bool | None:
    """True/False if we can tell whether Quest 6DoF poses are streaming, None if we can't."""
    state = getattr(controller, "_state", None)
    if isinstance(state, dict) and "poses" in state:
        return bool(state["poses"])
    reader = getattr(controller, "oculus_reader", None)
    if reader is None:
        return None
    try:
        transforms, _ = reader.get_transformations_and_buttons()
        return bool(transforms)
    except Exception:  # noqa: BLE001
        return None


def _get_vr_controller(timeout: float = 0.0):
    """Build the Quest controller once and reuse it across takeovers."""
    global _VR_CONTROLLER
    if _VR_CONTROLLER is None:
        from droid.controllers.oculus_controller import VRPolicy

        print("[teleop] starting Quest controller ...")
        _VR_CONTROLLER = VRPolicy(right_controller=True)

    if timeout > 0:
        t0 = time.time()
        while _controller_tracking(_VR_CONTROLLER) is False and time.time() - t0 < timeout:
            time.sleep(0.1)
    tracking = _controller_tracking(_VR_CONTROLLER)
    if tracking is True:
        print("[teleop] controller tracking live")
    elif tracking is False:
        print("⚠️  [teleop] no controller poses yet -- Quest 6DoF tracking is not live, so the arm "
              "will NOT respond. Wake the headset and check the guardian/lighting.")
    else:
        print("[teleop] controller ready")
    return _VR_CONTROLLER


class EpisodeRecorder:
    """Collects the (observation, action, actor(policy/human)) triples a DAgger round needs to train on.

    The loop already saves video and a progress trace, but neither can be trained on. This stores
    what pi0 uses -- both camera views at 224x224, joint positions, gripper position -- next to
    the action that was actually executed, in pi0's own action space (7 joint velocities + gripper
    position). Teleop actions come out of the Quest as Cartesian velocity, so the caller converts
    them through the NUC's IK first.

    actor: 0 = pi0, 1 = human.
    """

    def __init__(self):
        self.ext, self.wrist = [], []
        self.joints, self.grip, self.actions, self.actor = [], [], [], []

    def add(self, obs, action, actor, external_camera="left"):
        self.ext.append(image_tools.resize_with_pad(obs[f"{external_camera}_image"], 224, 224))
        self.wrist.append(image_tools.resize_with_pad(obs["wrist_image"], 224, 224))
        self.joints.append(np.asarray(obs["joint_position"], dtype=np.float32))
        self.grip.append(np.asarray(obs["gripper_position"], dtype=np.float32).reshape(1))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.actor.append(int(actor))

    def __len__(self):
        return len(self.actions)

    def save(self, path, control_hz=DROID_CONTROL_FREQUENCY, **meta):
        if not self.actions:
            print("[record] no steps to save")
            return
        actions = np.stack(self.actions)
        actor = np.asarray(self.actor, dtype=np.int8)
        human = actor == 1
        if human.any():
            # Fill in the human's arm labels in pi0's action space: the joint velocity that
            # produced the next observed pose
            q = np.stack(self.joints)
            dq = np.zeros_like(q)
            if len(q) > 1:
                dq[:-1] = (q[1:] - q[:-1]) * float(control_hz)
                dq[-1] = dq[-2]
            actions[human, :7] = dq[human]
            # Remove leading no-op corrections
            h_idx = np.flatnonzero(human)
            moving = np.flatnonzero(np.abs(actions[h_idx, :7]).max(axis=1) > 1e-3)
            if len(moving) and moving[0] > 0:
                drop = h_idx[: moving[0]]
                keep = np.ones(len(actions), bool)
                keep[drop] = False
                print(f"[record] dropped {len(drop)} no-op frames after handoff")
                actions, actor, human = actions[keep], actor[keep], human[keep]
                self.ext = [f for f, k in zip(self.ext, keep) if k]
                self.wrist = [f for f, k in zip(self.wrist, keep) if k]
                self.joints = [f for f, k in zip(self.joints, keep) if k]
                self.grip = [f for f, k in zip(self.grip, keep) if k]
                self.actor = [a for a, k in zip(self.actor, keep) if k]
        np.savez_compressed(
            path,
            control_hz=float(control_hz),
            exterior_image=np.stack(self.ext).astype(np.uint8),
            wrist_image=np.stack(self.wrist).astype(np.uint8),
            joint_position=np.stack(self.joints),
            gripper_position=np.stack(self.grip),
            actions=actions,
            actor=actor,
            **{k: np.asarray(v) for k, v in meta.items()},
        )
        print(f"saved episode data  -> {path}  ({len(self)} steps, {int(np.sum(self.actor))} human)")


def human_takeover(env, local_gripper, args, video=None, scorer=None, progress_trace=None, start_step=0,
                   recorder=None) -> bool:
    """Operator finishes the episode by teleoperating with the Quest; returns success bool."""
    
    controller = _get_vr_controller()
    # DROID computes self.DoF (7 cartesian / 8 joint) ONCE in __init__, but step() reads
    # self.action_space dynamically -- so switching the space requires syncing DoF too, or
    # step()'s `assert len(action) == self.DoF` fails (VRPolicy emits a 7-dim cartesian action).
    prev_space, prev_dof = env.action_space, env.DoF
    env.action_space = args.teleop_action_space
    env.DoF = 7 if "cartesian" in env.action_space else 8
    print("\n🙋 HUMAN TAKEOVER — teleoperate to finish, then press Ctrl+C. You will be asked "
          "whether the task succeeded.")
    success = None
    tstep = int(start_step)
    _, last_version = scorer.latest if scorer is not None else (0.0, 0)
    _t0 = time.time()   # handoff instant; used to report how long until the operator had control
    # The first moments after a handoff are dead: the operator is still reaching for the grip, and
    # the reader may not be streaming poses yet. Commanding the arm with a near-zero action there
    # does nothing useful, and RECORDING it teaches the policy to freeze exactly when a correction
    # is needed. So hold until the human genuinely moves, then start.
    armed = False
    try:
        while success is None:
            start = time.time()
            obs = _get_env_observation(env, local_gripper)
            action, info = controller.forward(obs, include_info=True)
            _btns = {k: v for k, v in info.items() if isinstance(v, (bool, np.bool_)) and v}
            if _btns:
                print("[teleop] buttons:", _btns)
            # Record + score the exterior view through the takeover, so the graph shows the human
            # completing the task after the handoff (continuing the same causal history).
            eobs = _extract_observation(args, obs, local_gripper=local_gripper)
            ext = eobs[f"{args.external_camera}_image"]
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
            # Record the step BEFORE sending it. The action stored here is a placeholder: teleop
            # uses Cartesian velocity but pi0 uses joint velocity, and converting online meant
            # an extra NUC round-trip per step (create_action_dict), which hung the control loop.
            # EpisodeRecorder.save() derives the joint-velocity label offline instead, by finite-
            # differencing the joint positions we already log.
            if not armed:
                if float(np.abs(np.asarray(action)[:6]).max()) > 1e-3:
                    armed = True
                    print(f"[teleop] operator has control ({time.time() - _t0:.1f}s after handoff)")
                else:
                    elapsed = time.time() - start
                    if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                        time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)
                    continue

            if recorder is not None:
                # Gripper: store the command we send, not the measured position
                recorder.add(eobs,
                             np.concatenate([np.zeros(7, dtype=np.float32),
                                             [float(np.clip(action[-1], 0.0, 1.0))]]),
                             actor=1, external_camera=args.external_camera)
            if local_gripper is not None:
                send_gripper_command_openpi(None, action[-1], local_gripper=local_gripper, blocking=False)
                # update_pose converts Cartesian velocity to joint velocity on the NUC and calls
                # update_joints itself
                env._robot.update_pose(action[:6], velocity=True, blocking=False)
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
        ans = ""
        while ans not in ("y", "n"):
            try:
                ans = input("\n↩️  takeover ended. Did the task succeed? (y/n) ").strip().lower()
            except KeyboardInterrupt:
                ans = "n"
        success = ans == "y"
    finally:
        env.action_space, env.DoF = prev_space, prev_dof  # restore joint_velocity for pi0
    print(f"↩️  takeover ended (success={success})")
    return bool(success)


def save_progress_trace(progress_trace, fire_steps, meta, save_base, handoff_step=None):
    """Store one rollout's Robometer progress series: JSON (raw) + PNG (graph with gate fires)."""
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
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment using joint velocity action space and gripper position action space.
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

    if args.handoff_mode == "teleop":
        _get_vr_controller(timeout=10.0)

    results = []  # Collect results as list of dicts, convert to DataFrame at end

    while True:
        instruction = args.prompt or input("Enter instruction: ")
        if args.prompt:
            print(f"Task: {instruction}") 
            
        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Reward-DAgger per-episode state
        scorer.reset(instruction)
        gate.reset()
        _, last_version = scorer.latest      # skip a score still in flight from the last episode
        fire_steps = []
        progress_trace = []                  # (env_step, robometer_progress) per completed score
        gate_success = None                  # set by human_takeover in teleop mode
        intervened = False
        handoff_step = None                  # env step where control passed to the human (teleop)

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        recorder = EpisodeRecorder() if args.save_episodes else None
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
                if recorder is not None:
                    recorder.add(curr_obs, action, actor=0, external_camera=args.external_camera)
                _step_env(env, action, local_gripper)

                progress, version = scorer.latest
                if version != last_version:
                    last_version = version
                    progress_trace.append((t_step, float(progress)))
                    if gate.update(progress):
                        fire_steps.append(t_step)
                        print(f"\n[gate] fired at step {t_step} (progress={progress:.3f}, "
                              f"trigger={gate.last_trigger}) [handoff_mode={args.handoff_mode}]")
                        if args.handoff_mode == "teleop":
                            handoff_step = t_step
                            gate_success = human_takeover(env, local_gripper, args, video=video,
                                                          scorer=scorer, progress_trace=progress_trace,
                                                          start_step=t_step + 1, recorder=recorder)
                            intervened = True
                            break
                        elif args.handoff_mode == "stop":
                            print("  no-Quest: ending episode here (a human WOULD take over).")
                            intervened = True
                            break
                        else:  # "shadow": log the fire and let pi0 keep driving
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

        print(scorer.latency_report(args.score_every, DROID_CONTROL_FREQUENCY))
        if fire_steps:
            print(f"[gate] fired at steps: {fire_steps}")
        # In teleop mode the y/n prompt at the end of the takeover already labeled the episode;
        # otherwise ask here as usual.
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
        if recorder is not None:
            os.makedirs(args.episode_dir, exist_ok=True)
            recorder.save(
                os.path.join(args.episode_dir, f"episode_{timestamp}.npz"),
                prompt=instruction, success=success, intervened=intervened,
                handoff_step=-1 if handoff_step is None else handoff_step, gate_fires=fire_steps,
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
