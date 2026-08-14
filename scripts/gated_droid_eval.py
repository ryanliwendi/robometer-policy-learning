# ruff: noqa
"""Reward-gated pi0 rollouts on the REAL DROID rig, with human-teleop rescue.

A drop-in extension of the openpi DROID eval loop (examples/droid/main.py = your main_openpi.py).
pi0 drives via the policy server (port 8000) as usual; every few steps we ask a Robometer server
for a progress score, a gate watches that series, and when it fires the operator finishes the
episode by teleoperating with the Quest (handoff -> "takeover" until an end button is pressed).

RUNS ON THE WORKSTATION. It is self-contained -- copy just this one file to
~/Documents/droid/scripts/ and run it in the `robot` conda env. Robometer stays on the lab
server (see scripts/robometer_http_server.py); this loop reaches it over the network, exactly
like it already reaches the pi0 policy server.

Prereqs (unchanged from your normal eval):
  1. Lab server:  uv run python scripts/robometer_http_server.py --port 8900   (+ tunnel to here)
  2. Workstation: serve_policy.py ... --policy.config=pi0_droid   (policy server, port 8000)
  3. Control stack up (Desk FCI enabled, NUC launch_robot.py), gripper on the WORKSTATION.
  4. Quest connected (adb) for the handoff, cover the proximity sensor.

Run:
    conda activate robot
    python scripts/gated_droid_eval.py --external_camera=left \
      --left_camera_id=37998989 --right_camera_id=33790348 --wrist_camera_id=14064085 \
      --reward_host=localhost --reward_port=8900 \
      --prompt="Pick up the red ball and put it in the purple cup"

⚠️ Set --left_camera_id to the SAME scene-camera serial you scored offline (label_real_world.py
default was 37998989) so the online progress matches your offline traces.
⚠️ Two spots need on-rig verification, both clearly marked below: the Quest teleop controller API
(`human_takeover`) and the teleop action-space switch.
"""

import contextlib
import dataclasses
import datetime
import math
import pickle
import signal
import threading
import time
import urllib.request
from collections import deque

import numpy as np
import tqdm
import tyro
from droid.robot_env import RobotEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image

DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # --- Hardware (same as main_openpi.py) ---
    left_camera_id: str = "37998989"
    right_camera_id: str = "33790348"
    wrist_camera_id: str = "14064085"
    external_camera: str | None = None  # "left" or "right" -- which scene cam feeds the policy AND Robometer

    # --- Rollout ---
    prompt: str | None = None  # task instruction; if None you'll be prompted per episode
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # --- pi0 policy server (workstation, port 8000) ---
    remote_host: str = "127.0.1.1"
    remote_port: int = 8000

    # --- Robometer server (lab server, reached over the network) ---
    reward_host: str = "localhost"
    reward_port: int = 8900
    reward_url: str | None = None  # full URL; overrides host/port. Use for a Pinggy HTTP(S) tunnel,
    #   e.g. --reward_url=https://shoep-68-181-17-183.run.pinggy-free.link
    score_every: int = 5          # env steps between Robometer calls (VLM can't keep up with 15 Hz)
    reward_timeout: float = 15.0  # seconds per Robometer request
    frame_width: int = 640        # resize the scored frame to match the offline traces (decord 640x360)
    frame_height: int = 360

    # --- Gate (placeholders; plug in your sweep_gate_real_world.py values) ---
    method: str = "spearman"      # "spearman" | "pearson" | "naive"
    short_window: int = 10        # in SCORER TICKS (= score_every env steps)
    long_window: int = 40
    drop_threshold: float = -0.8
    plateau_threshold: float = 0.1
    min_drop_magnitude: float = 0.15
    smoothing: float = 0.5        # EMA weight on history in [0, 1); 0 = off
    warmup: int = 10              # env steps before the gate may fire

    # --- Handoff ---
    handoff_mode: str = "teleop"  # what happens when the gate fires:
    #   "teleop" = hand to the Quest human (needs the Quest set up);
    #   "stop"   = end the episode at the fire (no Quest) -- validates the full detect pipeline;
    #   "shadow" = log the fire and let pi0 keep driving (no Quest) -- see ALL fire points per episode.
    teleop_action_space: str = "cartesian_velocity"  # VRPolicy emits Cartesian vel; eval runs joint_velocity


# ---------------------------------------------------------------------------
# Reward gate (self-contained; numpy-only correlation, no scipy dependency)
# ---------------------------------------------------------------------------
def _pearson(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    return float((xm * ym).sum() / denom) if denom > 0 else float("nan")


def _rankdata(a):
    """Average ranks (matches scipy for ties)."""
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
    """Fires on a sharp progress drop (short window) or a long plateau. Mirrors
    robometer_policy_learning/utils/reward_gate.py. Call reset() after an intervention."""

    def __init__(self, short_window=10, drop_threshold=-0.8, long_window=40, plateau_threshold=0.1,
                 method="spearman", smoothing=0.0, min_drop_magnitude=0.0):
        self.short_window = short_window
        self.drop_threshold = drop_threshold
        self.long_window = long_window
        self.plateau_threshold = plateau_threshold
        self.method = method
        self.smoothing = smoothing
        self.min_drop_magnitude = min_drop_magnitude
        self.history = deque(maxlen=long_window)
        self._ema = None
        self.last_trigger = None
        assert long_window >= short_window
        assert 0 <= smoothing < 1

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
        self.history.clear()
        self._ema = None
        self.last_trigger = None


# ---------------------------------------------------------------------------
# Async Robometer client: scores off the control thread so the ~15 Hz loop never blocks
# ---------------------------------------------------------------------------
class AsyncRobometerClient:
    def __init__(self, url, max_frames=8, timeout=15.0):
        self.url = url
        self.max_frames = int(max_frames)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._frames, self._prompt, self._episode = [], "", 0
        self._progress, self._version = 0.0, 0
        self._req = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def reset(self, prompt):
        with self._lock:
            self._frames = []
            self._prompt = str(prompt)
            self._episode += 1
            self._progress = 0.0

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
                snap = list(self._frames)          # cheap ref copy under the lock
                prompt, eid = self._prompt, self._episode
            try:
                progress = self._post(self._subsample(snap), prompt)
            except Exception as e:
                print(f"[robometer] request failed: {e}")
                continue
            with self._lock:
                if eid != self._episode:           # episode changed while scoring -> stale
                    continue
                self._progress, self._version = progress, self._version + 1

    def stop(self):
        self._stop = True
        self._req.set()


# ---------------------------------------------------------------------------
# Human takeover (Quest teleop). ⚠️ The two lines marked VERIFY are rig-specific.
# ---------------------------------------------------------------------------
def human_takeover(env, args) -> bool:
    """Operator finishes the episode by teleoperating; returns the final success bool.

    The workstation owns the arm (Polymetis) + gripper the whole time, so this just swaps the
    action SOURCE from the policy server to the Quest -- no hardware is re-plugged.

    ⚠️ VERIFY ON-RIG (1): the VRPolicy import + forward() signature, and which controller-info
        keys carry the operator's SUCCESS / FAILURE button presses (DROID collection uses the
        Quest A/B buttons). Adjust the two marked lines to your droid install.
    ⚠️ VERIFY ON-RIG (2): the action-space switch. VRPolicy emits Cartesian velocity but eval
        runs joint_velocity; we switch env.action_space for the segment. Confirm your RobotEnv
        supports a live switch (if not, this is where a dedicated teleop env would go).
    """
    from droid.controllers.oculus_controller import VRPolicy  # ⚠️ VERIFY (1)

    controller = VRPolicy()
    prev_space = getattr(env, "action_space", None)
    switched = False
    if args.teleop_action_space and prev_space != args.teleop_action_space:
        try:
            env.action_space = args.teleop_action_space  # ⚠️ VERIFY (2)
            switched = True
        except Exception as e:
            print(f"⚠️ could not switch action_space for teleop: {e}")

    print("\n🙋 HUMAN TAKEOVER — teleoperate to finish. Press the Quest SUCCESS / FAILURE button "
          "(or Ctrl+C = failure) to end.")
    success = None
    try:
        while success is None:
            start = time.time()
            obs = env.get_observation()
            action, info = controller.forward(obs, include_info=True)  # ⚠️ VERIFY (1)
            env.step(np.asarray(action, dtype=np.float32))
            if info.get("success"):
                success = True
            elif info.get("failure"):
                success = False
            elapsed = time.time() - start
            if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)
    except KeyboardInterrupt:
        success = False
    finally:
        if switched:
            try:
                env.action_space = prev_space
            except Exception:
                pass
    print(f"↩️  takeover ended (success={success})")
    return bool(success)


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original)
        if interrupted:
            raise KeyboardInterrupt


def safe_extract_observation(args, obs_dict):
    """Like main_openpi.py's _extract_observation but tolerates a dropped camera (returns None
    for that image instead of crashing on `img[..., :3]`), so a ZED glitch doesn't hard-crash."""
    images = obs_dict["image"]
    picks = {"left": args.left_camera_id, "right": args.right_camera_id, "wrist": args.wrist_camera_id}
    out = {}
    for name, serial in picks.items():
        img = next((images[k] for k in images if serial in k and "left" in k), None)
        if img is not None:
            img = img[..., :3][..., ::-1]  # drop alpha, BGR->RGB
        out[f"{name}_image"] = img
    rs = obs_dict["robot_state"]
    out["joint_position"] = np.array(rs["joint_positions"])
    out["gripper_position"] = np.array([rs["gripper_position"]])
    return out


def cameras_ok(curr_obs, external_camera):
    for key in (f"{external_camera}_image", "wrist_image"):
        img = curr_obs.get(key)
        if img is None or (isinstance(img, np.ndarray) and img.size == 0):
            return False, key
    return True, None


def resize_frame(img, w, h):
    return np.asarray(Image.fromarray(img).resize((w, h)), dtype=np.uint8)


def main(args: Args):
    assert args.external_camera in ("left", "right"), "set --external_camera to 'left' or 'right'"

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    reward_url = args.reward_url or f"http://{args.reward_host}:{args.reward_port}/"
    scorer = AsyncRobometerClient(reward_url, timeout=args.reward_timeout)
    if not scorer.health():
        print(f"⚠️ Robometer server not reachable at {reward_url} -- start robometer_http_server.py "
              f"on the lab server and tunnel it here. Continuing; the gate simply won't fire.")
    gate = RewardGate(short_window=args.short_window, drop_threshold=args.drop_threshold,
                      long_window=args.long_window, plateau_threshold=args.plateau_threshold,
                      method=args.method, smoothing=args.smoothing, min_drop_magnitude=args.min_drop_magnitude)

    while True:
        instruction = args.prompt or input("Enter instruction: ")
        scorer.reset(instruction)
        gate.reset()

        # ---- Validate cameras BEFORE moving the robot (wrist ZED-M is the weak link) ----
        ok = False
        for _ in range(10):
            curr_obs = safe_extract_observation(args, env.get_observation())
            ok, bad = cameras_ok(curr_obs, args.external_camera)
            if ok:
                break
            print(f"⚠️ camera '{bad}' not streaming; retrying... (check `lsusb | grep -i zed` for all 6 interfaces)")
            time.sleep(0.5)
        if not ok:
            print("❌ cameras not ready — skipping this episode. Fix the ZED stream and retry.")
            if input("Try another episode? (y/n) ").lower() != "y":
                break
            continue

        scorer.append(resize_frame(curr_obs[f"{args.external_camera}_image"], args.frame_width, args.frame_height))
        _, last_version = scorer.latest

        actions_done, chunk = 0, None
        video, success, intervened, crashed = [], None, False, False
        fire_steps = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running gated rollout... Ctrl+C to stop early.")
        try:
            for t_step in bar:
                start = time.time()
                curr_obs = safe_extract_observation(args, env.get_observation())
                ok, bad = cameras_ok(curr_obs, args.external_camera)
                if not ok:
                    print(f"\n⚠️ camera '{bad}' dropped mid-rollout at step {t_step} — ending episode.")
                    break
                exterior = curr_obs[f"{args.external_camera}_image"]
                video.append(exterior)
                scorer.append(resize_frame(exterior, args.frame_width, args.frame_height))
                if t_step % args.score_every == 0:
                    scorer.request()

                # ---- pi0 (policy server), receding-horizon chunk ----
                if actions_done == 0 or actions_done >= args.open_loop_horizon:
                    actions_done = 0
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(exterior, 224, 224),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }
                    with prevent_keyboard_interrupt():
                        chunk = policy_client.infer(request_data)["actions"]  # (10, 8)
                action = chunk[actions_done]
                actions_done += 1
                action = np.concatenate([action[:-1], np.ones(1) if action[-1].item() > 0.5 else np.zeros(1)])
                action = np.clip(action, -1, 1)

                # ---- Step the robot (LostRemote / joint-limit crashes surface here) ----
                try:
                    env.step(action)
                except Exception as e:
                    crashed = True
                    print(f"\n❌ LostRemote / robot error at step {t_step}: {e}\n"
                          "   Recover: on the NUC `sudo pkill -9 run_server franka_panda_client`, relaunch,\n"
                          "   clear the fault in Desk, and hand-guide the arm back to a centered pose.")
                    break

                # ---- Gate: consume each completed score once, after warmup ----
                progress, version = scorer.latest
                if t_step >= args.warmup and version != last_version:
                    last_version = version
                    if gate.update(progress):
                        fire_steps.append(t_step)
                        print(f"\n[gate] fired at step {t_step} (progress={progress:.3f}, "
                              f"trigger={gate.last_trigger}) [handoff_mode={args.handoff_mode}]")
                        if args.handoff_mode == "teleop":
                            success = human_takeover(env, args)
                            intervened = True
                            break
                        elif args.handoff_mode == "stop":
                            print("  no-Quest: ending episode here (a human WOULD take over).")
                            intervened = True
                            break
                        else:  # "shadow": log the fire and let pi0 keep driving, so you see
                            # every fire point in one rollout. Reset so it can fire again later.
                            gate.reset()

                elapsed = time.time() - start
                if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)
        except KeyboardInterrupt:
            print("\n(stopped early)")

        # ---- Save video + record label ----
        if video:
            try:
                from moviepy.editor import ImageSequenceClip
                fn = "gated_" + datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S") + ".mp4"
                ImageSequenceClip(list(np.stack(video)), fps=10).write_videofile(fn, codec="libx264", logger=None)
                print(f"saved {fn}")
            except Exception as e:
                print(f"(video save failed: {e})")

        if fire_steps:
            print(f"gate fired at steps: {fire_steps}")
        if not intervened and not crashed and success is None:
            ans = input("Did the autonomous rollout succeed? (y/n) ").strip().lower()
            success = ans == "y"
        print(f"episode: success={success} intervened={intervened} crashed={crashed} fires={fire_steps}")

        if input("Do one more? (y/n) ").lower() != "y":
            break
        try:
            env.reset()
        except Exception as e:
            print(f"⚠️ env.reset failed ({e}); recover the robot before continuing.")

    scorer.stop()


if __name__ == "__main__":
    main(tyro.cli(Args))
