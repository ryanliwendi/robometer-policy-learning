#!/usr/bin/env python3
"""Human-in-the-loop rollout of a trained robomimic policy.

Roll out a trained policy in its robomimic environment and take over with the keyboard at
any time to provide corrections. Press the takeover key (default: Tab) to seize control; the
robot then follows YOUR input until you press it again, which hands control back to the policy
(which replans from the corrected state).

This is an *interactive* tool (no data is saved). It reuses the exact training env build
(``setup_robomimic_env``) and obs handling (``EvaluationWorker``) so the policy sees the same
observations it was trained on.

Loading: driven by the same Hydra config used for training; point ``training.load_dir`` at a
checkpoint directory that contains ``actor.pt`` (the deployable BaseActor, e.g. the EMA
DiffusionActor for DP).

Usage (local machine with a display):
    uv run python scripts/rollout_hitl.py --config-name robomimic_image_dp \
        training.load_dir=/path/to/checkpoints/50000

Optional overrides (Hydra ``+`` syntax adds the keys):
    +teleop.pos_sensitivity=2.0 +teleop.rot_sensitivity=2.0 \
    +teleop.render_size=640 +teleop.num_episodes=10 +teleop.camera=agentview

Keyboard controls (robosuite Keyboard device, captured globally via pynput):
    Tab         : toggle takeover (human control on/off)
    w/s a/d r/f : translate ee   |   z/x t/g c/v : rotate ee
    spacebar    : toggle gripper |   q : reset (abort) episode   |   ESC : quit
"""

import os

# Offscreen rendering for camera observations + the cv2 viewer (set before importing robosuite).
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import time

import cv2
import numpy as np
import torch
from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.samplers import RandomSampler
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker

# In human-control mode the sim is paused until the human issues a command; a command is a
# non-trivial end-effector motion or a gripper-state change.
_CMD_EPS = 1e-6


class TakeoverToggle:
    """Edge-triggered keyboard toggle on its own global ``pynput`` listener.

    Press the toggle key to switch control between policy and human; press again to switch
    back. Auto-repeat while the key is held is ignored (only the rising edge flips ``active``).
    Runs independently of robosuite's ``Keyboard`` device, so the toggle key never feeds the
    teleop controller (pick a key outside robosuite's w/a/s/d/r/f/z/x/t/g/c/v/space/q set).
    """

    def __init__(self, key: str = "tab"):
        from pynput import keyboard as pk

        named = {"tab": pk.Key.tab, "enter": pk.Key.enter, "space": pk.Key.space, "esc": pk.Key.esc}
        key = str(key).lower()
        # Either a special Key.* (named) or a single-character string matched against KeyCode.char.
        self._target = named.get(key, key)
        self.active = False
        self._down = False
        self._listener = pk.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _match(self, k) -> bool:
        if isinstance(self._target, str):
            return getattr(k, "char", None) == self._target
        return k == self._target

    def _on_press(self, k):
        if self._match(k) and not self._down:  # rising edge only (ignore auto-repeat)
            self._down = True
            self.active = not self.active

    def _on_release(self, k):
        if self._match(k):
            self._down = False

    def reset(self, active: bool = False):
        self.active = active
        self._down = False

    def stop(self):
        try:
            self._listener.stop()
        except Exception:
            pass


class RandomActor:
    """Minimal stand-in actor for debugging the HITL loop without a trained checkpoint.

    Mirrors ``BaseActor.act``: returns ``(action, actor_state)`` where ``action`` is a torch
    tensor of uniform-random actions in ``[-1, 1]``. Emits an ``(1, chunk_size, action_dim)``
    chunk so the receding-horizon replanning path is exercised too.
    """

    def __init__(self, action_dim: int, chunk_size: int = 1, device: str = "cpu"):
        self.action_dim = int(action_dim)
        self.chunk_size = max(1, int(chunk_size))
        self.device = device

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def act(self, obs, deterministic: bool = False, actor_state=None):
        action = torch.empty(1, self.chunk_size, self.action_dim, device=self.device).uniform_(-1.0, 1.0)
        return action, actor_state


def _find_robosuite_env(env):
    """Walk the gym/vector wrapper stack down to the underlying robosuite env (has ``.robots``)."""
    e = env
    for _ in range(32):
        if hasattr(e, "robots"):
            return e
        if hasattr(e, "env"):
            e = e.env
        elif hasattr(e, "envs"):
            e = e.envs[0]
        elif hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        else:
            break
    raise RuntimeError("Could not locate the underlying robosuite env (no `.robots` found).")


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load the trained actor (a deployable BaseActor pickled by Algorithm.save) ----
    # Debug mode (``+debug=true``): skip checkpoint loading and drive with a RandomActor so the
    # HITL loop / env / rendering can be exercised end-to-end without a trained policy.
    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    actor = None
    if debug:
        logger.warning("debug=true -> using RandomActor (no checkpoint loaded).")
    else:
        load_dir = OmegaConf.select(cfg, "training.load_dir", default=None)
        if not load_dir:
            raise ValueError("Set training.load_dir=<checkpoint dir containing actor.pt> (or +debug=true).")
        actor_path = os.path.join(load_dir, "actor.pt")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"actor.pt not found in {load_dir}. Pass a checkpoint dir (e.g. .../checkpoints/50000).")
        actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
        actor.eval()
        logger.info(f"Loaded actor {type(actor).__name__} from {actor_path}")

    # DINO-embedding policies (Mode A) need the embedding wrapper on the env; not wired here yet.
    if OmegaConf.select(cfg, "env.dino_image_keys", default=None):
        raise NotImplementedError(
            "HITL rollout doesn't support DINO-embedding (Mode A) policies yet. Use a state-only "
            "or featurizer-image policy (model.image_encoder.type set, env.dino_image_keys null)."
        )

    # ---- Build a single, non-chunked robomimic env (we manage chunking manually so we can take
    # over at every step). Same wrapper stack as training => observations align. ----
    env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path,
        n_envs=1,
        device=device,
        seed=OmegaConf.select(cfg, "training.seed", default=0),
        max_episode_steps=cfg.env.max_episode_steps,
        use_full_state=cfg.env.use_full_state,
        terminate_on_success=False,
        chunk_size=None,
        n_action_steps=1,
    )
    base_env = _find_robosuite_env(env)
    robot = base_env.robots[0]
    action_dim = int(env.single_action_space.shape[0])

    if debug:
        chunk_size = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)
        actor = RandomActor(action_dim=action_dim, chunk_size=chunk_size, device=device)
        actor.eval()

    # ---- Low-dim obs normalization stats (must match training). Cheap: low-dim cache only,
    # no image preload, no embeddings. ----
    lowdim_stats = {}
    if OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False):
        logger.info("Computing low-dim obs normalization stats from the dataset...")
        statbuf = H5ReplayBuffer(
            h5_paths=[cfg.env.h5_dataset_path], sampler=RandomSampler(), normalize_lowdim_obs=True
        )
        lowdim_stats = statbuf.lowdim_obs_stats
        logger.info(f"Normalizing low-dim keys: {list(lowdim_stats.keys())}")

    # Reuse EvaluationWorker's obs helpers (extract env 0 + to-device + z-score) so the policy
    # sees exactly what it sees during eval.
    obs_helper = EvaluationWorker(eval_env=env, device=device, record_video=False, lowdim_obs_stats=lowdim_stats)

    # ---- Keyboard device (robosuite 1.4: global pynput listener, no viewer wiring needed) ----
    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action

    keyboard = Keyboard(
        pos_sensitivity=OmegaConf.select(cfg, "teleop.pos_sensitivity", default=1.0),
        rot_sensitivity=OmegaConf.select(cfg, "teleop.rot_sensitivity", default=1.0),
    )

    # Takeover toggle: press to seize control, press again to release back to the policy.
    takeover_key = str(OmegaConf.select(cfg, "teleop.takeover_key", default="tab"))
    toggle = TakeoverToggle(takeover_key)
    _ROBO_KEYS = set("wasdrfzxtgcv") | {"space", "q"}
    if takeover_key.lower() in _ROBO_KEYS:
        logger.warning(
            f"takeover_key '{takeover_key}' is also a robosuite teleop control; it will be shadowed "
            f"while teleoping. Pick a free key, e.g. +teleop.takeover_key=tab."
        )

    n_exec = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)  # replan cadence for chunked policies
    num_episodes = int(OmegaConf.select(cfg, "teleop.num_episodes", default=10**9))
    render_size = int(OmegaConf.select(cfg, "teleop.render_size", default=512))
    camera = OmegaConf.select(cfg, "teleop.camera", default="agentview")
    wrist_camera = OmegaConf.select(cfg, "teleop.wrist_camera", default="robot0_eye_in_hand")
    show_wrist = bool(OmegaConf.select(cfg, "teleop.show_wrist", default=True))
    window = f"HITL rollout  ({takeover_key}: take/release control, move: wasd/rf, rotate: zx/tg/cv, grip: space, q: reset, ESC: quit)"

    def _render_cam(name):
        # robosuite renders upside-down and in RGB; flip vertically and convert to BGR for cv2.
        img = base_env.sim.render(height=render_size, width=render_size, camera_name=name)
        img = np.ascontiguousarray(img[::-1, :, ::-1])
        cv2.putText(img, name, (8, render_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # Render both cameras only if the wrist camera is actually available for this env.
    if show_wrist and wrist_camera != camera:
        try:
            base_env.sim.render(height=8, width=8, camera_name=wrist_camera)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Wrist camera '{wrist_camera}' unavailable ({e}); rendering '{camera}' only.")
            show_wrist = False
    else:
        show_wrist = False

    def render(step, mode, success):
        panels = [_render_cam(camera)]
        if show_wrist:
            panels.append(_render_cam(wrist_camera))
        frame = np.hstack(panels)
        color = (0, 200, 0) if mode == "POLICY" else (0, 0, 255)
        label = f"ep {ep}  step {step}  [{mode}]" + ("  SUCCESS" if success else "")
        cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.imshow(window, frame)
        return cv2.waitKey(1) & 0xFF

    logger.info(
        f"action_dim={action_dim}, replan every n_action_steps={n_exec}. Press '{takeover_key}' to "
        f"take/release control. Keys are captured globally (keep this display focused)."
    )

    ep = 0
    try:
        while ep < num_episodes:
            obs_b, _ = env.reset()
            obs = obs_helper._extract_env_data(obs_b, 0)
            keyboard.start_control()
            toggle.reset(active=False)  # every episode starts under policy control

            policy_chunk, chunk_pos, prev_human, last_grasp = None, 0, False, False
            steps, human_steps, success, done = 0, 0, False, False

            while not done:
                # 'q' aborts the episode in either mode (robosuite device sets this on release).
                if keyboard._reset_state:
                    logger.info("Reset (q): aborting this episode.")
                    break

                human_in_control = toggle.active
                if human_in_control and not prev_human:
                    keyboard.start_control()  # entering teleop: clear any stale accumulated deltas
                    last_grasp = keyboard.grasp  # sync gripper state (False after reset)
                elif prev_human and not human_in_control:
                    policy_chunk, chunk_pos = None, 0  # released: policy replans from the corrected state
                prev_human = human_in_control

                # --- Choose action: human (toggle on) vs policy (manual receding-horizon chunking) ---
                if human_in_control:
                    # Pause the sim and wait for a deliberate human command; advance one step per command.
                    action = None
                    while toggle.active and not keyboard._reset_state:
                        human_action, _grasp = input2action(
                            device=keyboard, robot=robot, active_arm="right", env_configuration=None
                        )
                        if human_action is None:  # 'q' pressed during teleop
                            break
                        human_action = np.asarray(human_action, dtype=np.float32)
                        if human_action.shape[0] != action_dim:  # pad/trim defensively to the env's action dim
                            fixed = np.zeros(action_dim, dtype=np.float32)
                            n = min(action_dim, human_action.shape[0])
                            fixed[:n] = human_action[:n]
                            human_action = fixed
                        # A command = non-trivial ee motion or a gripper-state change.
                        commanded = bool(
                            np.linalg.norm(human_action[:-1]) > _CMD_EPS or keyboard.grasp != last_grasp
                        )
                        last_grasp = keyboard.grasp
                        if commanded:
                            action = human_action
                            break
                        # Idle: keep the window live and wait, without advancing the sim.
                        if render(steps, "HUMAN  (waiting for input)", success) == 27:  # ESC
                            raise KeyboardInterrupt
                        time.sleep(0.01)

                    if action is None:
                        if keyboard._reset_state:
                            logger.info("Reset (q): aborting this episode.")
                            break
                        continue  # control released without a command -> let the policy drive next iteration
                    mode = "HUMAN"
                    human_steps += 1
                else:
                    if policy_chunk is None or chunk_pos >= len(policy_chunk) or chunk_pos >= n_exec:
                        obs_t = obs_helper._prepare_obs(obs)
                        with torch.inference_mode():
                            pred, _ = actor.act(obs_t, deterministic=True)
                        pred = pred.detach().cpu().numpy()
                        # (1, H, A) chunk -> (H, A); (1, A) -> (1, A)
                        policy_chunk = pred.reshape(-1, action_dim) if pred.ndim == 3 else np.atleast_2d(pred)
                        chunk_pos = 0
                    action = policy_chunk[chunk_pos]
                    chunk_pos += 1
                    mode = "POLICY"

                # --- Step the (single) vectorized env ---
                obs_b, _, term, trunc, info = env.step(action.reshape(1, action_dim).astype(np.float32))
                obs = obs_helper._extract_env_data(obs_b, 0)
                done = bool(np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0])
                info0 = obs_helper._extract_info(info, 0)
                if isinstance(info0, dict) and (info0.get("is_success") or info0.get("success")):
                    success = True
                steps += 1

                if render(steps, mode, success) == 27:  # ESC
                    raise KeyboardInterrupt

            logger.success(f"Episode {ep}: steps={steps}, human_steps={human_steps}, success={success}")
            ep += 1
    except KeyboardInterrupt:
        logger.info("Quit requested.")
    finally:
        toggle.stop()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
