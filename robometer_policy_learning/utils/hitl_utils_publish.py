"""Human-in-the-loop (HITL) utilities for HG-DAgger collection/training (public release).

Contains the keyboard/SpaceMouse teleop toggle, env helpers, the rollout/collection worker, and the
dataset-level reweighting schemes (SIRIUS and IWR) for weighted behavioral cloning. The human is a
real teleoperator; there is no simulated-human / intervention-criterion machinery.
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from loguru import logger

from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

# Action-mode class names (SIRIUS).
DEMO, INTERVENTION, PRE_INTERVENTION, ROLLOUT = "demo", "intervention", "pre_intervention", "rollout"

# Intervention-label convention (matches the HITL buffers):
# 0 = autonomous policy/rollout, 1 = human intervention, 2 = offline demo.
ROLLOUT_LABEL, INTERVENTION_LABEL, DEMO_LABEL = 0, 1, 2


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


def describe_control_mode(env) -> str:
    """Human-readable description of the env controller's action mode (DELTA vs ABSOLUTE).

    Reads the underlying robosuite controller's ``use_delta`` flag: True => delta end-effector
    commands, False => absolute-pose targets. Used by the HITL scripts to log which control mode
    the rollouts/dataset are in.
    """
    try:
        base_env = _find_robosuite_env(env)
        controller = base_env.robots[0].controller
        name = getattr(controller, "name", type(controller).__name__)
        use_delta = bool(getattr(controller, "use_delta", True))
        return f"{name} ({'DELTA' if use_delta else 'ABSOLUTE'} control, use_delta={use_delta})"
    except Exception as e:  # noqa: BLE001
        return f"unknown (could not inspect controller: {e})"


def _extract0(batched):
    """Extract env 0 from a vectorized obs dict / array (n_envs=1)."""
    if isinstance(batched, dict):
        return {k: v[0] for k, v in batched.items()}
    return batched[0]


def _scalar(x):
    return np.asarray(x).reshape(-1)[0]


def _success_from_info(info) -> bool:
    """Best-effort success flag from a (possibly vectorized) env info."""
    if isinstance(info, dict):
        for key in ("is_success", "success"):
            if key in info:
                try:
                    return bool(np.asarray(info[key]).reshape(-1)[0])
                except Exception:  # noqa: BLE001
                    return bool(info[key])
    if isinstance(info, (list, tuple)) and info:
        d = info[0] or {}
        return bool(d.get("is_success") or d.get("success"))
    return False


class HitlRolloutWorker:
    """Runs HG-DAgger HITL rollouts on a single robomimic env with keyboard/SpaceMouse teleop.

    During collection (``allow_human=True``) the human toggles control with the takeover key and
    drives the robot with the teleop device; otherwise the rollout policy acts.

    Manages receding-horizon chunk execution, normalizes/filters observations for both ``act()``
    and storage, optionally renders the agent view(s) to a cv2 window, and (when ``store``) buffers
    a whole episode and adds it to ``online_buffer`` at episode end with per-step intervention
    labels (0=policy, 1=human) plus episode-level stats (``episode_len``,
    ``episode_num_interventions``). Keyboard 'q' aborts (discards) an episode; ESC raises
    KeyboardInterrupt.
    """

    def __init__(
        self,
        env,
        actor,
        online_buffer,
        device,
        action_dim: int,
        *,
        lowdim_stats: dict = None,
        remove_obs_keys=None,
        n_action_steps: int = 1,
        store_only_human: bool = False,
        enable_render: bool = True,
        teleop_device: str = "keyboard",
        takeover_key: str = "tab",
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        render_size: int = 512,
        camera: str = "agentview",
        wrist_camera: str = "robot0_eye_in_hand",
        show_wrist: bool = True,
        cmd_eps: float = 1e-6,
        record_video: bool = False,
        video_dir: str = None,
        video_fps: int = 20,
    ):
        self.env = env
        self.actor = actor
        self.online_buffer = online_buffer
        self.device = device
        self.action_dim = int(action_dim)
        self.lowdim_stats = lowdim_stats or {}
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.n_action_steps = int(n_action_steps)
        self.store_only_human = bool(store_only_human)
        self.cmd_eps = float(cmd_eps)
        self.enable_render = bool(enable_render)

        # Debug video recording: when on, each episode's executed-step frames (with the
        # POLICY/HUMAN overlay) are buffered and written to an mp4 under ``video_dir``.
        self.record_video = bool(record_video)
        self.video_dir = video_dir
        self.video_fps = int(video_fps)
        self._record_frames = None  # per-episode frame buffer (None when not recording)
        if self.record_video and self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

        self.base_env = _find_robosuite_env(env)
        self.robot = self.base_env.robots[0]
        self.controller = self.robot.controller

        # robosuite's input2action / Keyboard always emit *delta* end-effector commands. If the
        # controller is in absolute-pose mode (control_delta=False) those deltas must be integrated
        # onto the current eef pose and re-emitted as an absolute target (see _delta_to_absolute_action),
        # otherwise the controller reads each tiny delta as an absolute Cartesian goal.
        self.absolute_pose = not bool(getattr(self.controller, "use_delta", True))
        if self.absolute_pose:
            cname = getattr(self.controller, "name", "")
            if cname not in ("OSC_POSE", "OSC_POSITION") or getattr(self.controller, "impedance_mode", "fixed") != "fixed":
                raise NotImplementedError(
                    f"Absolute-pose keyboard teleop only supports fixed-impedance OSC_POSE/OSC_POSITION "
                    f"controllers, got name={cname!r} impedance_mode={getattr(self.controller, 'impedance_mode', None)!r}."
                )
            logger.info(
                f"Controller {cname} is in absolute-pose mode (control_delta=False); keyboard deltas "
                "will be integrated onto the current eef pose and emitted as absolute targets."
            )

        # ---- Rendering setup. The teleop view is rendered at full ``render_size`` directly from
        # the sim (sim.render) so the window is sharp rather than an upscaled low-res obs image. ----
        self.teleop_device = str(teleop_device).lower()
        if self.teleop_device in ("space_mouse", "3dmouse"):
            self.teleop_device = "spacemouse"
        self.render_size = int(render_size)
        self.camera = camera
        self.wrist_camera = wrist_camera
        self.show_wrist = bool(show_wrist) and wrist_camera != camera
        if self.teleop_device == "spacemouse":
            controls = "move/twist puck: move, left btn: grip, right btn: reset"
        else:
            controls = "wasd/rf+zx/tg/cv: move, space: grip, q: reset"
        self.window = f"HITL ({takeover_key}: take/release, {controls}, ESC: quit)"
        # NOTE: cv2 must be imported before torch/TF (the caller does this), otherwise cv2's HighGUI
        # imshow/waitKey deadlocks against the pynput keyboard listener.

        # Teleop device (keyboard or 3D SpaceMouse). Both robosuite devices expose the same
        # start_control / get_controller_state / _reset_state interface and both are handled by
        # input2action; the takeover toggle stays on the keyboard (the SpaceMouse has no spare
        # button for it). The SpaceMouse reports gripper state via control_gripper, not .grasp.
        from robosuite.utils.input_utils import input2action

        if self.teleop_device == "keyboard":
            from robosuite.devices import Keyboard

            self._device = Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        elif self.teleop_device == "spacemouse":
            from robosuite.devices import SpaceMouse

            self._device = SpaceMouse(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        else:
            raise ValueError(f"teleop_device must be 'keyboard' or 'spacemouse', got {teleop_device!r}.")
        self._input2action = input2action
        self.toggle = TakeoverToggle(takeover_key)
        logger.info(f"HITL teleop device: {self.teleop_device} (takeover key: '{takeover_key}')")

    def close(self):
        if self.toggle is not None:
            self.toggle.stop()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass

    def _device_grasp(self):
        """Current gripper state of the teleop device, normalized across devices.

        The Keyboard exposes a toggled ``grasp`` bool; the SpaceMouse exposes ``control_gripper``
        (1.0 while the left button is held, else 0). Both are non-consuming reads (they do not
        advance the device's delta-position tracking the way get_controller_state() does).
        """
        dev = self._device
        g = getattr(dev, "grasp", None)
        if g is None:
            g = getattr(dev, "control_gripper", 0)
        return float(g)

    def _policy_action(self, obs_t, st):
        """Next single (env-space) rollout-policy action using receding-horizon chunking.

        ``st`` is a mutable ``{"chunk", "pos"}`` dict carried across steps; set ``st["chunk"]=None``
        to force a replan (e.g. after an intervention).
        """
        if st["chunk"] is None or st["pos"] >= len(st["chunk"]) or st["pos"] >= self.n_action_steps:
            with torch.inference_mode():
                pred, _ = self.actor.act(obs_t, deterministic=True)
            pred = pred.detach().cpu().numpy()
            st["chunk"] = pred.reshape(-1, self.action_dim) if pred.ndim == 3 else np.atleast_2d(pred)
            st["pos"] = 0
        a = st["chunk"][st["pos"]]
        st["pos"] += 1
        return a

    def _prep_obs(self, obs):
        """Normalize low-dim keys and drop unused keys -> dict used for BOTH act() and storage."""
        out = {}
        for k, v in obs.items():
            if k in self.remove_obs_keys:
                continue
            if k in self.lowdim_stats:
                st = self.lowdim_stats[k]
                v = ((np.asarray(v, dtype=np.float32) - st["mean"]) / st["std"]).astype(np.float32)
            out[k] = v
        return out

    def _fit_action(self, a):
        a = np.asarray(a, dtype=np.float32)
        if a.shape[0] == self.action_dim:
            return a
        fixed = np.zeros(self.action_dim, dtype=np.float32)
        n = min(self.action_dim, a.shape[0])
        fixed[:n] = a[:n]
        return fixed

    def _delta_to_absolute_action(self, ha):
        """Convert a delta-mode teleop action into the equivalent absolute-pose action.

        ``input2action`` returns ``[dpos, drotation_axisangle, grasp]`` meant for a delta OSC
        controller. We reproduce the goal that controller would reach -- the current eef pose
        composed with the *scaled* delta (``OSC.scale_action`` clips to the input range and maps
        it to metric output units) -- then re-express it as an absolute target pose so teleop
        feels identical under ``control_delta=False``::

            goal_pos = ee_pos + scaled_delta[:3]
            goal_ori = quat2mat(axisangle2quat(scaled_delta[3:6])) @ ee_ori_mat   (use_ori only)

        (mirrors ``set_goal_position`` / ``set_goal_orientation`` in robosuite's control_utils).
        """
        import robosuite.utils.transform_utils as T

        c = self.controller
        c.update(force=True)  # refresh ee_pos / ee_ori_mat from the current sim state
        ha = np.asarray(ha, dtype=np.float64)
        grasp = ha[-1]
        cdim = 6 if c.use_ori else 3
        scaled = c.scale_action(ha[:cdim])  # clip to input range + scale to metric output range
        goal_pos = c.ee_pos + scaled[:3]
        if c.use_ori:
            rot_err = T.quat2mat(T.axisangle2quat(scaled[3:6]))
            goal_ori = rot_err @ c.ee_ori_mat
            abs_action = np.concatenate([goal_pos, T.quat2axisangle(T.mat2quat(goal_ori)), [grasp]])
        else:
            abs_action = np.concatenate([goal_pos, [grasp]])
        return abs_action.astype(np.float32)

    def _render(self, phase, ep_tag, step, mode, success, record=True):
        """Render the agent view(s) with a who's-in-control banner.

        When debug recording is active the (executed-step) frame is appended to the episode
        buffer; pass ``record=False`` for transient frames (e.g. the teleop "waiting" loop) so
        only real steps land in the video. Builds the frame even when ``enable_render`` is False
        (headless) as long as recording is on.
        """
        want_record = record and self._record_frames is not None
        if not self.enable_render and not want_record:
            return -1
        import cv2

        def _cam(name):
            # Full-resolution offscreen render (robosuite renders upside-down RGB; flip + BGR for cv2).
            img = self.base_env.sim.render(height=self.render_size, width=self.render_size, camera_name=name)
            img = np.ascontiguousarray(img[::-1, :, ::-1])
            cv2.putText(img, name, (8, self.render_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return img

        panels = [_cam(self.camera)]
        if self.show_wrist:
            try:
                panels.append(_cam(self.wrist_camera))
            except Exception:  # noqa: BLE001  (wrist camera unavailable for this env)
                self.show_wrist = False
        frame = np.hstack(panels)
        # Banner color/label encode who is driving: red = HUMAN, green = POLICY.
        is_human = mode.startswith("HUMAN")
        color = (0, 0, 255) if is_human else (0, 180, 0)  # BGR
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), color, -1)
        cv2.putText(
            frame,
            f"{phase} {ep_tag}  step {step}  [{'HUMAN' if is_human else 'POLICY'} in control]"
            + ("  SUCCESS" if success else ""),
            (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        if want_record:
            self._record_frames.append(frame.copy())
        if not self.enable_render:
            return -1
        cv2.imshow(self.window, frame)
        return cv2.waitKey(1) & 0xFF

    def _write_video(self, episode_id, phase, aborted=False):
        """Flush the current episode's buffered frames to an mp4 (no-op if not recording)."""
        frames, self._record_frames = self._record_frames, None
        if not frames or not self.video_dir:
            return None
        import cv2

        h, w = frames[0].shape[:2]
        safe = str(episode_id).replace("/", "_")
        path = os.path.join(self.video_dir, f"{phase}_{safe}{'_aborted' if aborted else ''}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(self.video_fps), (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info(f"Saved HITL debug video ({len(frames)} frames) to {path}")
        return path

    def rollout_episode(self, episode_id, phase="COLLECT", allow_human=True, store=True,
                        require_success=False, require_intervention=False):
        """Run one episode. Returns (steps, human_steps, stored, success).

        Transitions are accumulated and added to the buffer ALL AT ONCE at episode end (not per
        step), so episode-level quantities are known before storing. Episodes aborted with 'q'
        are discarded.

        Storage policy:
          * store_only_human=False -> store every step under the contiguous episode index;
          * store_only_human=True  -> store only human-correction steps, each contiguous human
            segment under its own episode id (so chunked sampling forms valid chunks within a
            segment and never spans a policy gap).
          * require_success=True   -> the episode is stored ONLY if it ended in success; failed
            episodes are run (and rendered/recorded) but discarded (``stored`` stays 0).
          * require_intervention=True -> the episode is stored ONLY if it contains at least one human
            intervention step (``intervention``==1); rollouts with no correction are discarded.
            (Combined with require_success: keep only successful rollouts that had >= 1 intervention.)
        """
        was_training = self.actor.training
        self.actor.eval()
        env = self.env
        dev, toggle = self._device, self.toggle
        use_teleop = allow_human  # the human is always a real teleoperator

        obs = _extract0(env.reset()[0])
        if use_teleop:
            dev.start_control()
            toggle.reset(active=False)
        self._record_frames = [] if self.record_video else None  # debug video buffer for this episode
        chunk_st = {"chunk": None, "pos": 0}  # receding-horizon rollout chunk state
        prev_human, last_grasp = False, False
        steps, human_steps, stored, success, done = 0, 0, 0, False, False
        human_seg, seg_step = 0, 0  # contiguous human-segment index + step within it
        pending, aborted = [], False  # transitions buffered until episode end

        while not done:
            cur = self._prep_obs(obs)  # normalized + filtered (for act + storage)
            obs_t = move_to_device(convert_to_tensor(cur), self.device)

            if use_teleop and toggle.active:
                if not prev_human:
                    dev.start_control()
                    last_grasp = self._device_grasp()
                    human_seg += 1  # new contiguous human segment
                    seg_step = 0
                prev_human = True
                action = None  # wait for a deliberate human command (pause the sim until then)
                while toggle.active and not dev._reset_state:
                    ha, _ = self._input2action(device=dev, robot=self.robot, active_arm="right", env_configuration=None)
                    if ha is None:
                        break
                    ha = self._fit_action(ha)
                    # Gate on the *delta* magnitude (a deliberate move/grasp); only then convert to
                    # an absolute target if the controller is in absolute-pose mode.
                    grasp = self._device_grasp()
                    if np.linalg.norm(ha[:-1]) > self.cmd_eps or grasp != last_grasp:
                        cmd = self._delta_to_absolute_action(ha) if self.absolute_pose else ha
                        action, last_grasp = self._fit_action(cmd), grasp
                        break
                    last_grasp = grasp
                    if self._render(phase, episode_id, steps, "HUMAN  (waiting)", success, record=False) == 27:
                        raise KeyboardInterrupt
                    time.sleep(0.01)
                if dev._reset_state:
                    aborted = True  # 'q' -> discard this episode's buffered transitions
                    break
                if action is None:  # released without a command -> let the policy act this step
                    prev_human = False
                    continue
                mode, label = "HUMAN", INTERVENTION_LABEL
                human_steps += 1
                chunk_st["chunk"] = None  # replan after intervention

            else:
                if prev_human:  # just released -> replan from the corrected state
                    prev_human, chunk_st["chunk"] = False, None
                action = self._policy_action(obs_t, chunk_st)
                mode, label = "POLICY", ROLLOUT_LABEL

            next_b, rew, term, trunc, info = env.step(action.reshape(1, self.action_dim).astype(np.float32))
            next_obs = _extract0(next_b)
            terminated, truncated = bool(_scalar(term)), bool(_scalar(trunc))
            done = terminated or truncated
            if _success_from_info(info):
                success = True

            if store and (not self.store_only_human or label == INTERVENTION_LABEL):
                if self.store_only_human:
                    ep_store, step_store = f"{episode_id}_h{human_seg}", seg_step
                    seg_step += 1
                else:
                    ep_store, step_store = episode_id, steps
                pending.append(
                    dict(
                        obs=cur,
                        action=np.asarray(action, dtype=np.float32),
                        reward=float(_scalar(rew)),
                        next_obs=self._prep_obs(next_obs),
                        done=float(terminated),
                        truncated=float(truncated),
                        episode_id=ep_store,
                        step_in_episode=step_store,
                        info={"intervention": label},
                    )
                )

            obs = next_obs
            steps += 1
            if self._render(phase, episode_id, steps, mode, success) == 27:
                raise KeyboardInterrupt

        # Flush the whole episode at once (unless aborted), stamping episode-level stats onto
        # every transition's info so downstream sample-weighting can use them. When require_success
        # is set only successful episodes are stored; when require_intervention is set only episodes
        # with >= 1 human-correction step are stored (failed/skipped episodes leave ``stored`` at 0).
        kept = False
        if store and not aborted and pending:
            n_interventions = sum(1 for t in pending if t["info"]["intervention"] == INTERVENTION_LABEL)
            if (success or not require_success) and (n_interventions > 0 or not require_intervention):
                for t in pending:
                    t["info"]["episode_len"] = len(pending)
                    t["info"]["episode_num_interventions"] = n_interventions
                    self.online_buffer.add(**t)
                stored = len(pending)
                kept = True

        # Record a debug video only for KEPT rollouts (those added to the buffer); discard the
        # buffered frames of failed / filtered-out / aborted episodes.
        if self.record_video:
            if kept:
                self._write_video(episode_id, phase)
            else:
                self._record_frames = None

        if was_training:
            self.actor.train()

        return steps, human_steps, stored, success


def label_action_modes(online_buffer, pre_intervention_window: int):
    """Classify every online transition into an action mode.

    Returns ``(labeled, counts)`` where ``labeled`` is a list of ``(transition, mode)`` and
    ``counts`` maps each mode to its online sample count. A rollout step (label 0) is
    ``pre_intervention`` if it lies within ``pre_intervention_window`` steps immediately before
    the onset of a contiguous human-intervention segment in the same episode.
    """
    episodes = defaultdict(list)
    for t in online_buffer.get_all_transitions():
        if t is not None:
            episodes[t.episode_id].append(t)

    labeled = []
    counts = {DEMO: 0, INTERVENTION: 0, PRE_INTERVENTION: 0, ROLLOUT: 0}
    for trs in episodes.values():
        trs = sorted(trs, key=lambda x: x.step_in_episode)
        labels = [int(round(float((x.info or {}).get("intervention", ROLLOUT_LABEL)))) for x in trs]

        is_pre = [False] * len(trs)
        for i, lab in enumerate(labels):
            # Onset of a contiguous intervention segment: mark the preceding rollout window.
            if lab == INTERVENTION_LABEL and (i == 0 or labels[i - 1] != INTERVENTION_LABEL):
                for j in range(max(0, i - pre_intervention_window), i):
                    if labels[j] == ROLLOUT_LABEL:
                        is_pre[j] = True

        for i, (t, lab) in enumerate(zip(trs, labels)):
            if lab == DEMO_LABEL:
                mode = DEMO
            elif lab == INTERVENTION_LABEL:
                mode = INTERVENTION
            elif is_pre[i]:
                mode = PRE_INTERVENTION
            else:
                mode = ROLLOUT
            labeled.append((t, mode))
            counts[mode] += 1

    return labeled, counts


def compute_sirius_weights(
    online_buffer,
    offline_buffer=None,
    pre_intervention_window: int = 15,
    target_intv: float = 0.5,
    target_pre_intv: float = 0.002,
) -> Dict[str, Any]:
    """Assign SIRIUS class weights to samples IN PLACE and return diagnostics.

    Args:
        online_buffer: ReplayBuffer of collected rollouts (intervention labels 0/1, optionally 2).
        offline_buffer: optional H5ReplayBuffer of demos (all ``demo`` class, uniform weight 1.0).
        pre_intervention_window: window length W of rollout steps before an intervention onset.
        target_intv / target_pre_intv: target probability masses for those two classes.

    Returns:
        dict with ``counts``, ``ratios``, and per-class ``weights`` (for logging).
    """
    labeled, counts = label_action_modes(online_buffer, pre_intervention_window)
    counts[DEMO] += len(offline_buffer) if offline_buffer is not None else 0

    total = sum(counts.values())
    if total == 0:
        logger.warning("SIRIUS reweighting: dataset is empty; skipping.")
        return {"counts": counts, "ratios": {}, "weights": {}}

    ratios = {k: counts[k] / total for k in counts}

    w_demo = 1.0
    w_intv = target_intv / ratios[INTERVENTION] if ratios[INTERVENTION] > 0 else 0.0
    w_pre = target_pre_intv / ratios[PRE_INTERVENTION] if ratios[PRE_INTERVENTION] > 0 else 0.0
    rollout_mass = 1.0 - target_intv - ratios[DEMO] - target_pre_intv
    if rollout_mass < 0.0:
        logger.warning(
            f"SIRIUS: residual rollout mass is negative ({rollout_mass:.3f}); demos dominate "
            f"(ratio_demo={ratios[DEMO]:.3f}). Clamping rollout weight to 0."
        )
        rollout_mass = 0.0
    w_rollout = rollout_mass / ratios[ROLLOUT] if ratios[ROLLOUT] > 0 else 0.0

    weights = {DEMO: w_demo, INTERVENTION: w_intv, PRE_INTERVENTION: w_pre, ROLLOUT: w_rollout}

    # Assign weights: online per-sample (in place), offline uniform via set_weights.
    for t, mode in labeled:
        t.weight = float(weights[mode])
    if offline_buffer is not None:
        offline_buffer.set_weights(w_demo)

    logger.info(
        "SIRIUS reweighting | counts={counts} ratios={ratios} weights={weights}".format(
            counts={k: counts[k] for k in counts},
            ratios={k: round(ratios[k], 4) for k in ratios},
            weights={k: round(weights[k], 4) for k in weights},
        )
    )
    return {"counts": counts, "ratios": ratios, "weights": weights}


def compute_iwr_weights(online_buffer) -> Dict[str, Any]:
    """IWR (Intervention Weighted Regression) reweighting: balance corrections vs autonomy.

    Reference: Mandlekar et al., "Human-in-the-Loop Imitation Learning using Remote Teleoperation".

    Each sample is labelled by who was in control:
      * ``intervention`` (D_I): human corrections (intervention label 1),
      * ``robot``        (D_R): the policy acting autonomously (everything else).
    IWR makes the two groups contribute equally during BC even when corrections are rare::

        w_intervention = 0.5 / (num_i / (num_i + num_r))
        w_robot        = 0.5 / (num_r / (num_i + num_r))

    Operates on online HITL data only (no offline demos). Weights are assigned in place per
    online sample and surface as ``batch['weight']`` for weighted BC. Returns counts/weights.
    """
    labeled = []  # (transition, is_intervention)
    num_i = num_r = 0
    for t in online_buffer.get_all_transitions():
        if t is None:
            continue
        lab = int(round(float((t.info or {}).get("intervention", ROLLOUT_LABEL))))
        is_intv = lab == INTERVENTION_LABEL  # corrections only
        labeled.append((t, is_intv))
        num_i += int(is_intv)
        num_r += int(not is_intv)

    total = num_i + num_r
    if total == 0:
        logger.warning("IWR reweighting: dataset is empty; skipping.")
        return {"counts": {"intervention": 0, "robot": 0}, "weights": {}}

    w_intervention = 0.5 / (num_i / total) if num_i > 0 else 0.0
    w_robot = 0.5 / (num_r / total) if num_r > 0 else 0.0

    for t, is_intv in labeled:
        t.weight = float(w_intervention if is_intv else w_robot)

    logger.info(
        f"IWR reweighting | num_i={num_i} num_r={num_r} "
        f"w_intervention={w_intervention:.4f} w_robot={w_robot:.4f}"
    )
    return {
        "counts": {"intervention": num_i, "robot": num_r},
        "weights": {"intervention": w_intervention, "robot": w_robot},
    }


def load_precollected_hitl(path, online_buffer, lowdim_stats, remove_obs_keys):
    """Load a precollected HITL HDF5 (written by ``scripts/collect_hitl_rollouts_publish.py``) into a buffer.

    Each ``/data/demo_i`` group becomes one episode (``episode_id`` = demo name); per-step
    ``intervention`` labels are restored into ``info['intervention']`` (0=policy, 1=human; falls back
    to 0 if absent). Observations are stored RAW in the file, so low-dim keys present in
    ``lowdim_stats`` are re-normalized (z-scored) here to match what online collection would have
    stored (and what the algorithm / eval expect). ``next_obs`` is derived from the next step's obs
    (terminal step reuses its own obs), matching H5ReplayBuffer's convention.

    Returns a dict with ``transitions`` / ``episodes`` / ``human`` counts.
    """
    import h5py

    remove = set(remove_obs_keys or [])

    def _demo_sort_key(d):
        tail = d.rsplit("_", 1)[-1]
        return (0, int(tail)) if tail.isdigit() else (1, d)

    n_added = n_episodes = n_human = 0
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{path} is not a valid dataset (no /data group).")
        for demo in sorted(f["data"].keys(), key=_demo_sort_key):
            g = f["data"][demo]
            actions = np.asarray(g["actions"], dtype=np.float32)
            n = len(actions)
            if n == 0:
                continue
            rewards = np.asarray(g["rewards"], dtype=np.float32) if "rewards" in g else np.zeros(n, np.float32)
            dones = np.asarray(g["dones"], dtype=np.float32) if "dones" in g else np.zeros(n, np.float32)
            interv = np.asarray(g["intervention"]).astype(int) if "intervention" in g else np.zeros(n, dtype=int)

            obs_grp = g["obs"]
            obs_keys = [k for k in obs_grp.keys() if k not in remove]
            obs_arrays = {k: np.asarray(obs_grp[k]) for k in obs_keys}

            def _obs(t):
                od = {}
                for k in obs_keys:
                    v = obs_arrays[k][t]
                    st = lowdim_stats.get(k)
                    if st is not None:  # invert the raw-obs storage -> z-scored, as the worker stores
                        v = ((np.asarray(v, dtype=np.float32) - st["mean"]) / st["std"]).astype(np.float32)
                    od[k] = v
                return od

            obs_list = [_obs(t) for t in range(n)]
            n_interv_ep = int((interv == 1).sum())
            for t in range(n):
                next_obs = obs_list[t + 1] if t < n - 1 else obs_list[t]
                online_buffer.add(
                    obs=obs_list[t],
                    action=actions[t],
                    reward=float(rewards[t]),
                    next_obs=next_obs,
                    done=float(bool(dones[t])),
                    truncated=0.0,
                    episode_id=str(demo),
                    step_in_episode=t,
                    info={
                        "intervention": int(interv[t]),
                        "episode_len": n,
                        "episode_num_interventions": n_interv_ep,
                    },
                )
            n_added += n
            n_episodes += 1
            n_human += n_interv_ep
    return {"transitions": n_added, "episodes": n_episodes, "human": n_human}
