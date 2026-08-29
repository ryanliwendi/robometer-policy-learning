"""ThriftyDAgger gating on pi05 in the real world.

The core method is implemented in thrifty_gate.py. This file supplies the integration with real
world droid in the following ways: 

    sim (LIBERO)                     real (DROID)
    -------------------------------  --------------------------------------------------
    DP student's ``encode_obs``      frozen DINOv2 over exterior + wrist, plus the state
    episodic replay buffer           the round's ``episode_*.npz`` files
    DP ``sample_actions``            the action recorded at that frame

Q risk is trained on both success and failure rollouts (which can be obtained from the 
eval rollouts). The policy ensemble can be trained on both offline successes and 
expert intervention data. 
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from robometer_policy_learning.modules.encoders.image_encoders import DinoImageFeaturizer

LOWDIM_KEYS = ("joint_position", "gripper_position")
IMAGE_KEYS = ("exterior_image", "wrist_image")
CACHE_DIRNAME = "dino_cache"        # sidecar features, next to but not among the episodes


def cache_path(episode: pathlib.Path, model_id: str) -> pathlib.Path:
    """Where one episode's DINOv2 features gets stored."""
    return episode.parent / CACHE_DIRNAME / f"{episode.stem}.{model_id.replace('/', '-')}.npz"


def episode_image_features(episode: pathlib.Path, npz, actor, batch: int = 64) -> np.ndarray:
    """Both views' DINOv2 features for every frame of one episode, cached on disk.

    Staleness is judged from the source file's size and mtime rather than its contents, so a valid
    cache is confirmed without ever decompressing the images. Rewriting an npz in place (the
    success backfill, the length repair) invalidates the cache and costs a re-encode.
    """
    path = cache_path(episode, actor.model_id)
    st = episode.stat()
    n = len(npz["actions"])
    if path.exists():
        with np.load(path, allow_pickle=False) as c:
            if (str(c["model_id"]) == actor.model_id and int(c["n_frames"]) == n
                    and int(c["src_size"]) == st.st_size
                    and int(c["src_mtime_ns"]) == st.st_mtime_ns):
                return c["feat"]

    if any(p.requires_grad for p in getattr(actor, "encoder", torch.nn.Module()).parameters()):
        raise RuntimeError(
            "the image encoder has trainable parameters, so its features change as it trains and "
            "caching them would feed the ensemble stale vectors. Freeze it or drop the cache.")
    ext, wrist = npz[IMAGE_KEYS[0]], npz[IMAGE_KEYS[1]]        # the only read of the images
    feats = [actor.encode_images(ext[i:i + batch], wrist[i:i + batch]).cpu().numpy()
             for i in range(0, n, batch)]
    feat = np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, 0), np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")        # atomic, so a killed refresh cannot leave a partial
    np.savez(tmp, feat=feat, model_id=np.asarray(actor.model_id), n_frames=np.asarray(n),
             src_size=np.asarray(st.st_size), src_mtime_ns=np.asarray(st.st_mtime_ns))
    tmp.replace(path)
    return feat


def parse_source(spec: str) -> Dict[str, Any]:
    """Converts a string containing ``dir[:key=val,...]`` -> which episodes and which frames of them to take.
    
    ``segments=human|all`` keeps only the human expert's frames or every frame, 
    and ``require_success=0|1`` keeps only the episodes the human marked successful.
    Example input: data/task1_stack/rdagger/round1_train:segments=human,require_success=1 
    """
    path, _, opts = spec.partition(":")
    out: Dict[str, Any] = {"dir": path, "segments": "all", "require_success": False}
    for item in filter(None, opts.split(",")):
        key, _, val = item.partition("=")
        if key not in ("segments", "require_success"):
            raise ValueError(f"unknown option {key!r} in source {spec!r}")
        out[key] = val if key == "segments" else val.lower() not in ("0", "false", "no", "")
    if out["segments"] not in ("all", "human"):
        raise ValueError(f"segments must be 'all' or 'human', got {out['segments']!r} in {spec!r}")
    return out


class _Transition:
    __slots__ = ("done", "info")

    def __init__(self, done: float, success: float):
        self.done = np.asarray([float(done)], dtype=np.float32)
        self.info = {"is_success": bool(success > 0)}


class NpzThriftyBuffer:
    """One or more sources' episodes flattened into one indexable list of transitions.

    Mirrors the interface ``_IndexSpace`` expects of a sim replay buffer: length, a batch by
    explicit indices, per-row transitions, and episode boundaries.

    A ``segments=human`` source keeps only the takeover frames of an episode, which leaves gaps
    between them. Rows stay in time order and every episode still ends in a terminal, but ``row
    + 1`` may skip across a gap, so such a source belongs in the ensemble's buffer -- which only
    ever reads ``obs`` and ``action`` -- and not in the critics'.
    """

    def __init__(self, sources: List[Any] | str, actor):
        if isinstance(sources, str):
            sources = [sources]
        specs = [parse_source(s) if isinstance(s, str) else dict(s) for s in sources]

        self.feat, self.low, self.act = [], [], []
        self.done, self.success, self.actor = [], [], []
        self.boundaries: Dict[str, tuple] = {}
        self.sources: List[Dict[str, Any]] = []
        cursor = 0
        for spec in specs:
            files = sorted(pathlib.Path(spec["dir"]).expanduser().glob("episode_*.npz"))
            if not files:
                raise FileNotFoundError(f"no episode_*.npz under {spec['dir']}")
            kept, rows = 0, 0
            for path in files:
                # np.load on an npz is lazy, so the cheap keys are read here and the images are
                # only ever touched by episode_image_features, and only on a cache miss.
                with np.load(path, allow_pickle=False) as d:
                    succ = float(d["success"]) if "success" in d.files else 0.0
                    if spec["require_success"] and succ < 1.0:
                        continue
                    n = len(d["actions"])
                    actor_flags = d["actor"].astype(np.int8)
                    keep = (actor_flags == 1 if spec["segments"] == "human"
                            else np.ones(n, bool))
                    m = int(keep.sum())
                    if m < 2:
                        continue
                    low = np.concatenate(
                        [d["joint_position"], d["gripper_position"].reshape(n, -1)], axis=1)
                    actions = d["actions"]
                    feat = episode_image_features(path, d, actor)
                self.feat.append(feat[keep])
                self.low.append(low[keep].astype(np.float32))
                self.act.append(actions[keep].astype(np.float32))
                # Only the final step ends the episode, and it is a success only if the operator
                # said so. Failed endings are not filler: they are the only hard 0 the critics get.
                dn = np.zeros(m, np.float32); dn[-1] = 1.0
                sc = np.zeros(m, np.float32); sc[-1] = 1.0 if succ >= 1.0 else 0.0
                self.done.append(dn); self.success.append(sc)
                self.actor.append(actor_flags[keep])
                # Sources can share a directory basename, so key on the source too.
                self.boundaries[f"{len(self.sources)}/{path.stem}"] = (cursor, cursor + m - 1)
                cursor += m
                kept += 1
                rows += m
            if not kept:
                raise ValueError(f"every episode filtered out of {spec['dir']} "
                                 f"(segments={spec['segments']}, "
                                 f"require_success={spec['require_success']})")
            self.sources.append(dict(spec, episodes=kept, files=len(files), transitions=rows))
            print(f"[thrifty-real]   {spec['dir']} (segments={spec['segments']}, "
                  f"require_success={int(spec['require_success'])}): {kept}/{len(files)} episodes, "
                  f"{rows} transitions")

        self.feat = np.concatenate(self.feat)
        self.low = np.concatenate(self.low); self.act = np.concatenate(self.act)
        self.done = np.concatenate(self.done); self.success = np.concatenate(self.success)
        self.actor = np.concatenate(self.actor)
        self.total = int(len(self.act))
        self.n_success_endings = int(self.success.sum())
        self.n_failed_endings = int(((self.done > 0) & (self.success == 0)).sum())
        print(f"[thrifty-real] {len(self.boundaries)} episodes, {self.total} transitions, "
              f"{int((self.actor == 1).sum())} human, {self.n_success_endings} successful and "
              f"{self.n_failed_endings} failed endings")

    def __len__(self) -> int:
        return self.total

    def _next_index(self, idx: np.ndarray) -> np.ndarray:
        """Returns the next index, except return the current index at an episode's last step."""
        nxt = np.minimum(idx + 1, self.total - 1)
        return np.where(self.done[idx] > 0, idx, nxt)

    def _obs(self, idx: np.ndarray, device) -> Dict[str, torch.Tensor]:
        return {
            # The images' features, already computed. encode_obs still concatenates the state
            # onto them, so an offline batch and a live observation are assembled by one function.
            "image_feat": torch.as_tensor(self.feat[idx]).to(device),
            "lowdim": torch.as_tensor(self.low[idx]).to(device),
            "policy_action": torch.as_tensor(self.act[idx]).to(device),
        }

    def batch_from_indices(self, idx: np.ndarray, device=None) -> Dict[str, Any]:
        idx = np.asarray(idx, dtype=np.int64)
        nxt = self._next_index(idx)
        return {
            "obs": self._obs(idx, device),
            "next_obs": self._obs(nxt, device),
            "action": torch.as_tensor(self.act[idx]).to(device),
            "done": torch.as_tensor(self.done[idx]).to(device),
            "truncated": torch.as_tensor(self.done[idx]).to(device),
        }

    def sample(self, batch_size: int, device=None) -> Dict[str, Any]:
        return self.batch_from_indices(
            np.random.randint(0, self.total, size=int(batch_size)), device=device)

    def transitions_from_indices(self, idx) -> List[_Transition]:
        idx = np.asarray(idx, dtype=np.int64)
        return [_Transition(self.done[i], self.success[i]) for i in idx]

    def get_episode_boundaries(self) -> Dict[str, tuple]:
        return self.boundaries


class DinoFeatureActor(nn.Module):
    """Truns the observation (left cam + wrist image + proprioceptive dims) into a feature vector,
    which we train the thrifty detectors (policy ensemble, Q critics) from.
    """

    def __init__(self, dinov2_model: str = "facebook/dinov2-base", device=None, lowdim_dim: int = 8):
        super().__init__()
        self.encoder = DinoImageFeaturizer(dinov2_model)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder.to(self.device_)
        self.model_id = dinov2_model if isinstance(dinov2_model, str) else "facebook/dinov2-base"
        self.image_dim = int(self.encoder.output_dim)
        self.lowdim_dim = int(lowdim_dim)        # 7 joint positions + 1 gripper
        self.image_feat_dim = len(IMAGE_KEYS) * self.image_dim
        self.feat_dim = self.image_feat_dim + self.lowdim_dim

    @torch.no_grad()
    def encode_images(self, exterior, wrist) -> torch.Tensor:
        """Both views through DINOv2, concatenated. The expensive half, and the cached one."""
        return torch.cat([self.encoder(torch.as_tensor(v).to(self.device_))
                          for v in (exterior, wrist)], dim=1)

    @torch.no_grad()
    def encode_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Image features plus the state. Takes ``image_feat`` when it has already been computed
        (an offline batch, from the cache) and the raw images otherwise (a live observation), so
        both paths produce the same vector by construction rather than by agreement."""
        img = (torch.as_tensor(obs["image_feat"]).to(self.device_).float()
               if "image_feat" in obs else
               self.encode_images(*(obs[k] for k in IMAGE_KEYS)))
        return torch.cat([img, torch.as_tensor(obs["lowdim"]).to(self.device_).float()], dim=1)

    @torch.no_grad()
    def sample_actions(self, obs: Dict[str, Any]) -> torch.Tensor:
        return torch.as_tensor(obs["policy_action"]).to(self.device_).float()


class RealThriftyAlgo:
    """Adapter object for train_thrifty_models and collect_thrifty_scores."""

    def __init__(self, actor: DinoFeatureActor, buffer: NpzThriftyBuffer, batch_size: int = 64):
        self.actor = actor
        self.buffer = buffer
        self.batch_size = int(batch_size)

    def _prepare_actions(self, action) -> torch.Tensor:
        # pi05 actions are already clipped to [-1, 1], the range the ensemble's action space
        # assumes, so this only adds the chunk axis the sim code indexes with [:, 0, :].
        a = torch.as_tensor(action).to(self.actor.device_).float()
        return a.unsqueeze(1) if a.dim() == 2 else a


def fit_from_npz(bc_sources, q_sources, *, target_rate: float = 0.01, grad_steps: int = 500,
                 num_nets: int = 5, batch_size: int = 64, lr: float = 1e-3, gamma: float = 0.9999,
                 seed: int = 0, dinov2_model: str = "facebook/dinov2-base", device=None,
                 actor: Optional["DinoFeatureActor"] = None):
    """Refit the policy ensemble, the critics and the gate thresholds from the rounds collected so far.

    ``bc_sources`` include the human's takeover frames and the policy's own successful rollouts.
    Both the ensemble policy and the threshold for novelty firing is calibrated using these sources.
    ``q_sources`` needs to include episodes that failed, so we use the eval rollouts.

    Returns (gate, ac, info).
    """
    from robometer_policy_learning.utils.thrifty_gate import (
        ThriftyGate, build_ensemble, collect_thrifty_scores, train_thrifty_models)

    actor = actor or DinoFeatureActor(dinov2_model=dinov2_model, device=device)
    device = actor.device_
    print("[thrifty-real] ensemble (novelty) buffer:")
    bc_buffer = NpzThriftyBuffer(bc_sources, actor)
    print("[thrifty-real] critic (risk) buffer:")
    q_buffer = NpzThriftyBuffer(q_sources, actor)
    algo = RealThriftyAlgo(actor, bc_buffer, batch_size=batch_size)

    act_dim = int(bc_buffer.act.shape[1])
    if int(q_buffer.act.shape[1]) != act_dim:
        raise ValueError(f"the two buffers disagree on the action dim "
                         f"({act_dim} vs {q_buffer.act.shape[1]})")
    ac = build_ensemble(actor.feat_dim, act_dim, device, num_nets=num_nets)
    ac_targ = build_ensemble(actor.feat_dim, act_dim, device, num_nets=num_nets)
    q_opt = torch.optim.Adam(list(ac.q1.parameters()) + list(ac.q2.parameters()), lr=lr)

    info = train_thrifty_models(
        algo, ac, ac_targ, lambda params: torch.optim.Adam(params, lr=lr), q_opt,
        grad_steps=grad_steps, gamma=gamma, num_nets=num_nets,
        feat_dim=actor.feat_dim, act_dim=act_dim, device=device, seed=seed, q_buffer=q_buffer)

    if len(bc_buffer) * target_rate < 10:
        print(f"[thrifty-real] WARNING: {len(bc_buffer)} calibration states cannot resolve a "
              f"target_rate of {target_rate} -- recalibrate takes the score at index "
              f"int((1-target_rate)*n), which lands at or near the sample maximum, so only a state "
              f"more novel than anything in the buffer will fire. Collect more episodes or raise "
              f"target_rate.")
    novelties, safeties = collect_thrifty_scores(
        algo, ac, device=device, indices=np.arange(len(bc_buffer)))
    gate = ThriftyGate()
    
    n_pos, n_neg = int(info.get("n_positive", 0)), int(info.get("n_negative", 0))
    risk_enabled = n_pos > 0 and n_neg > 0
    gate.recalibrate(novelties, safeties, target_rate=target_rate, risk_enabled=risk_enabled)
    if not risk_enabled:
        missing = "successful" if n_pos == 0 else "failed"
        print(f"[thrifty-real] the critic buffer holds no {missing} episode endings -> risk gate "
              f"disabled, novelty only. Point --q-sources at the solo eval rollouts, which is "
              f"where the failures are.")
    info["act_dim"] = act_dim
    info["bc_sources"] = bc_buffer.sources
    info["q_sources"] = q_buffer.sources
    print(f"[thrifty-real] {info} | delta_h={gate.delta_h:.4g} beta_h={gate.beta_h:.4g}")
    return gate, ac, info


class RealtimeThriftyScorer:
    """Scores the live DROID observation."""

    def __init__(self, actor: "DinoFeatureActor", ac, external_camera: str = "left"):
        self.actor = actor
        self.ac = ac
        self.external_camera = external_camera
        self.latencies: List[float] = []

    def reset(self):
        self.latencies = []

    def _obs_dict(self, curr_obs) -> Dict[str, Any]:
        from openpi_client import image_tools

        ext = image_tools.resize_with_pad(curr_obs[f"{self.external_camera}_image"], 224, 224)
        wrist = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        low = np.concatenate([
            np.asarray(curr_obs["joint_position"], dtype=np.float32).reshape(-1),
            np.asarray(curr_obs["gripper_position"], dtype=np.float32).reshape(-1),
        ]).astype(np.float32)
        return {
            "exterior_image": np.asarray(ext, dtype=np.uint8)[None],
            "wrist_image": np.asarray(wrist, dtype=np.uint8)[None],
            "lowdim": low[None],
        }

    @torch.no_grad()
    def score(self, curr_obs, action) -> tuple:
        """(novelty, safety) for the state pi05 is about to act in, with its proposed action."""
        import time as _time

        t0 = _time.time()
        feat = self.actor.encode_obs(self._obs_dict(curr_obs)).cpu().numpy()
        a = np.asarray(action, dtype=np.float32).reshape(1, -1)
        novelty = float(self.ac.variance(feat))
        safety = float(self.ac.safety(feat, a))
        self.latencies.append(_time.time() - t0)
        return novelty, safety

    def latency_report(self, control_hz: float) -> str:
        if not self.latencies:
            return "[thrifty] no scores this episode"
        lat = np.asarray(self.latencies)
        budget = 1.0 / float(control_hz)
        return (f"[thrifty] {len(lat)} scores | p50={np.percentile(lat, 50) * 1e3:.0f}ms "
                f"p95={np.percentile(lat, 95) * 1e3:.0f}ms | budget={budget * 1e3:.0f}ms "
                f"({float((lat > budget).mean()):.0%} over)")


def save_thrifty(path, gate, ac, actor: "DinoFeatureActor", act_dim: int, meta=None) -> None:
    """Everything the control loop needs to rebuild the gate, in one file."""
    torch.save({
        "ensemble": ac.state_dict(),
        "delta_h": float(gate.delta_h),
        "beta_h": float(gate.beta_h),
        "risk_enabled": bool(gate.risk_enabled),
        "feat_dim": int(actor.feat_dim),
        "act_dim": int(act_dim),
        "num_nets": len(ac.pis),
        "dinov2_model": getattr(actor, "model_id", "facebook/dinov2-base"),
        "meta": meta or {},
    }, str(path))
    print(f"[thrifty-real] saved gate -> {path}")


def load_thrifty(path, device=None):
    """Rebuild (gate, ensemble, actor) from a refresh checkpoint. Returns them ready to score."""
    from robometer_policy_learning.utils.thrifty_gate import ThriftyGate, build_ensemble

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    actor = DinoFeatureActor(dinov2_model=ckpt["dinov2_model"], device=device)
    if actor.feat_dim != ckpt["feat_dim"]:
        raise ValueError(
            f"feature dim changed since the refresh ({actor.feat_dim} now vs {ckpt['feat_dim']} "
            "then) -- the thresholds would be meaningless. Re-run the refresh.")
    ac = build_ensemble(ckpt["feat_dim"], ckpt["act_dim"], actor.device_, num_nets=ckpt["num_nets"])
    ac.load_state_dict(ckpt["ensemble"])
    gate = ThriftyGate(delta_h=ckpt["delta_h"], beta_h=ckpt["beta_h"])
    gate.risk_enabled = bool(ckpt["risk_enabled"])
    print(f"[thrifty-real] loaded gate from {path} | delta_h={gate.delta_h:.4g} "
          f"beta_h={gate.beta_h:.4g} risk_enabled={gate.risk_enabled}")
    return gate, ac, actor
