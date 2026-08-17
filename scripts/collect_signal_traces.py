#!/usr/bin/env python3
"""Q1 in simulation: record every gate's signal on the SAME rollouts.

The real-world Q1 (`analyze_real_world_labels.py`) asks whether the Robometer progress signal
separates successful teleop episodes from failed ones. This is its sim twin, extended to the two
baseline signals -- Diff-DAgger's diffusion loss and ThriftyDAgger's novelty / Q-risk -- so the
three can be compared as *failure detectors* before any gating rule is applied.

Everything is recorded in ONE ungated pass per task. That matters: each signal is computed from
the student's own network, and scoring draws from the torch RNG, so three separate runs would
diverge into three different sets of episodes and the AUCs would no longer be comparable
episode-for-episode. Here every signal sees the same trajectory, the same failures, and (for the
signals that need one) the same sampled action.

The student flies solo -- the gate never fires (`NeverGate`), so no expert is loaded and the
trajectories are the policy's unaltered behaviour, exactly like the `n200_*` corpora.

Both baselines are taken at their DAgger iteration-0 state, which is the state that matches these
rollouts: the student is the pretrained checkpoint, so Diff-DAgger calibrates on the offline demos
and ThriftyDAgger fits its ensemble on the offline demos. Several variants of each are carried at
once (`--dd-batch-multipliers`, `--thrifty-train-steps`) because their scoring cost is negligible
next to one shared action sample -- that is the baseline hyperparameter sweep, done in a single
pass instead of one rollout campaign per setting.

Usage (one task):
    uv run python scripts/collect_signal_traces.py \
        --student-dir outputs/task1_dp_50k/20-06-31 --student-checkpoint 50000 \
        --episodes 200 --out gated_videos/sigq_t1/signal_traces.json
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import copy
import json
import sys
import time

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_rollout_worker import GatedRolloutWorker, load_actor  # noqa: E402
from robometer_policy_learning.algorithms.dp import DPConfig  # noqa: E402
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer  # noqa: E402
from robometer_policy_learning.buffers.samplers import (  # noqa: E402
    ChunkedSequentialSampler, RandomSampler)
from robometer_policy_learning.utils.diffdagger_gate import (  # noqa: E402
    DiffusionLossScorer, quantile_threshold)
from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device  # noqa: E402
from robometer_policy_learning.utils.reward_gate import NeverGate  # noqa: E402
from robometer_policy_learning.utils.thrifty_gate import (  # noqa: E402
    build_ensemble, collect_thrifty_scores, train_thrifty_models)

LABEL_OFFLINE = 2


class MultiSignalScorer:
    """Every gating signal at once, from one observation and one sampled action.

    Returns a tuple of floats ordered like ``self.names``. The per-signal maths is the same
    computation the deployed scorers do (``RobometerScorer.score``, ``DiffusionLossScorer.score``,
    ``ThriftyScorer.score``); the only change is that the student's encoding and its sampled action
    chunk are computed once and shared, rather than once per scorer. That makes the Diff-DAgger
    variants an exact N_b ablation (same action, different sample count) and removes ~4 redundant
    reverse-diffusion passes per scored step.
    """

    def __init__(self, actor, robometer_scorer, dd_scorers, thrifty_acs, remove_obs_keys=None,
                 device=None):
        self.actor = actor
        self.robometer = robometer_scorer
        self.dd_scorers = list(dd_scorers)          # [(name, DiffusionLossScorer)]
        self.thrifty_acs = list(thrifty_acs)        # [(name, Ensemble)]
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.device = device or next(actor.parameters()).device
        self._last_obs = None

        self.names = (["robometer_progress", "robometer_success_prob"]
                      + [f"dd_loss_{n}" for n, _ in self.dd_scorers]
                      + [f"{p}_{n}" for n, _ in self.thrifty_acs for p in ("novelty", "safety")])

    def reset(self, task: str = ""):
        self.robometer.reset(task=task)
        self._last_obs = None

    def observe(self, obs, frame_key=None):
        self.robometer.observe(obs, frame_key)
        self._last_obs = obs

    @torch.no_grad()
    def score(self):
        progress, success_prob = self.robometer.score()
        values = [progress, success_prob]

        prepped = {k: v for k, v in self._last_obs.items() if k not in self.remove_obs_keys}
        actor_obs = move_to_device(convert_to_tensor(prepped), self.device)
        global_cond = self.actor.encode_obs(actor_obs)            # (1, D)
        action = self.actor.sample_actions(actor_obs)             # (1, H, A), normalised
        if action.dim() == 2:
            action = action.unsqueeze(1)

        for _, s in self.dd_scorers:
            values.append(float(s._avg_loss(global_cond, action)[0].item()))

        feat = global_cond.cpu().numpy()
        act0 = action[:, 0, :].cpu().numpy()
        for _, ac in self.thrifty_acs:
            values.append(float(ac.variance(feat)))
            values.append(float(ac.safety(feat, act0)))

        return tuple(values), 0.0


def build_thrifty(algo, feat_dim, action_dim, device, train_steps, num_nets, lr, gamma, seed):
    """ThriftyDAgger's iteration-0 fit: novelty ensemble + twin Q-risk critics on the offline
    demos, exactly as `train_dagger.py` does before its first collection round."""
    ac = build_ensemble(feat_dim, action_dim, device, num_nets=num_nets)
    ac_targ = copy.deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False
    q_opt = torch.optim.Adam(list(ac.q1.parameters()) + list(ac.q2.parameters()), lr=lr)
    t0 = time.time()
    losses = train_thrifty_models(
        algo, ac, ac_targ, ens_opt_fn=lambda p: torch.optim.Adam(p, lr=lr), q_opt=q_opt,
        grad_steps=train_steps, gamma=gamma, num_nets=num_nets, feat_dim=feat_dim,
        act_dim=action_dim, device=device, seed=seed)
    nov, saf = collect_thrifty_scores(algo, ac, device=device)
    logger.info(f"[thrifty:{train_steps}steps] {time.time() - t0:.0f}s "
                f"ens_loss={losses['ensemble_loss']:.5f} q_loss={losses['qrisk_loss']:.5f} "
                f"n_pos={losses['n_positive']} n_trans={losses['n_transitions']} | "
                f"offline novelty med={np.median(nov):.5f} risk(Q) med={np.median(saf):.5f}")
    return ac, dict(train_steps=train_steps, num_nets=num_nets,
                    ensemble_loss=losses["ensemble_loss"], qrisk_loss=losses["qrisk_loss"],
                    n_positive=int(losses["n_positive"]),
                    n_transitions=int(losses["n_transitions"]),
                    novelty_median=float(np.median(nov)), risk_median=float(np.median(saf)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-dir", required=True)
    ap.add_argument("--student-checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-every", type=int, default=1)
    ap.add_argument("--out", required=True, help="output JSON of per-episode signal traces")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing --out instead of overwriting it (scavenger-safe)")
    ap.add_argument("--reward-model", default="jesbu1/robometer-4b-fft-libero")
    ap.add_argument("--no-robometer", action="store_true",
                    help="skip the 4B reward model (fast dry run of the baseline signals)")
    # Diff-DAgger: N_b = num_train_timesteps * batch_multiplier. The paper's Table IV N_b = 512
    # is batch_multiplier 5 at our T = 100; larger values buy a lower-variance score.
    ap.add_argument("--dd-batch-multipliers", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--dd-calib-samples", type=int, default=1024)
    ap.add_argument("--dd-alpha", type=float, default=0.99)
    # ThriftyDAgger: the reference fits its ensemble to convergence on D_exp; the rdagger loop's
    # default is 200 steps, which may simply be undertrained. Carry both and let the data say.
    ap.add_argument("--thrifty-train-steps", type=int, nargs="+", default=[200, 2000])
    ap.add_argument("--thrifty-num-nets", type=int, default=5)
    ap.add_argument("--thrifty-lr", type=float, default=1e-3)
    ap.add_argument("--thrifty-gamma", type=float, default=0.9999)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Env / model / chunking come from the student's own pretraining config, like every other
    # entry point in this repo.
    pre_cfg = OmegaConf.load(os.path.join(args.student_dir, ".hydra", "config.yaml"))
    dino_image_keys = list(OmegaConf.select(pre_cfg, "env.dino_image_keys", default=[]) or [])
    n_exec = int(OmegaConf.select(pre_cfg, "training.n_action_steps", default=10))
    chunk_size = OmegaConf.select(pre_cfg, "training.chunk_size", default=None)
    normalize_lowdim = bool(OmegaConf.select(pre_cfg, "training.normalize_lowdim_obs",
                                             default=False))

    dinov2_model = dinov2_processor = None
    if dino_image_keys:
        from transformers import AutoImageProcessor, AutoModel

        model_id = OmegaConf.select(pre_cfg, "model.dinov2_model", default="facebook/dinov2-base")
        dinov2_model = AutoModel.from_pretrained(model_id).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(model_id)

    from robometer_policy_learning.utils.env_utils import make_env

    env, _ = make_env(
        env_name=f"{pre_cfg.env.env_name}/{pre_cfg.env.task_id}",
        num_envs=1,
        max_episode_steps=int(pre_cfg.env.max_episode_steps),
        chunk_size=None,
        n_action_steps=1,
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        device=device,
        dino_image_keys=dino_image_keys,
        seed=args.seed,
    )
    action_dim = int(env.single_action_space.shape[0])

    student = load_actor(args.student_dir, device, args.student_checkpoint)
    remove_obs_keys = list(getattr(student, "remove_obs_keys", None)
                           or OmegaConf.select(pre_cfg, "env.extra_keys_to_drop", default=[]) or [])

    # ---- Offline demos: Diff-DAgger's calibration set and ThriftyDAgger's training set ----
    asp = env.single_action_space
    finite = np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high))
    sampler = (RandomSampler() if chunk_size is None else
               ChunkedSequentialSampler(chunk_size=int(chunk_size), gamma=0.99,
                                        obs_as_sequence=False))
    offline_buffer = H5ReplayBuffer(
        h5_paths=[pre_cfg.env.h5_dataset_path],
        sampler=sampler,
        remove_obs_keys=list(remove_obs_keys),
        dinov2_model=dinov2_model,
        dinov2_processor=dinov2_processor,
        dino_embedding_keys=dino_image_keys,
        min_action=np.asarray(asp.low, np.float32) if finite else None,
        max_action=np.asarray(asp.high, np.float32) if finite else None,
        normalize_lowdim_obs=normalize_lowdim,
        default_intervention_label=LABEL_OFFLINE,
    )
    lowdim_stats = offline_buffer.lowdim_obs_stats

    # A real DP algo over those demos: ThriftyDAgger's fit needs `algo.buffer`, `algo.batch_size`,
    # `algo.actor.encode_obs` and `algo._prepare_actions`. use_ema=False so `algo.actor` IS the
    # checkpoint being rolled out -- nothing is trained here, so the EMA copy would only introduce
    # a second network.
    algo_dict = OmegaConf.to_container(OmegaConf.select(pre_cfg, "offline_algorithm"),
                                       resolve=True)
    algo_dict["use_ema"] = False
    algo_cfg = DPConfig(**algo_dict)
    algo_cfg.actor = student
    algo_cfg.buffer = offline_buffer
    algo = algo_cfg.create()

    # ---- Signals ----
    dd_scorers, dd_meta = [], {}
    for bm in args.dd_batch_multipliers:
        s = DiffusionLossScorer(student, batch_multiplier=bm, num_per_batch=1, device=device)
        name = f"nb{s.num_train_timesteps * bm}"
        losses = s.calibrate_from_buffer(offline_buffer, num_samples=args.dd_calib_samples)
        thr = quantile_threshold(losses, args.dd_alpha)
        dd_meta[name] = dict(batch_multiplier=bm, n_b=s.num_train_timesteps * bm,
                             alpha=args.dd_alpha, threshold=float(thr),
                             calib_n=int(len(losses)), calib_mean=float(losses.mean()),
                             calib_max=float(losses.max()))
        logger.info(f"[diffdagger:{name}] N_b={dd_meta[name]['n_b']} "
                    f"offline loss mean={losses.mean():.6f} max={losses.max():.6f} "
                    f"alpha={args.dd_alpha} threshold={thr:.6f}")
        dd_scorers.append((name, s))

    thrifty_acs, thrifty_meta = [], {}
    feat_dim = int(student.global_cond_dim)
    for steps in args.thrifty_train_steps:
        ac, info = build_thrifty(algo, feat_dim, action_dim, device, steps,
                                 args.thrifty_num_nets, args.thrifty_lr, args.thrifty_gamma,
                                 seed=args.seed)
        name = f"s{steps}"
        thrifty_acs.append((name, ac))
        thrifty_meta[name] = info

    if args.no_robometer:
        class _NullRobometer:
            def reset(self, task=""): pass
            def observe(self, obs, frame_key=None): pass
            def score(self): return (float("nan"), float("nan"))
        robometer = _NullRobometer()
    else:
        from robometer_policy_learning.utils.reward_gate import RobometerScorer

        robometer = RobometerScorer(model_path=args.reward_model, device=device)

    scorer = MultiSignalScorer(student, robometer, dd_scorers, thrifty_acs,
                               remove_obs_keys=remove_obs_keys, device=device)
    logger.info(f"signals: {scorer.names}")

    worker = GatedRolloutWorker(
        env=env,
        student=student,
        expert=student,   # never used: NeverGate never hands over. Avoids loading pi0/JAX.
        scorer=scorer,
        gate=NeverGate(),
        online_buffer=None,
        device=device,
        action_dim=action_dim,
        lowdim_stats=lowdim_stats,
        remove_obs_keys=remove_obs_keys,
        student_n_action_steps=n_exec,
        expert_n_action_steps=n_exec,
        score_every=args.score_every,
        video_dir=None,
    )

    meta = dict(
        student_dir=args.student_dir, student_checkpoint=args.student_checkpoint,
        task_id=int(OmegaConf.select(pre_cfg, "env.task_id", default=-1)),
        env_name=str(OmegaConf.select(pre_cfg, "env.env_name", default="")),
        max_episode_steps=int(pre_cfg.env.max_episode_steps),
        n_action_steps=n_exec, seed=args.seed, score_every=args.score_every,
        signals=scorer.names,
        reward_model=(None if args.no_robometer else args.reward_model),
        diffdagger=dd_meta, thrifty=thrifty_meta,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ---- picking up where a killed run left off ----------------------------------------------
    # Each episode reseeds the random number generators from `seed + ep`, and the scorers are
    # rebuilt from `seed` when the script starts. So episode 37 is the same episode whether you
    # get there in one run or three. That means a job killed halfway can just carry on from the
    # last saved episode instead of starting over. The check below refuses to continue a file that
    # came from a different policy or a different set of signals.
    episodes = []
    if args.resume and os.path.exists(args.out):
        prev = json.load(open(args.out))
        keyed = ("student_dir", "student_checkpoint", "task_id", "seed", "score_every",
                 "signals", "max_episode_steps")
        mismatch = [k for k in keyed if prev["meta"].get(k) != meta.get(k)]
        if mismatch:
            raise SystemExit(f"--resume refused: {args.out} differs on {mismatch}. "
                             f"Move it aside or drop --resume to overwrite.")
        episodes = prev["episodes"][:args.episodes]
        meta.setdefault("resumed_at", []).extend(prev["meta"].get("resumed_at", []))
        meta["resumed_at"].append(len(episodes))
        # The scorers get retrained when the script starts, not loaded from disk. Same seed and
        # same order of operations, so they should come out identical -- but they won't if the
        # training code itself changed since the file was written. Then the old episodes were
        # scored by one network and the new ones by another, and the file is silently two different
        # datasets glued together. This actually happened: the 2026-08-10 run predates the
        # compute_loss_q rewrite, so its novelty scores still match exactly while its risk scores
        # don't. We stop with an error rather than warn, because nothing further down the pipeline
        # could ever notice the problem.
        drift = []
        for name, old in prev["meta"].get("thrifty", {}).items():
            new = thrifty_meta.get(name, {})
            for key in ("novelty_median", "risk_median", "ensemble_loss", "qrisk_loss"):
                a, b = old.get(key), new.get(key)
                if a is None or b is None or (abs(a) and abs(b - a) > 0.01 * abs(a)):
                    drift.append(f"{name}.{key}: stored {a} -> refit {b}")
        if drift:
            raise SystemExit(
                "--resume refused: the refit scorers do not match the ones that produced "
                f"{args.out}, so its {len(episodes)} episodes were scored by different networks.\n"
                + "\n".join(f"  {d}" for d in drift)
                + "\nMove the file aside and recollect from scratch.")
        logger.info(f"resuming {args.out} at episode {len(episodes)}/{args.episodes} "
                    f"({sum(e['success'] for e in episodes)} success so far)")

    for ep in range(len(episodes), args.episodes):
        _random.seed(args.seed + ep)
        np.random.seed(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        torch.cuda.manual_seed_all(args.seed + ep)

        t0 = time.time()
        stats = worker.rollout_episode(f"sig_{ep}", store=False)
        traces = np.asarray(stats["score_trace"], dtype=np.float64)   # (steps, n_signals)
        episodes.append(dict(
            episode=ep, success=bool(stats["success"]), steps=int(stats["steps"]),
            traces={n: traces[:, i].tolist() for i, n in enumerate(scorer.names)},
        ))
        dt = time.time() - t0
        logger.info(f"episode {ep}: steps={stats['steps']} success={stats['success']} "
                    f"({dt:.0f}s, {1000 * dt / max(1, stats['steps']):.0f} ms/step)")

        # Re-dump every episode: these runs are hours long, don't lose them to a late crash.
        with open(args.out, "w") as f:
            json.dump(dict(meta=meta, episodes=episodes), f)

    n_succ = sum(e["success"] for e in episodes)
    logger.info(f"DONE: {n_succ}/{len(episodes)} success -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
