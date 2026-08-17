#!/usr/bin/env python3
"""Rescore every gate's signal on rollouts that were already recorded.

`collect_signal_traces.py` computes the same signals, but it has to drive the simulator to do it,
so a new alpha grid or a bigger ensemble means another rollout campaign. This reads the arrays
`gated_rollout_worker.py --dump-obs` wrote and recomputes the signals offline: no simulator, no
Robometer, minutes instead of hours.

What comes out is a `signal_traces.json` in the same format `collect_signal_traces.py` writes, so
`score_detection_frontier.py` and `analyze_signal_quality.py` read it without changes.

  Robometer progress   copied across from the run's episode_stats.json (it needs camera frames,
                       which are far too large to have kept)
  dd_loss_*            Diff-DAgger's diffusion loss at the action that was actually executed
  novelty_*, safety_*  ThriftyDAgger's ensemble disagreement and twin-Q risk
  logpzo_*             LogpZO's flow-matching density score

Diff-DAgger and ThriftyDAgger are fit on the offline demos, which is what those methods do.
LogpZO is different: the paper fits it on successful rollouts of the same policy, so that is what
happens here. The episodes are split into groups and each group is scored by a flow trained on
successful episodes from the other groups, so no episode is scored by a model that saw it.

Each of the two learned detectors can be taken at two points in its life, and they answer different
questions:

  fit here      trained on the offline demos, matching the state the rollouts came from. The
                like-for-like comparison against a gate that needs no training.
  --thrifty-ac  loaded from a finished DAgger run, after every retrain. The baseline at its
  --logpzo-ckpt strongest. Note it has seen states collected by a better student than the one that
                produced these rollouts, so it is a best case, not a fair detector comparison.

Usage:
    uv run python scripts/score_signals_offline.py \
        --student-dir outputs/task1_dp_50k/20-06-31 --student-checkpoint 50000 \
        --arrays-dir gated_videos/n200_t1_dp_arrays/arrays \
        --stats gated_videos/n200_t1_dp_arrays/episode_stats.json \
        --out gated_videos/sigq_t1_offline/signal_traces.json
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import copy
import zlib
import glob
import json
import re
import sys
import time

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_rollout_worker import load_actor  # noqa: E402
from robometer_policy_learning.algorithms.dp import DPConfig  # noqa: E402
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer  # noqa: E402
from robometer_policy_learning.buffers.samplers import (  # noqa: E402
    ChunkedSequentialSampler, RandomSampler)
from robometer_policy_learning.utils.diffdagger_gate import (  # noqa: E402
    DiffusionLossScorer, quantile_threshold)
from robometer_policy_learning.utils.logpzo_gate import (  # noqa: E402
    LogpZOModel, fit_flow_on_episodes, score_features)
from robometer_policy_learning.utils.thrifty_gate import (  # noqa: E402
    build_ensemble, collect_thrifty_scores, train_thrifty_models)

LABEL_OFFLINE = 2


def episode_number(path: str) -> int:
    """`.../watch_37.npz` -> 37. The stats JSON lists episodes in this order."""
    m = re.search(r"(\d+)\.npz$", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot read an episode number from {path}")
    return int(m.group(1))


HELD_IN = -2  # marks an episode LogpZO was built from, so the frontier can drop it


def assign_folds(success: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Put each episode in one of k groups, dealing successes and failures out one at a time so
    every group has some of both."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(success), dtype=np.int64)
    for cls in (True, False):
        idx = np.flatnonzero(success == cls)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def assign_budget(n_episodes: int, budget: int, seed: int) -> np.ndarray:
    """Hand LogpZO `budget` randomly chosen episodes and mark the rest as the evaluation set.

    This is the realistic version: on a real robot you collect a fixed number of rollouts, label
    them, and only the successful ones are usable. Which ones succeed is not yours to choose, so
    the draw is not balanced by outcome.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_episodes)
    fold = np.zeros(n_episodes, dtype=np.int64)
    fold[idx[:budget]] = HELD_IN
    return fold


def fit_logpzo_folds(cache, fold_of, feat_dim, device, train_steps, lr, seed, folds, batch_size):
    """Fit LogpZO the way the paper does: on successful rollouts of the same policy.

    Two ways to split the episodes, decided by `fold_of`:

    groups   every episode is scored by a flow built from the other groups, so all 200 episodes
             end up in the frontier. Uses as many rollouts as exist.
    budget   one flow is built from a fixed set of held-in episodes and scores everything. The
             held-in episodes are dropped from the frontier, so the numbers say what LogpZO can
             do after collecting only that many rollouts.

    Either way the successful episodes it is given are split in half: one half trains the flow, the
    other half is scored and kept so the frontier can build the band from episodes the flow never
    trained on.

    Returns the per-episode score traces, the calibration traces per group, and a fit summary.
    """
    success = np.array([c["success"] for c in cache])
    scores = [None] * len(cache)
    calib_traces = {}
    fit_logs = []

    # With a budget, one flow is built from the held-in episodes and scores everything else.
    # Otherwise every group gets its own flow, built from the other groups.
    budget_mode = HELD_IN in set(fold_of.tolist())
    groups = [HELD_IN] if budget_mode else list(range(folds))

    for k in groups:
        outside = np.flatnonzero(fold_of == HELD_IN) if budget_mode else np.flatnonzero(fold_of != k)
        succ_out = outside[success[outside]]
        if len(succ_out) < 4:
            raise SystemExit(f"fold {k}: only {len(succ_out)} successful episodes outside it; "
                             "LogpZO needs successful rollouts to fit and to calibrate")
        rng = np.random.default_rng(seed + 1000 + k)
        rng.shuffle(succ_out)
        half = len(succ_out) // 2
        flow_eps, band_eps = succ_out[:half], succ_out[half:]

        torch.manual_seed(seed + zlib.crc32(f"logpzo_fold{k}_s{train_steps}".encode()) % 100000)
        model = LogpZOModel(feat_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        info = fit_flow_on_episodes(model, opt, [cache[i]["feat"] for i in flow_eps], train_steps,
                                    device, batch_size=256, seed=seed)

        # With a budget this one flow scores every episode; the held-in ones are dropped later.
        judged = np.arange(len(cache)) if budget_mode else np.flatnonzero(fold_of == k)
        for i in judged:
            scores[i] = score_features(model, cache[i]["feat"], device, batch_size)
        # "0" is the group the evaluation episodes end up in once the held-in ones are dropped.
        calib_traces["0" if budget_mode else str(k)] = [
            score_features(model, cache[i]["feat"], device, batch_size).tolist() for i in band_eps]
        fit_logs.append(dict(fold=k, n_flow_episodes=int(len(flow_eps)),
                             n_band_episodes=int(len(band_eps)),
                             n_transitions=int(info["n_transitions"]),
                             train_loss=float(info["train_loss"]),
                             val_loss=float(info["val_loss"])))
        logger.info(f"[logpzo:fold{k}] flow on {len(flow_eps)} episodes "
                    f"({info['n_transitions']} steps) train_loss={info['train_loss']:.5f} "
                    f"val_loss={info['val_loss']:.5f}; band from {len(band_eps)} episodes")
        del model

    meta = dict(fit_on="successful_rollouts", train_steps=int(train_steps), folds=int(folds),
                lr=float(lr), per_fold=fit_logs)
    return scores, calib_traces, meta


def build_thrifty(algo, feat_dim, action_dim, device, train_steps, num_nets, lr, gamma, seed):
    """ThriftyDAgger's iteration-0 fit: novelty ensemble + twin Q on the offline demos."""
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
                f"n_pos={losses['n_positive']} | offline novelty med={np.median(nov):.5f} "
                f"risk med={np.median(saf):.5f}")
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
    ap.add_argument("--arrays-dir", required=True, help="directory of per-episode .npz files")
    ap.add_argument("--stats", default=None,
                    help="episode_stats.json from the same run; supplies the Robometer trace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=256, help="steps scored at once")
    ap.add_argument("--seed", type=int, default=0)
    # Diff-DAgger
    ap.add_argument("--dd-batch-multipliers", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--dd-calib-samples", type=int, default=1024)
    ap.add_argument("--dd-alpha", type=float, default=0.99)
    # ThriftyDAgger
    ap.add_argument("--thrifty-train-steps", type=int, nargs="+", default=[200, 2000])
    ap.add_argument("--thrifty-num-nets", type=int, default=5)
    ap.add_argument("--thrifty-lr", type=float, default=1e-3)
    ap.add_argument("--thrifty-gamma", type=float, default=0.9999)
    ap.add_argument("--thrifty-ac", nargs="+", default=None,
                    help="thrifty_ac.pt files from finished DAgger runs, as name=path. Each one "
                         "becomes its own variant next to the s<steps> ones fit here, so the "
                         "baseline can be read with the ensemble it actually ends up with. A bare "
                         "path is named 'final'.")
    # LogpZO
    ap.add_argument("--logpzo-train-steps", type=int, nargs="+", default=[2000])
    ap.add_argument("--logpzo-lr", type=float, default=1e-4)
    ap.add_argument("--logpzo-folds", type=int, default=5,
                    help="episodes are split into this many groups; each group is scored by a "
                         "flow trained on successful episodes from the other groups")
    ap.add_argument("--logpzo-budget", type=int, default=0,
                    help="instead of groups, give LogpZO this many randomly drawn episodes to "
                         "build itself from and drop them from the frontier. Answers 'how good "
                         "is LogpZO after collecting N rollouts', which is the realistic case")
    ap.add_argument("--logpzo-ckpt", default=None,
                    help="logpzo.pt from a finished run; adds a 'final' variant")
    ap.add_argument("--action-low", type=float, default=-1.0)
    ap.add_argument("--action-high", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pre_cfg = OmegaConf.load(os.path.join(args.student_dir, ".hydra", "config.yaml"))
    dino_image_keys = list(OmegaConf.select(pre_cfg, "env.dino_image_keys", default=[]) or [])
    chunk_size = OmegaConf.select(pre_cfg, "training.chunk_size", default=None)
    normalize_lowdim = bool(OmegaConf.select(pre_cfg, "training.normalize_lowdim_obs",
                                             default=False))

    student = load_actor(args.student_dir, device, args.student_checkpoint)
    student.eval()
    feat_dim = int(student.global_cond_dim)

    # The demos are the calibration set for Diff-DAgger and the training set for the other two.
    # DINOv2 is only needed to embed the demo images; the rollouts already carry their embeddings.
    dinov2_model = dinov2_processor = None
    if dino_image_keys:
        from transformers import AutoImageProcessor, AutoModel
        model_id = OmegaConf.select(pre_cfg, "model.dinov2_model", default="facebook/dinov2-base")
        dinov2_model = AutoModel.from_pretrained(model_id).to(device).eval()
        dinov2_processor = AutoImageProcessor.from_pretrained(model_id)

    sampler = (RandomSampler() if chunk_size is None else
               ChunkedSequentialSampler(chunk_size=int(chunk_size), gamma=0.99,
                                        obs_as_sequence=False))
    files = sorted(glob.glob(os.path.join(args.arrays_dir, "*.npz")), key=episode_number)
    if not files:
        raise SystemExit(f"no .npz files under {args.arrays_dir}")
    action_dim = int(np.load(files[0])["action"].shape[1])

    offline_buffer = H5ReplayBuffer(
        h5_paths=[pre_cfg.env.h5_dataset_path], sampler=sampler,
        remove_obs_keys=list(getattr(student, "remove_obs_keys", None)
                             or OmegaConf.select(pre_cfg, "env.extra_keys_to_drop", default=[])
                             or []),
        dinov2_model=dinov2_model, dinov2_processor=dinov2_processor,
        dino_embedding_keys=dino_image_keys,
        min_action=np.full(action_dim, args.action_low, np.float32),
        max_action=np.full(action_dim, args.action_high, np.float32),
        normalize_lowdim_obs=normalize_lowdim, default_intervention_label=LABEL_OFFLINE)

    algo_dict = OmegaConf.to_container(OmegaConf.select(pre_cfg, "offline_algorithm"), resolve=True)
    algo_dict["use_ema"] = False  # score the checkpoint that was rolled out, not an EMA copy
    algo_cfg = DPConfig(**algo_dict)
    algo_cfg.actor = student
    algo_cfg.buffer = offline_buffer
    algo = algo_cfg.create()

    # ---- fit every detector once ----
    # The losses Diff-DAgger assigns to the demos. Its threshold is a quantile of this
    # distribution, so the frontier has to sweep that quantile rather than pick thresholds off the
    # rollouts -- otherwise it credits the method with an operating point it cannot reach without
    # collecting rollouts first.
    calib_scores = {}
    dd_scorers, dd_meta = [], {}
    for bm in args.dd_batch_multipliers:
        s = DiffusionLossScorer(student, batch_multiplier=bm, num_per_batch=1, device=device)
        name = f"nb{s.num_train_timesteps * bm}"
        losses = s.calibrate_from_buffer(offline_buffer, num_samples=args.dd_calib_samples)
        dd_meta[name] = dict(batch_multiplier=bm, n_b=s.num_train_timesteps * bm,
                             alpha=args.dd_alpha,
                             threshold=float(quantile_threshold(losses, args.dd_alpha)),
                             calib_n=int(len(losses)), calib_mean=float(losses.mean()),
                             calib_max=float(losses.max()))
        calib_scores[f"dd_loss_{name}"] = [float(x) for x in losses]
        logger.info(f"[diffdagger:{name}] offline loss mean={losses.mean():.6f} "
                    f"max={losses.max():.6f} threshold={dd_meta[name]['threshold']:.6f}")
        dd_scorers.append((name, s))

    thrifty_acs, thrifty_meta = [], {}
    for steps in args.thrifty_train_steps:
        ac, info = build_thrifty(algo, feat_dim, action_dim, device, steps, args.thrifty_num_nets,
                                 args.thrifty_lr, args.thrifty_gamma, seed=args.seed)
        thrifty_acs.append((f"s{steps}", ac))
        thrifty_meta[f"s{steps}"] = info
    for spec in args.thrifty_ac or []:
        name, _, ckpt = spec.rpartition("=")
        name = name or "final"
        ac = build_ensemble(feat_dim, action_dim, device, num_nets=args.thrifty_num_nets)
        ac.load_state_dict(torch.load(ckpt, map_location=device))
        thrifty_acs.append((name, ac))
        thrifty_meta[name] = dict(source=ckpt)
        logger.info(f"[thrifty:{name}] loaded {ckpt}")

    # ---- encode every recorded step once ----
    # The features are needed twice, once to fit LogpZO on the rollouts and once to score every
    # detector, so keep them in memory instead of reading the .npz files twice.
    cache = []
    t0 = time.time()
    for path in files:
        z = np.load(path, allow_pickle=True)
        # Row t is the state action t was taken in; the extra last row is only a next_obs.
        obs = {"dino_embedding": torch.from_numpy(z["obs/dino_embedding"][:-1]).to(device),
               "observation/state": torch.from_numpy(z["obs/observation/state"][:-1]).to(device)}
        act = z["action"].astype(np.float32)
        T = len(act)
        with torch.no_grad():
            feat = np.concatenate([
                student.encode_obs({k: v[s:s + args.batch_size] for k, v in obs.items()})
                .float().cpu().numpy()
                for s in range(0, T, args.batch_size)])
        cache.append(dict(ei=episode_number(path), feat=feat, act=act, steps=T,
                          success=bool(z["success"])))
        if len(cache) % 50 == 0:
            logger.info(f"  encoded {len(cache)}/{len(files)} episodes "
                        f"({time.time() - t0:.0f}s)")
    n_succ = sum(c["success"] for c in cache)
    logger.info(f"encoded {len(cache)} episodes ({n_succ} success) in {time.time() - t0:.0f}s")

    # ---- LogpZO, fit the way the paper does: on successful rollouts of this same student ----
    if args.logpzo_budget > 0:
        fold_of = assign_budget(len(cache), args.logpzo_budget, args.seed)
        held_in = np.flatnonzero(fold_of == HELD_IN)
        n_ok = int(sum(cache[i]["success"] for i in held_in))
        logger.info(f"[logpzo] budget {args.logpzo_budget} episodes, {n_ok} of them successful; "
                    f"the other {len(cache) - len(held_in)} episodes are the evaluation set")
    else:
        fold_of = assign_folds(np.array([c["success"] for c in cache]), args.logpzo_folds,
                               args.seed)
    logpzo_traces, logpzo_meta = {}, {}
    for steps in args.logpzo_train_steps:
        sc, calib, info = fit_logpzo_folds(cache, fold_of, feat_dim, device, steps,
                                           args.logpzo_lr, args.seed, args.logpzo_folds,
                                           args.batch_size)
        info["calib_traces"] = calib
        logpzo_traces[f"s{steps}"] = sc
        logpzo_meta[f"s{steps}"] = info
    if args.logpzo_ckpt:
        # A model from a finished DAgger run. It never saw these episodes, so no folds are needed;
        # the band still needs successful episodes the band itself was not fit on, so half of them
        # are kept back for that.
        m = LogpZOModel(feat_dim).to(device)
        m.load_state_dict(torch.load(args.logpzo_ckpt, map_location=device))
        sc = [score_features(m, c["feat"], device, args.batch_size) for c in cache]
        succ_idx = np.flatnonzero([c["success"] for c in cache])
        np.random.default_rng(args.seed).shuffle(succ_idx)
        band_eps = succ_idx[len(succ_idx) // 2:]
        logpzo_traces["final"] = sc
        logpzo_meta["final"] = dict(fit_on=args.logpzo_ckpt, folds=0,
                                    calib_traces={"0": [sc[i].tolist() for i in band_eps]})
        logger.info(f"[logpzo:final] loaded {args.logpzo_ckpt}; band from {len(band_eps)} episodes")
        del m

    # episode_stats.json only carries the progress trace, so robometer_success_prob -- which
    # collect_signal_traces.py records -- has no source here and is left out rather than written
    # as a column of NaN.
    names = (["robometer_progress"]
             + [f"dd_loss_{n}" for n, _ in dd_scorers]
             + [f"{p}_{n}" for n, _ in thrifty_acs for p in ("novelty", "safety")]
             + [f"logpzo_{n}" for n in logpzo_traces])
    logger.info(f"signals: {names}")

    stats = json.load(open(args.stats))["episodes"] if args.stats else None

    # ---- score every recorded step ----
    episodes = []
    t0 = time.time()
    for ci, c in enumerate(cache):
        ei, T = c["ei"], c["steps"]
        traces = {n: np.full(T, np.nan, dtype=np.float64) for n in names}
        with torch.no_grad():
            for s in range(0, T, args.batch_size):
                e = min(s + args.batch_size, T)
                gc = torch.from_numpy(c["feat"][s:e]).to(device)
                a = torch.from_numpy(c["act"][s:e]).to(device).unsqueeze(1)  # executed action
                for n, sc in dd_scorers:
                    traces[f"dd_loss_{n}"][s:e] = sc._avg_loss(gc, a).cpu().numpy()
                f = c["feat"][s:e]
                a0 = c["act"][s:e]
                for n, ac in thrifty_acs:
                    traces[f"novelty_{n}"][s:e] = [float(ac.variance(f[i:i + 1]))
                                                   for i in range(len(f))]
                    traces[f"safety_{n}"][s:e] = [float(ac.safety(f[i:i + 1], a0[i:i + 1]))
                                                  for i in range(len(f))]
        for n, sc in logpzo_traces.items():
            traces[f"logpzo_{n}"] = np.asarray(sc[ci], dtype=np.float64)

        if stats is not None and ei < len(stats):
            p = np.asarray(stats[ei]["progress_trace"], dtype=np.float64).reshape(-1)
            traces["robometer_progress"][:min(T, len(p))] = p[:T]

        episodes.append(dict(episode=ei, steps=int(T), success=c["success"],
                             logpzo_fold=int(fold_of[ci]),
                             traces={n: [float(x) for x in v] for n, v in traces.items()}))
        if (len(episodes) % 20) == 0:
            logger.info(f"  {len(episodes)}/{len(files)} episodes ({time.time() - t0:.0f}s)")

    payload = dict(
        meta=dict(student_dir=args.student_dir, student_checkpoint=args.student_checkpoint,
                  task_id=int(OmegaConf.select(pre_cfg, "env.task_id", default=-1)),
                  env_name=str(OmegaConf.select(pre_cfg, "env.env_name", default="")),
                  max_episode_steps=int(OmegaConf.select(pre_cfg, "env.max_episode_steps",
                                                         default=0)),
                  seed=args.seed, score_every=1, signals=names, source="score_signals_offline",
                  arrays_dir=args.arrays_dir, stats=args.stats,
                  diffdagger=dd_meta, thrifty=thrifty_meta, logpzo=logpzo_meta,
                  calib_scores=calib_scores),
        episodes=episodes)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    n_succ = sum(e["success"] for e in episodes)
    logger.info(f"wrote {len(episodes)} episodes ({n_succ} success) in {time.time() - t0:.0f}s "
                f"-> {args.out}")


if __name__ == "__main__":
    main()
