"""The Diff-DAgger intervention gate, ported from the reference implementation.

One comparison arm for the Robometer progress gate; the others live in ``thrifty_gate.py`` and
``human_gate.py``. It replaces *only* the gating rule -- same DP student, same pi0 expert, same
DAgger loop -- so any difference in the resulting policy is attributable to the gate.

Diff-DAgger (Lee, Kang & Kuo, ICRA 2025; arXiv:2410.14868)
    Uncertainty = the diffusion training loss evaluated at the policy's own action. If the model
    denoises its own chosen action badly, the state is out of distribution.

    Algorithm 1 of the paper, per trial: train the policy on D_exp; set the threshold from D_exp
    and alpha; roll out, and hand control to the expert once CDF(loss) > alpha for the last K
    timesteps; record ONLY the expert's (o, a) pairs back into D_exp.
    Paper hyperparameters (Table IV): alpha = 0.99, K = 2, N_b = 512, where N_b is the batch used
    to estimate the expected loss of Eq. 2 over noise and diffusion timestep.

    Reference implementation: https://github.com/sean1295/DiffDAgger
        diffdagger/agents/diffusion_policy.py :: get_avg_diffusion_loss_ndata   (the score)
        diffdagger/agents/diffusion_policy.py :: get_stats_from_dataset         (calibration)
        diffdagger/agents/diffusion_policy.py :: DiffDAggerPolicy.get_action    (the trigger)
        diffdagger/util/cdf.py                :: CDF.get_quantile               (the threshold)
    The CDF is used verbatim from ``baselines/vendored/diffdagger_cdf.py``; the rest is
    reimplemented against this repo's DiffusionActor, which exposes the same three primitives the
    reference needs (encode obs -> conditioning, predict noise, sample an action chunk).

    Deliberate deviations, all forced by running the gate inside *this* DAgger loop:
      * The student is fine-tuned continuously; the reference re-instantiates the policy and
        trains from scratch each round, then cycles through checkpoints saved at 80-100% of
        training. Ours is the recipe every other arm in this study uses, so it is what keeps the
        comparison about the gate.
      * Calibration draws ``dd_calib_samples`` datapoints from the live training buffer (offline
        demos mixed with collected corrections at ``offline_sample_ratio``). The reference sweeps
        its whole aggregated D_exp 16x -- infeasible here, where a single LIBERO task has ~20k
        datapoints and each costs a 500-row forward.
      * The loss is evaluated with EMA weights. So is the reference's, indirectly:
        ``DiffDAggerTrainingEndCallback`` copies the EMA into the live model before recalibrating,
        and ``normalize_obs`` / ``get_naction`` both run off ``ema_model``.
      * No timeout-forced expert intervention (the reference hands control to the expert at the
        time limit even when the gate stayed quiet, and lets it run past the limit). Our env
        horizon is fixed, so a timed-out episode is simply discarded by ``require_success``.
      * No ``wait_timestep`` settling pause before the expert engages, and no image augmentation
        during calibration -- the student consumes frozen DINOv2 embeddings, not raw pixels.

Gates here duck-type ``RewardGate`` (update/reset/last_trigger/history/short_window) so the
rollout worker can hold any of them without branching.
"""

import os
import sys
from collections import deque
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

_VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "baselines")
if _VENDORED not in sys.path:
    sys.path.insert(0, _VENDORED)
from vendored.diffdagger_cdf import CDF  # noqa: E402  (authors' verbatim CDF)


def _as_action_chunk(actions: torch.Tensor) -> torch.Tensor:
    """(B, A) -> (B, 1, A); mirrors ``DP._prepare_actions`` so calibration sees the same shape the
    training step does, without needing a DP instance (the standalone worker has no algo)."""
    if actions.dim() == 2:
        return actions.unsqueeze(1)
    if actions.dim() != 3:
        raise ValueError(f"Unexpected action shape for the diffusion loss: {tuple(actions.shape)}")
    return actions


class DiffusionLossScorer:
    """Diff-DAgger's per-step uncertainty: E_{eps,t}[ ||target - f_theta(o, a_noised, t)||^2 ].

    Matches the reference: it tiles the already-encoded conditioning vector across the timestep
    grid (``nobs.repeat(T * batch_multiplier, 1, 1)`` feeding ``compute_loss``, which passes
    ``obs_batch`` straight to the policy head without re-encoding), so the observation is encoded
    once per scored step in both implementations.

    Also following the reference, timesteps are *enumerated* (arange % T) rather than sampled,
    which removes timestep sampling noise from the score; only the noise draw is random, averaged
    over ``num_per_batch`` repeats.

    The paper's N_b (samples used to estimate Eq. 2) is ``num_train_timesteps * batch_multiplier
    * num_per_batch``. Table IV sets N_b = 512 with T = 16, i.e. batch_multiplier = 32; our
    scheduler has T = 100, so batch_multiplier = 5 gives the same effective sample count.
    """

    def __init__(
        self,
        actor,
        batch_multiplier: int = 1,
        num_per_batch: int = 1,
        device: Optional[torch.device] = None,
    ):
        if not all(hasattr(actor, a) for a in ("scheduler", "predict_noise", "encode_obs")):
            raise TypeError(
                "Diff-DAgger gates on the student's OWN denoising loss, so the student must be a "
                f"DiffusionActor; got {type(actor).__name__}. Use a DP student for this arm.")
        self.actor = actor
        self.batch_multiplier = int(batch_multiplier)
        self.num_per_batch = int(num_per_batch)
        self.device = device or next(actor.parameters()).device
        self.num_train_timesteps = int(actor.scheduler.config.num_train_timesteps)
        self.prediction_type = getattr(actor.scheduler.config, "prediction_type", "epsilon")

    @torch.no_grad()
    def _avg_loss(self, global_cond: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-item diffusion loss averaged over every timestep and ``num_per_batch`` noise draws.

        ``global_cond`` is (k, D) and ``actions`` (k, H, A); returns (k,). Each item is paired with
        the full enumerated timestep grid, so the returned statistic is an *average*, not a single
        noisy draw. Calibration and deployment must both use this, otherwise the quantile is taken
        over a high-variance single-draw distribution while the gate compares against a
        low-variance average -- which puts the threshold far above anything ever observed.
        """
        k = global_cond.shape[0]
        reps = self.num_train_timesteps * self.batch_multiplier
        gc = global_cond.repeat_interleave(reps, dim=0)                       # (k*reps, D)
        act = actions.to(self.device).repeat_interleave(reps, dim=0).contiguous()
        timesteps = (torch.arange(reps, device=self.device).repeat(k)
                     % self.num_train_timesteps).long()

        total = torch.zeros(k, device=self.device, dtype=torch.float32)
        for _ in range(self.num_per_batch):
            noise = torch.randn_like(act)
            noisy = self.actor.scheduler.add_noise(act, noise, timesteps)
            pred = self.actor.predict_noise(noisy, timesteps, gc)
            target = self._target(act, noise, timesteps)
            per = F.mse_loss(pred, target, reduction="none")                  # (k*reps, H, A)
            total += per.reshape(k, reps, -1).mean(dim=(1, 2))
        return total / self.num_per_batch

    def _target(self, act: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        """Regression target for the denoising loss, matching the reference's ``compute_loss``
        (and this repo's ``DP.train_step``, which handles only epsilon/sample)."""
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "v_prediction":
            return self.actor.scheduler.get_velocity(act, noise, timesteps)
        return act  # "sample"

    @torch.no_grad()
    def score(
        self,
        obs: Union[dict, torch.Tensor],
        action_norm: Optional[torch.Tensor] = None,
    ) -> float:
        """Expected diffusion loss at ``obs``, evaluated at the policy's *own* action.

        ``action_norm`` is that action in the normalised [-1, 1] space (i.e. straight from
        ``sample_actions``); if absent it is sampled here. The reference does the same -- its
        ``get_action(dagger=True)`` draws a fresh action chunk on every scored step and scores
        that -- so the cost of a scored step is one reverse-diffusion pass plus one batched
        forward over the timestep grid.
        """
        global_cond = self.actor.encode_obs(obs)                     # (B, D), B == 1 at rollout
        if global_cond.shape[0] != 1:
            raise ValueError(f"DiffusionLossScorer expects batch size 1, got {global_cond.shape[0]}")

        if action_norm is None:
            action_norm = self.actor.sample_actions(obs)
        if action_norm.dim() == 2:                                   # (B, A) -> (B, 1, A)
            action_norm = action_norm.unsqueeze(1)
        return float(self._avg_loss(global_cond, action_norm)[0].item())

    @torch.no_grad()
    def calibrate_from_buffer(self, buffer, batch_size: int = 64, num_samples: int = 1024,
                              max_rows: int = 8192) -> np.ndarray:
        """Averaged diffusion losses over a training buffer, one value per datapoint.

        This is the reference's ``get_stats_from_dataset``: score every datapoint of the training
        set at its *demonstrated* action (not the policy's own), and keep the resulting
        distribution. Diff-DAgger recalibrates it every DAgger iteration and takes the
        alpha-quantile -- as the policy improves its training losses fall, so a threshold fixed at
        iteration 0 drifts.

        Crucially this uses ``_avg_loss`` -- the same statistic the gate compares against at
        deployment. Calibration and deployment must both use the average over the enumerated
        timestep grid, otherwise the quantile is taken over a high-variance single-draw
        distribution while the gate compares against a low-variance average, which puts the
        threshold far above anything ever observed.

        ``max_rows`` caps the expanded batch (each datapoint becomes ``T * batch_multiplier``
        rows), so raising ``batch_multiplier`` costs time rather than memory.
        """
        reps = self.num_train_timesteps * self.batch_multiplier
        chunk = max(1, int(max_rows) // reps)
        losses = []
        while len(losses) < num_samples:
            batch = buffer.sample(int(batch_size), device=self.device)
            if not batch or len(batch.get("action", [])) == 0:
                break
            actions = _as_action_chunk(torch.as_tensor(batch["action"]).to(self.device)).float()
            global_cond = self.actor.encode_obs(batch["obs"])

            for i in range(0, global_cond.shape[0], chunk):
                if len(losses) >= num_samples:
                    break
                losses.extend(self._avg_loss(global_cond[i:i + chunk],
                                             actions[i:i + chunk]).cpu().numpy().tolist())
        return np.asarray(losses[:num_samples], dtype=np.float64)

    def calibrate_from_algo(self, algo, num_samples: int = 1024, max_rows: int = 8192) -> np.ndarray:
        """``calibrate_from_buffer`` against the algo's own training buffer, so the calibration
        distribution is exactly the one the training step draws from (offline demos mixed with the
        collected corrections, at the configured ratio)."""
        return self.calibrate_from_buffer(algo.buffer, algo.batch_size, num_samples, max_rows)


class DiffDaggerScorer:
    """``RobometerScorer``-compatible facade so the rollout worker can hold either one.

    Same surface -- ``reset(task)`` / ``observe(obs)`` / ``score() -> (value, aux)`` -- but the
    value is the student's own diffusion loss rather than a Robometer progress estimate. Note the
    direction flips: Robometer progress is *higher is better*, diffusion loss is *higher is worse*.
    Each gate is paired with its own scorer, so nothing downstream needs to know.
    """

    def __init__(self, actor, remove_obs_keys=None, device=None, **scorer_kwargs):
        self.actor = actor
        self.remove_obs_keys = list(remove_obs_keys or [])
        self.device = device or next(actor.parameters()).device
        self.inner = DiffusionLossScorer(actor, device=self.device, **scorer_kwargs)
        self.task = ""
        self.episode_id = 0
        self._last_obs = None

    def reset(self, task: str = ""):
        self.task = str(task)
        self.episode_id += 1
        self._last_obs = None

    def observe(self, obs, frame_key: Optional[str] = None):
        """Store the raw env obs; the loss is computed lazily in ``score()`` (which is only called
        every ``score_every`` steps, so we avoid encoding on unscored steps). ``frame_key`` is
        accepted and ignored -- this scorer consumes the whole obs, not one camera."""
        self._last_obs = obs

    def score(self):
        from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

        if self._last_obs is None:
            return 0.0, 0.0
        prepped = {k: v for k, v in self._last_obs.items() if k not in self.remove_obs_keys}
        actor_obs = move_to_device(convert_to_tensor(prepped), self.device)
        return float(self.inner.score(actor_obs)), 0.0


def _get_quantile(cdf: CDF, alpha: float) -> float:
    """``CDF.get_quantile`` with one guard: it indexes with ``int(n * q)``, which walks off the end
    at q = 1.0. Clamping q to (n-1)/n is the only deviation and never bites for alpha < 1."""
    n = len(cdf.sorted_data)
    return float(cdf.get_quantile(min(float(alpha), (n - 1) / n)))


def quantile_threshold(losses: np.ndarray, alpha: float) -> float:
    """alpha-quantile of the training-loss distribution (Diff-DAgger's ``CDF.get_quantile``)."""
    if len(losses) == 0:
        raise ValueError("no calibration losses collected")
    return _get_quantile(CDF(np.asarray(losses, dtype=np.float64)), alpha)


class QuantileGate:
    """Diff-DAgger's trigger: fire once ``patience`` of the last ``patience_window`` steps exceed
    the calibrated threshold.

    Algorithm 1 line 15 is ``if CDF(loss) > alpha for last K timesteps`` -- K *consecutive*
    violations, K = 2 in Table IV. The released code generalises this to M-of-N via
    ``deque(maxlen=patience_window)`` and ``sum(deque) >= patience``. We follow the code, and
    ``patience_window == patience`` recovers the paper's consecutive rule (the default).
    Comparing the loss against the alpha-quantile is identical to the paper's ``CDF(loss) > alpha``.

    The gate owns the training-loss CDF, not just its alpha-quantile, so ``recalibrate`` is the
    single place the threshold is set and every score can also be reported as a CDF value (the
    reference's ``get_cdf_value``, which it overlays on rollout video).
    """

    def __init__(
        self,
        threshold: float = float("inf"),
        patience: int = 1,
        patience_window: Optional[int] = None,
        alpha: float = 0.99,
        **_ignored: Any,
    ):
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.cdf: Optional[CDF] = None
        self.patience = int(patience)
        self.patience_window = int(patience_window if patience_window is not None else patience)
        if self.patience_window < self.patience:
            raise ValueError("patience_window must be >= patience")
        self.deque = deque([], maxlen=self.patience_window)
        self.history = deque(maxlen=self.patience_window)   # RewardGate-compatible surface
        self.short_window = self.patience_window
        self.last_trigger = None
        self.last_cdf: Optional[float] = None

    def recalibrate(self, losses: np.ndarray, alpha: Optional[float] = None) -> Dict[str, Any]:
        """Refit the training-loss CDF and move the threshold to its alpha-quantile.

        Called once before collection starts and again after every retrain, matching
        ``DiffDAggerTrainingEndCallback``, which recalibrates at the end of each round of
        training against the dataset the policy was just fit to.
        """
        if alpha is not None:
            self.alpha = float(alpha)
        losses = np.asarray(losses, dtype=np.float64)
        if losses.size == 0:
            raise ValueError("no calibration losses collected")
        self.cdf = CDF(losses)
        self.threshold = _get_quantile(self.cdf, self.alpha)
        return dict(threshold=self.threshold, alpha=self.alpha, n=int(losses.size),
                    loss_mean=float(losses.mean()), loss_max=float(losses.max()))

    def update(self, value: float) -> bool:
        value = float(value)
        self.history.append(value)
        self.last_cdf = float(self.cdf(value)) if self.cdf is not None else None
        self.deque.append(value > self.threshold)
        fired = sum(self.deque) >= self.patience
        self.last_trigger = "diffusion_loss" if fired else None
        return fired

    def reset(self):
        # `last_trigger` deliberately survives reset, matching RewardGate: the worker resets the
        # gate on takeover and only *then* logs which trigger fired. Clearing the deque per
        # episode is the reference's `policy.reset()`.
        self.deque.clear()
        self.history.clear()

    def describe(self) -> Dict[str, Any]:
        return dict(gate="diffdagger", threshold=self.threshold, alpha=self.alpha,
                    calibrated=self.cdf is not None,
                    patience=self.patience, patience_window=self.patience_window)

    def plot_trace(self, stats: Dict[str, Any], save_path: str, title: Optional[str] = None):
        """Per-episode diagnostic: the student's own diffusion loss against the calibrated
        threshold. The Diff-DAgger arm's analogue of ``plot_progress_trace`` (and of the
        loss/CDF overlay the reference draws on its evaluation video).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        losses = stats["progress_trace"]   # on-disk key is signal-agnostic; here it is the loss
        steps = len(losses)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(range(steps), losses, color="#1f77b4", lw=1.5, zorder=2, label="diffusion loss")
        ax.axhline(self.threshold, color="#2ca02c", lw=1.2, ls="--", zorder=3,
                   label=f"threshold (alpha={self.alpha:g})")
        for f in stats["gate_fires"]:
            ax.axvspan(f, steps - 1, color="#d62728", alpha=0.08, zorder=1)
            ax.axvline(f, color="#d62728", lw=1.5, zorder=4)
            ax.annotate(f"takeover\n@{f}", xy=(f, max(losses, default=1.0)), xytext=(2, -2),
                        textcoords="offset points", ha="left", va="top", fontsize=7,
                        color="#d62728")

        ax.set_xlabel("environment step")
        ax.set_ylabel("diffusion loss")
        ax.set_title(title or f"success={stats['success']}  "
                              f"interventions={stats['num_interventions']}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
