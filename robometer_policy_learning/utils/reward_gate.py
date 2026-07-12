"""Robot-gated intervention trigger: fires on a sharp progress drop or a long plateau
in the Robometer progress signal, for reward-DAgger."""

import math
from collections import deque

from scipy.stats import spearmanr
from scipy.stats import pearsonr


def _compute_spearman(values) -> float:
    """Spearman correlation between step index and `values`."""
    n = len(values)
    if n < 2 or min(values) == max(values):
        return float("nan")
    res, _ = spearmanr(range(n), values)
    return float(res)


def _compute_pearson(values) -> float:
    """Pearson correlation between step index and `values`."""
    n = len(values)
    if n < 2 or min(values) == max(values):
        return float("nan")
    res, _ = pearsonr(range(n), values)
    return float(res)


class RewardGate:
    """Consumes the causal progress series (one value per step) and decides when the
    expert should take over. Stateful; call reset() after an intervention."""

    def __init__(self,
        short_window=5,
        drop_threshold=-0.5,
        long_window=30,
        plateau_threshold=0.05,
        method="spearman",
        smoothing=0,
        min_drop_magnitude=0.0,
    ):
        self.short_window = short_window
        self.drop_threshold = drop_threshold
        self.long_window = long_window
        self.plateau_threshold = plateau_threshold
        # For the correlation methods; requires the absolute drop to be large enough to fire
        self.min_drop_magnitude = min_drop_magnitude
        self.history = deque(maxlen=long_window)
        self.method = method
        self.smoothing = smoothing
        self._ema = None
        self._corr = {"spearman": _compute_spearman, "pearson": _compute_pearson}
        self.last_trigger = None  # "drop" | "plateau" | None

        assert long_window >= short_window, "Long window should be greater than or equal to short window"
        assert 0 <= smoothing < 1, "Smoothing should be in range [0, 1)"
        if method == "naive":
            assert 0 <= drop_threshold <= 1, "Drop threshold has to be in range [0, 1] for naive gating"
            assert 0 <= plateau_threshold <= 1, "Plateau threshold has to be in range [0, 1] for naive gating"
        elif method in ["pearson", "spearman"]:
            assert -1 <= drop_threshold <= 1, "Drop threshold has to be in range [-1, 1] for pearson/spearman gating"
            assert -1 <= plateau_threshold <= 1, "Plateau threshold has to be in range [-1, 1] for pearson/spearman gating"
        else:
            raise ValueError("Unknown gating method; expected 'naive', 'pearson', or 'spearman'")

    def update(self, progress_value: float) -> bool:
        """Ingest one causal progress value. Return True if either trigger fires (intervene now)."""
        self._ema = progress_value if self._ema is None else self.smoothing * self._ema + (1 - self.smoothing) * progress_value
        self.history.append(self._ema)

        # Only checks triggers once we have enough history
        if len(self.history) < self.short_window:
            return False
        
        if self.method == "naive":
            should_drop = self._check_drop_naive()

            if len(self.history) >= self.long_window:
                should_plateau = self._check_plateau_naive()
            else:
                should_plateau = False
        else: 
            should_drop = self._check_drop()

            if len(self.history) >= self.long_window:
                should_plateau = self._check_plateau()
            else:
                should_plateau = False

        fired = should_drop or should_plateau
        # Record which trigger fired (drop takes priority when both fire -- it's the sharper,
        # short-window signal). Read by callers after update() returns True.
        self.last_trigger = ("drop" if should_drop else "plateau") if fired else None
        return fired

    def _check_drop(self) -> bool:
        """Correlation gate: progress is trending down over the short window AND (optionally)
        the fall is large enough."""
        recent = list(self.history)[-self.short_window:]
        corr = self._corr[self.method](recent)
        if math.isnan(corr):
            return False
        if corr >= self.drop_threshold:
            return False
        if self.min_drop_magnitude > 0:
            if (max(recent) - recent[-1]) < self.min_drop_magnitude:
                return False  # trend is down, but the actual fall is noise-sized -> ignore
        return True

    def _check_plateau(self) -> bool:
        """Correlation gate: progress is not trending up over the long window."""
        recent = list(self.history)[-self.long_window:]
        corr = self._corr[self.method](recent)
        if math.isnan(corr):
            return True
        return corr < self.plateau_threshold

    def _check_drop_naive(self) -> bool:
        """Fire if progress just dropped sharply from a recent peak."""
        recent = list(self.history)[-self.short_window:]
        peak = max(recent)
        current = recent[-1]
        drop = peak - current
        fired = drop >= self.drop_threshold
        return fired
    
    def _check_plateau_naive(self) -> bool:
        """Fire if progress has stalled over a long window."""
        if len(self.history) < self.long_window:
            return False
        old_val = list(self.history)[0]
        current = list(self.history)[-1]
        improvement = current - old_val
        fired = improvement <= self.plateau_threshold
        return fired
    
    def reset(self):
        """Call after an intervention to clear history (don't re-fire on old data)"""
        self.history.clear()
        self._ema = None


if __name__ == "__main__":
    def run(label, trace, **kwargs):
        gate = RewardGate(**kwargs)
        fires = [gate.update(p) for p in trace]
        first = next((i for i, f in enumerate(fires) if f), None)
        print(f"    {label:34s} -> {'no fire' if first is None else f'fires @ step {first}'}")

    rising  = [0.20, 0.28, 0.35, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.90, 0.95] 
    regress = [0.20, 0.40, 0.60, 0.75, 0.82, 0.70, 0.55, 0.40, 0.25, 0.15, 0.10] 
    flat    = [0.50] * 11                                                         
    spike   = [0.20, 0.30, 0.40, 0.50, 0.60, 0.30, 0.70, 0.80, 0.90, 0.95, 0.98]

    corr = dict(short_window=5, long_window=10, drop_threshold=-0.5, plateau_threshold=0.3)
    for method in ("spearman", "pearson"):
        print(f"[{method}]  drop<{corr['drop_threshold']} over {corr['short_window']}, "
              f"plateau<{corr['plateau_threshold']} over {corr['long_window']}")
        run("rising  (expect: no fire)", rising,  method=method, **corr)
        run("regress (expect: drop)",    regress, method=method, **corr)
        run("flat    (expect: plateau)", flat,    method=method, **corr)

    print("[naive]  single down-spike glitch, drop>=0.15 over 5")
    naive = dict(short_window=5, long_window=10, drop_threshold=0.15, plateau_threshold=0.05, method="naive")
    run("smoothing=0.0 (expect: false fire)", spike, smoothing=0.0, **naive)
    run("smoothing=0.7 (expect: suppressed)", spike, smoothing=0.7, **naive)

