"""Robot-gated intervention trigger: fires on a sharp progress drop or a long plateau
in the Robometer progress signal, for reward-DAgger."""

from collections import deque

class RewardGate:
    """Consumes the causal progress series (one value per step) and decides when the
    expert should take over. Stateful; call reset() after an intervention."""

    def __init__(self, short_window=5, drop_threshold=0.15, long_window=30, plateau_threshold=0.05):
        self.short_window = short_window
        self.drop_threshold = drop_threshold
        self.long_window = long_window
        self.plateau_threshold = plateau_threshold
        self.history = deque(maxlen=long_window)

    def update(self, progress_value: float) -> bool:
        """Ingest one causal progress value. Return True if either trigger fires (intervene now)."""
        self.history.append(progress_value)

        # Only checks triggers once we have enough history
        if len(self.history) < self.short_window:
            return False
        
        should_drop = self._check_sharp_drop()
        if len(self.history) >= self.long_window:
            should_plateau = self._check_plateau()
        else:
            should_plateau = False

        return should_drop or should_plateau

    def _check_sharp_drop(self) -> bool:
        """Fire if progress just dropped sharply from a recent peak."""
        if len(self.history) < self.short_window:
            return False
        recent = list(self.history)[-self.short_window:]
        peak = max(recent)
        current = recent[-1]
        drop = peak - current
        fired = drop >= self.drop_threshold
        return fired
    
    def _check_plateau(self) -> bool:
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


if __name__ == "__main__":
    # Test on empirical demo trace: should NOT fire (demo is working)
    demo_1 = [0.35, 0.37, 0.40, 0.43, 0.45, 0.68, 0.70, 0.72, 0.82, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95]
    gate = RewardGate(short_window=5, drop_threshold=0.15, long_window=30, plateau_threshold=0.05)
    fires_demo_1 = [gate.update(p) for p in demo_1]
    print(f"Demo 1: {fires_demo_1}")

    # Test on failure (random): flat progress, should fire on plateau
    demo_2 = [0.25, 0.28, 0.30, 0.32, 0.33, 0.32, 0.17, 0.36, 0.19]
    gate.reset()
    fires_demo_2 = [gate.update(p) for p in demo_2]
    print(f"Demo 2: {fires_demo_2}")  # last 5
