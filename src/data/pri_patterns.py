import numpy as np
from typing import List

PRI_PATTERN_TYPES = ["constant", "jittered", "staggered", "sliding", "wobulated", "dwell_switch"]
"""The six standard radar PRI modulation types from the ESM/PRI-recognition literature. The
previous generator only ever produced 'jittered' (a fixed mean + Gaussian noise) — this module
adds the other five, which is what real PRI-transform deinterleaving (CDIF/SDIF) and PRI
modulation recognition research actually discriminate between."""


def assign_pri_pattern(rng: np.random.RandomState, pattern_types: List[str] = None) -> str:
    return rng.choice(pattern_types or PRI_PATTERN_TYPES)


def sample_pri_marginal(pattern: str, base_pri: float, rng: np.random.RandomState,
                         jitter_frac: float = 0.02, num_positions: int = 3,
                         spread_frac: float = 0.3) -> float:
    """Draws ONE PRI value matching a pattern's steady-state marginal distribution — for
    datasets built from independent, non-sequential pulse samples (the classification
    dataset), where there's no pulse train to walk through, only the *shape* of the
    distribution a real receiver would eventually observe from many intercepts of that
    pattern. Constant/jittered give a single-mode distribution (a spike or a narrow Gaussian);
    staggered/dwell_switch give a multi-modal distribution (a small number of discrete
    positions — indistinguishable from each other without temporal order, which a single
    isolated PDW row can't carry, so both map to the same discrete-positions sampling here);
    sliding/wobulated give a continuous spread across a range.
    """
    if pattern == "constant":
        return base_pri
    if pattern == "jittered":
        return rng.normal(base_pri, base_pri * jitter_frac)
    if pattern in ("staggered", "dwell_switch"):
        positions = base_pri * (1 + np.linspace(-spread_frac, spread_frac, num_positions))
        chosen = rng.choice(positions)
        return rng.normal(chosen, base_pri * jitter_frac * 0.5)
    if pattern in ("sliding", "wobulated"):
        lo, hi = base_pri * (1 - spread_frac), base_pri * (1 + spread_frac)
        return rng.uniform(lo, hi)
    return base_pri


def generate_pri_sequence(pattern: str, base_pri: float, n_pulses: int, rng: np.random.RandomState,
                           jitter_frac: float = 0.02, num_positions: int = 3,
                           spread_frac: float = 0.3) -> np.ndarray:
    """Generates the actual temporal sequence of `n_pulses` successive PRI values for a real
    pulse train (the interleaved dataset, which has genuine time-of-arrival ordering) —
    unlike `sample_pri_marginal`, this can and does distinguish staggered (cycles through
    fixed positions in a repeating order) from dwell_switch (stays on one position for a long
    block before switching) from sliding (ramps continuously) from wobulated (oscillates
    sinusoidally), since temporal order is exactly what separates them.
    """
    if pattern == "constant":
        return np.full(n_pulses, base_pri)

    if pattern == "jittered":
        return rng.normal(base_pri, base_pri * jitter_frac, size=n_pulses)

    positions = base_pri * (1 + np.linspace(-spread_frac, spread_frac, num_positions))

    if pattern == "staggered":
        seq = np.resize(positions, n_pulses)
        return seq + rng.normal(0, base_pri * jitter_frac * 0.5, size=n_pulses)

    if pattern == "dwell_switch":
        dwell_len = max(5, n_pulses // num_positions)
        seq = np.resize(np.repeat(positions, dwell_len), n_pulses)
        return seq + rng.normal(0, base_pri * jitter_frac * 0.5, size=n_pulses)

    lo, hi = base_pri * (1 - spread_frac), base_pri * (1 + spread_frac)
    t = np.arange(n_pulses)

    if pattern == "sliding":
        period = max(10, n_pulses // 3)
        phase = (t % period) / period
        triangle = np.where(phase < 0.5, phase * 2, 2 - phase * 2)  # ramps up then down
        return lo + triangle * (hi - lo)

    if pattern == "wobulated":
        period = max(10, n_pulses // 4)
        sine = (np.sin(2 * np.pi * t / period) + 1) / 2
        return lo + sine * (hi - lo)

    return np.full(n_pulses, base_pri)
