import numpy as np
from typing import Optional

# Correlated multi-parameter operating modes, ordered by escalation (search -> ... ->
# illumination/lock-on) — real multifunction radars change PW/PRI/RF TOGETHER as they switch
# mode (e.g. search = long PRI + wide PW + slightly lower frequency; track/illumination =
# short PRI + narrow PW + slightly higher frequency), not independently.
OPERATING_MODES = [
    {"name": "search", "freq_mult": 0.95, "pw_mult": 1.3, "pri_mult": 1.4},
    {"name": "acquisition", "freq_mult": 1.0, "pw_mult": 1.0, "pri_mult": 1.0},
    {"name": "track", "freq_mult": 1.05, "pw_mult": 0.6, "pri_mult": 0.5},
    {"name": "illumination", "freq_mult": 1.1, "pw_mult": 0.4, "pri_mult": 0.3},
]


def transition_mode(rng: np.random.RandomState, current_idx: int,
                     escalate_p: float = 0.5, deescalate_p: float = 0.3) -> int:
    """Picks the next operating mode when a dwell block ends — biased toward escalation
    (the "war mode" search -> track -> lock-on progression), but not deterministic: an
    emitter can lose lock and de-escalate, or jump to an unrelated mode entirely (losing
    track and re-scanning), matching real, imperfect tracking behavior rather than a scripted
    one-way escalation."""
    n = len(OPERATING_MODES)
    r = rng.random()
    if r < escalate_p:
        return min(current_idx + 1, n - 1)
    if r < escalate_p + deescalate_p:
        return max(current_idx - 1, 0)
    return int(rng.randint(n))


def sample_dwell(rng: np.random.RandomState, min_dwell: int = 15, max_dwell: int = 60) -> int:
    """How many consecutive pulses an emitter stays in one mode before the next transition
    check — real radars hold a mode for many pulses, not one, so mode switching should be
    visible as a persistent, temporary state, not a per-pulse coin flip."""
    return int(rng.randint(min_dwell, max_dwell + 1))
