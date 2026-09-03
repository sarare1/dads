import numpy as np
from typing import Dict, Optional, Tuple

from src.data.pri_patterns import PRI_PATTERN_TYPES, sample_pri_marginal
from src.data.operating_modes import OPERATING_MODES


def build_emitter_library(
    num_classes: int,
    seed: int = 42,
    freq_range: Tuple[float, float] = (2000.0, 18000.0),
    pw_range: Tuple[float, float] = (0.5, 50.0),
    pri_range: Tuple[float, float] = (10.0, 1000.0),
    rise_range: Tuple[float, float] = (5.0, 150.0),
) -> Dict[int, Dict[str, float]]:
    """Builds base PDW centroids for `num_classes` radar families. Shared by the live
    per-pulse generator and the offline dataset builder so both draw from the same
    underlying emitter population. `seed` controls which specific archetypes get drawn —
    pass a fresh seed each run for genuinely different training populations, not the same
    20 families every time.

    Each family also gets a `pri_pattern` (one of the six standard radar PRI modulation
    types — constant/jittered/staggered/sliding/wobulated/dwell_switch), so different
    families are distinguishable not just by their mean PRI but by the *shape* of its
    distribution — a real, literature-backed ESM discriminant this generator previously had
    no concept of at all (every family was implicitly "jittered").
    """
    rng = np.random.RandomState(seed)
    library = {}
    for c in range(num_classes):
        library[c] = {
            "freq": rng.uniform(*freq_range),
            "pw": rng.uniform(*pw_range),        # Pulse Width in us
            "pri": rng.uniform(*pri_range),      # Delta ToA / PRI in us
            "rise": rng.uniform(*rise_range),    # Rise time in ns
            "pri_pattern": rng.choice(PRI_PATTERN_TYPES),
        }
    return library


def sample_known_pulse(params: Dict[str, float], rng: np.random.RandomState = None,
                        mode: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Draws one operational-mode sample around a class centroid — freq/pw/pri move together
    per a correlated operating mode, and the final PRI value additionally follows the class's
    assigned PRI modulation pattern rather than always being simple Gaussian jitter around one
    fixed mean.

    `mode`, when omitted, is drawn uniformly at random each call (the classification dataset's
    behavior — i.i.d. per-sample mode diversity, no persistence). Callers that need a
    persistent, evolving mode across many consecutive calls (a real emitter dwelling in one
    mode before switching — the interleaved dataset and the Live Feed) instead track their own
    current mode (see `src.data.operating_modes.transition_mode`/`sample_dwell`) and pass it
    in explicitly here.
    """
    rng = rng or np.random
    if mode is None:
        mode = OPERATING_MODES[rng.randint(len(OPERATING_MODES))]
    pri_pattern = params.get("pri_pattern", "jittered")
    base_pri = params["pri"] * mode["pri_mult"]
    return {
        "freq": params["freq"] * mode["freq_mult"] + rng.normal(0, 15.0),
        "pw": params["pw"] * mode["pw_mult"] + rng.normal(0, 0.2),
        "pri": sample_pri_marginal(pri_pattern, base_pri, rng),
        "rise": params["rise"] + rng.normal(0, 1.0),
        "mode": mode["name"],
    }


def sample_ood_pulse(rng: np.random.RandomState = None) -> Dict[str, float]:
    """Draws one uncataloged agile-threat sample. Deliberately overlaps the low end of known
    parameter ranges rather than occupying an entirely disjoint band in every dimension —
    real novel emitters plausibly share some characteristics with cataloged ones (e.g. a
    similar frequency but an unusual PRI/PW combination), so an open-set task where "unknown"
    is trivially separable on frequency alone understates real difficulty."""
    rng = rng or np.random
    return {
        "freq": rng.uniform(4000.0, 30000.0),  # overlaps most of the known 2000-18000 band, extends beyond it
        "pw": rng.uniform(0.1, 5.0),
        "pri": rng.uniform(2.0, 60.0),
        "rise": rng.uniform(1.0, 20.0),
    }


def frequency_to_band(freq_mhz: float) -> str:
    """Maps a carrier frequency to its standard radar frequency band letter designation."""
    if freq_mhz < 2000: return "L-BAND"
    if freq_mhz < 4000: return "S-BAND"
    if freq_mhz < 8000: return "C-BAND"
    if freq_mhz < 12000: return "X-BAND"
    if freq_mhz < 18000: return "KU-BAND"
    if freq_mhz < 27000: return "K-BAND"
    return "KA-BAND"


def normalize_pdw(freq: float, pw: float, pri: float, rise: float,
                   toa_ns: int = 0) -> np.ndarray:
    """Maps raw physical PDW units to the model's normalized 6-dim input space.

    RSSI/amplitude is deliberately excluded entirely — not just as a model input, but from
    the simulator altogether: unlike frequency/PW/PRI/rise-time (intrinsic to the emitter's
    own waveform), RSSI is dominated by range, antenna pointing, and receiver AGC
    calibration — genuinely hard to derive consistently on real intercept hardware, and not
    an emitter fingerprint in the first place.

    Duty cycle (PW/PRI) fills the freed input slot instead — a direct interaction term
    between two already-present inputs that a single Linear classifier head can't otherwise
    construct on its own, and one of the strongest Operating-Mode signals in this dataset
    (search vs. track vs. illumination)."""
    duty_cycle = pw / pri if pri > 0 else 0.0
    return np.array([
        freq / 25000.0,
        pw / 100.0,
        pri / 2000.0,
        duty_cycle,
        (toa_ns / 1e9),
        rise / 200.0
    ], dtype=np.float32)
