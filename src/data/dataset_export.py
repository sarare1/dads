import pandas as pd
import numpy as np

from src.data.pri_patterns import PRI_PATTERN_TYPES, generate_pri_sequence
from src.data.operating_modes import OPERATING_MODES, transition_mode, sample_dwell


def generate_interleaved_pdws(
    filename="synthetic_radar_pdws_v2.csv",
    simulation_time_ns=50000000,
    num_emitters=4,
    num_classes=3,
    seed=None,
    freq_range=(3000.0, 11000.0),
    pw_range=(400.0, 1300.0),      # ns
    pri_range=(100000.0, 700000.0),  # ns
    aoa_range=(0.0, 180.0),        # degrees
    rise_range=(2.0, 65.0),        # ns
):
    """Generates a receiver-realistic interleaved PDW stream from multiple emitter instances,
    sorted by time-of-arrival as a real ESM receiver would intercept them.

    `num_emitters` physical emitters get randomly generated profiles (freq/PW/PRI/AOA/rise),
    then assigned across `num_classes` distinct class labels — when num_classes < num_emitters,
    some emitters share a class (identical signal "type" but different physical units), which
    is what actually makes deinterleaving hard: telling apart two same-class emitters requires
    picking up on subtler differences than two obviously different radar types.

    `seed` controls the whole random draw — pass a specific seed to reproduce an exact
    scenario, or leave None for a fresh one each call (persisted by the caller for later
    reference). Returns the generated DataFrame's emitter roster alongside the seed actually
    used, so the caller can persist a meta.json describing this run.
    """
    if seed is None:
        seed = int(np.random.randint(1_000_000))
    rng = np.random.RandomState(seed)

    num_classes = max(1, min(num_classes, num_emitters))
    class_labels = [f"EMITTER_CLASS_{c:02d}" for c in range(num_classes)]

    # One archetype per class — same-class emitters share a near-identical RF signature
    # (frequency/PW/PRI/rise), which is what actually makes them hard to deinterleave.
    # AOA is deliberately NOT tied to the archetype: two instances of the same radar type
    # sitting in different physical locations still arrive from different angles, so AOA
    # stays independent per instance even when everything else matches.
    archetypes = [{
        'freq': rng.uniform(*freq_range),
        'pw': rng.uniform(*pw_range),
        'pri': rng.uniform(*pri_range),
        'rise': rng.uniform(*rise_range),
        # Same-class emitters share a signal TYPE, which includes its PRI modulation pattern
        # (a real radar type's PRI behavior — staggered, sliding, etc. — doesn't vary between
        # individual physical units of that type), not just similar mean parameter values.
        'pri_pattern': rng.choice(PRI_PATTERN_TYPES),
    } for _ in range(num_classes)]

    # Every class gets at least one emitter; remaining emitters assigned randomly.
    assignments = list(range(num_classes)) + [int(rng.randint(0, num_classes)) for _ in range(num_emitters - num_classes)]
    rng.shuffle(assignments)

    freq_span, pw_span, pri_span, rise_span = (freq_range[1] - freq_range[0]), (pw_range[1] - pw_range[0]), \
        (pri_range[1] - pri_range[0]), (rise_range[1] - rise_range[0])

    emitters = []
    for i in range(num_emitters):
        archetype = archetypes[assignments[i]]
        emitters.append({
            'instance_id': f'EMITTER_{i:02d}',
            'class': class_labels[assignments[i]],
            # Small (~1-2% of range) per-instance jitter around the shared class archetype —
            # distinct physical units of the same radar type, not identical clones.
            'freq': archetype['freq'] + rng.normal(0, freq_span * 0.01),
            'pw': archetype['pw'] + rng.normal(0, pw_span * 0.02),
            'pri': archetype['pri'] + rng.normal(0, pri_span * 0.02),
            'aoa': rng.uniform(*aoa_range),
            'rise': archetype['rise'] + rng.normal(0, rise_span * 0.02),
            'pri_pattern': archetype['pri_pattern'],
        })

    data = []
    for em in emitters:
        # Real operating-mode switching: the emitter dwells in one mode (search/acquisition/
        # track/illumination) for a block of pulses, generates a fresh PRI sub-sequence for
        # that block at the mode-adjusted PRI (following its assigned modulation pattern),
        # then transitions — biased toward escalation but not deterministic, the same "war
        # mode" search -> lock-on progression a real multifunction radar exhibits, visible
        # here as an actual temporal behavior rather than just per-sample diversity.
        current_time = 0
        mode_idx = 0
        while current_time < simulation_time_ns:
            mode = OPERATING_MODES[mode_idx]
            block_len = sample_dwell(rng)
            block_base_pri = em['pri'] * mode['pri_mult']
            block_pri_seq = generate_pri_sequence(em['pri_pattern'], block_base_pri, block_len, rng)

            for pri_value in block_pri_seq:
                current_time += int(max(pri_value, 1.0))
                if current_time >= simulation_time_ns:
                    break

                row = {
                    'toa_ns': current_time,
                    'pw_ns': int(rng.normal(em['pw'] * mode['pw_mult'], em['pw'] * mode['pw_mult'] * 0.05)),
                    'frequency_mhz': round(float(rng.normal(em['freq'] * mode['freq_mult'], 2.0)), 2),
                    'aoa_deg': round(float(rng.normal(em['aoa'], 0.5)), 1),
                    'rise_time_ns': round(float(rng.normal(em['rise'], em['rise'] * 0.05)), 2),
                    'emitter_class': em['class'],
                    'emitter_instance_id': em['instance_id'],  # The hidden ground truth for deinterleaving
                    'pri_pattern': em['pri_pattern'],
                    'operating_mode': mode['name'],
                }
                data.append(row)

            mode_idx = transition_mode(rng, mode_idx)

    # Simulate the receiver intercepting all signals at once by sorting by Time of Arrival
    df = pd.DataFrame(data).sort_values(by='toa_ns').reset_index(drop=True)

    # Assign sequential pulse IDs as they arrived at the receiver
    df.insert(0, 'pulse_id', range(1, 1 + len(df)))

    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} heavily interleaved pulses from {num_emitters} emitters "
          f"({num_classes} distinct classes). Saved to {filename}")

    return {
        "seed": seed,
        "num_emitters": num_emitters,
        "num_classes": num_classes,
        "emitters": [{"instance_id": e["instance_id"], "class": e["class"], "pri_pattern": e["pri_pattern"]} for e in emitters],
        "total_rows": len(df),
    }
