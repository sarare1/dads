import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from src.data.generator import normalize_pdw

REQUIRED_COLUMNS = ["toa_ns", "pw_ns", "frequency_mhz", "amplitude_dbm"]
DEFAULT_RISE_TIME_NS = 50.0  # used only if an uploaded CSV lacks this column


def find_latest_dataset(*directories: str) -> Optional[str]:
    """Finds the most recently modified .csv across the given directories —
    whichever dataset the Dataset Inspector last showed."""
    candidates = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for f in os.listdir(directory):
            if f.endswith(".csv"):
                path = os.path.join(directory, f)
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def load_replay_dataset(path: str) -> pd.DataFrame:
    """Loads and validates a PDW CSV for live replay, sorted by time-of-arrival
    (the order a real receiver would actually intercept these pulses in)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset '{os.path.basename(path)}' is missing required columns: {missing}")
    if "rise_time_ns" not in df.columns:
        df["rise_time_ns"] = DEFAULT_RISE_TIME_NS
    return df.sort_values("toa_ns").reset_index(drop=True)


def row_to_pdw_vector(row: pd.Series, prev_toa_ns: Optional[int]) -> Tuple[np.ndarray, dict]:
    """Converts one interleaved-stream row into the model's normalized 6-dim input,
    deriving PRI as the delta-ToA between consecutive received pulses — exactly what
    the spec's own PDW schema defines PRI to be, not a fabricated field."""
    pri_us = (row["toa_ns"] - prev_toa_ns) / 1000.0 if prev_toa_ns is not None else 0.0
    pw_us = row["pw_ns"] / 1000.0

    vec = normalize_pdw(
        freq=row["frequency_mhz"],
        pw=pw_us,
        pri=max(pri_us, 0.0),
        rssi=row["amplitude_dbm"],
        rise=row["rise_time_ns"],
        toa_ns=int(row["toa_ns"]),
    )

    raw_meta = {
        "carrier_freq_mhz": round(float(row["frequency_mhz"]), 2),
        "pulse_width_us": round(float(pw_us), 3),
        "pri_us": round(float(pri_us), 2),
        "rssi_dbm": round(float(row["amplitude_dbm"]), 1),
        "toa_ns": int(row["toa_ns"]),
        "rise_time_ns": round(float(row["rise_time_ns"]), 1),
        "true_label": str(row["emitter_class"]) if "emitter_class" in row else "UNLABELED",
        "class_id": None,
    }
    return vec, raw_meta


CLASSIFICATION_REQUIRED_COLUMNS = ["carrier_freq_mhz", "pulse_width_us", "pri_us", "rssi_dbm", "rise_time_ns", "true_class_id", "true_label"]


def load_classification_replay_dataset(path: str) -> pd.DataFrame:
    """Loads the Classification Dataset for live replay with VERIFIABLE ground truth — unlike
    the interleaved dataset's emitter_class labels (a different, unrelated label space from
    what the classifier was actually trained on), this file's true_label directly corresponds
    to the model's own RADAR_FAMILY_XX classes, so predictions can be checked live as they
    stream. Rows are shuffled (fresh order each replay) since the file itself is grouped by
    class in generation order — replaying it unshuffled would show the same class for a long
    consecutive stretch rather than a realistic mixed live feed."""
    df = pd.read_csv(path)
    missing = [c for c in CLASSIFICATION_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset '{os.path.basename(path)}' is missing required columns: {missing}")
    return df.sample(frac=1).reset_index(drop=True)


def classification_row_to_pdw_vector(row: pd.Series) -> Tuple[np.ndarray, dict]:
    """Converts one classification-dataset row into the model's normalized 6-dim input. This
    dataset has no real time-of-arrival ordering (each row is an independent sample, not a
    pulse train) and already carries its own per-row PRI directly, so — unlike the
    interleaved replay — no delta-ToA derivation is needed."""
    vec = normalize_pdw(
        freq=row["carrier_freq_mhz"], pw=row["pulse_width_us"], pri=row["pri_us"],
        rssi=row["rssi_dbm"], rise=row["rise_time_ns"], toa_ns=0,
    )
    raw_meta = {
        "carrier_freq_mhz": round(float(row["carrier_freq_mhz"]), 2),
        "pulse_width_us": round(float(row["pulse_width_us"]), 3),
        "pri_us": round(float(row["pri_us"]), 2),
        "rssi_dbm": round(float(row["rssi_dbm"]), 1),
        "toa_ns": None,
        "rise_time_ns": round(float(row["rise_time_ns"]), 1),
        "true_label": str(row["true_label"]),
        "class_id": int(row["true_class_id"]),
    }
    return vec, raw_meta
