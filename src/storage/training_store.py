import os
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/training_history.db")
IST = timezone(timedelta(hours=5, minutes=30))

_COLUMNS = [
    "backbone_type", "distance_metric", "inference_engine",
    "hnsw_m", "hnsw_ef_construct", "quantization",
    "num_classes", "num_holdout_classes", "samples_per_class", "noise_pct", "epochs",
    "rpl_loss_weight", "rpl_adversarial",
    "learning_rate", "triplet_margin", "classifier_confidence_threshold", "ood_distance_threshold",
    "closed_set_accuracy", "open_set_auroc", "open_set_auroc_rpl", "final_loss",
    "training_duration_seconds", "loss_curve_json",
    "dataset_seed",
    "freq_min_mhz", "freq_max_mhz", "pw_min_us", "pw_max_us",
    "pri_min_us", "pri_max_us", "rssi_min_dbm", "rssi_max_dbm",  # rssi_* unused since RSSI's removal — kept for old rows, always NULL going forward
    "rise_min_ns", "rise_max_ns",
]

_TEST_COLUMNS = [
    "backbone_type", "distance_metric", "inference_engine",
    "hnsw_m", "hnsw_ef_construct", "quantization",
    "dataset_source", "num_known_classes", "num_holdout_classes",
    "closed_set_accuracy", "open_set_auroc",
    "num_known_pulses", "num_openset_pulses", "total_pulses", "dataset_seed",
]


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                {", ".join(f"{c} TEXT" for c in _COLUMNS)}
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                {", ".join(f"{c} TEXT" for c in _TEST_COLUMNS)}
            )
        """)
        conn.commit()

        # Safe migration for DB files created before these columns existed — a no-op on a
        # freshly created table (CREATE TABLE above already included them), harmless if run
        # more than once (duplicate-column errors are swallowed).
        for col in _COLUMNS:
            try:
                conn.execute(f"ALTER TABLE training_runs ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass
        for col in _TEST_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE test_runs ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()


def _to_ist_string(timestamp_utc: str) -> str:
    dt = datetime.fromisoformat(timestamp_utc).replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def insert_run(record: Dict[str, Any]) -> int:
    """Inserts one completed training run. `record` must contain every key in _COLUMNS;
    numeric/list values are stored as-is (sqlite3 coerces), loss_curve is JSON-encoded."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = dict(record)
        row["loss_curve_json"] = json.dumps(row.pop("loss_curve", row.get("loss_curve_json", [])))
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        columns = ["timestamp_utc"] + _COLUMNS
        values = [timestamp_utc] + [row.get(c) for c in _COLUMNS]
        placeholders = ", ".join("?" for _ in columns)
        cursor = conn.execute(
            f"INSERT INTO training_runs ({', '.join(columns)}) VALUES ({placeholders})", values
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["timestamp_ist"] = _to_ist_string(d["timestamp_utc"])
    d["loss_curve"] = json.loads(d.pop("loss_curve_json") or "[]")
    for key in ("closed_set_accuracy", "open_set_auroc", "open_set_auroc_rpl", "final_loss", "learning_rate",
                "triplet_margin", "classifier_confidence_threshold", "ood_distance_threshold",
                "training_duration_seconds", "noise_pct", "rpl_loss_weight",
                "freq_min_mhz", "freq_max_mhz", "pw_min_us", "pw_max_us",
                "pri_min_us", "pri_max_us", "rssi_min_dbm", "rssi_max_dbm",
                "rise_min_ns", "rise_max_ns"):
        if d.get(key) is not None:
            d[key] = float(d[key])
    for key in ("hnsw_m", "hnsw_ef_construct", "num_classes", "num_holdout_classes",
                "samples_per_class", "epochs", "dataset_seed"):
        if d.get(key) is not None:
            d[key] = int(float(d[key]))
    if d.get("rpl_adversarial") is not None:
        # sqlite3 stores a Python bool as integer 1/0 (bool is an int subclass), and the TEXT
        # column affinity here can coerce that to the string "1"/"0" on read — so both integer
        # and string truthy forms need handling, not just a literal "true"/"false" string.
        d["rpl_adversarial"] = str(d["rpl_adversarial"]).strip().lower() in ("1", "true", "yes")
    return d


def get_latest_run() -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def get_all_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def _test_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["timestamp_ist"] = _to_ist_string(d["timestamp_utc"])
    for key in ("closed_set_accuracy", "open_set_auroc"):
        if d.get(key) is not None:
            d[key] = float(d[key])
    for key in ("hnsw_m", "hnsw_ef_construct", "num_known_classes", "num_holdout_classes",
                "num_known_pulses", "num_openset_pulses", "total_pulses", "dataset_seed"):
        if d.get(key) is not None:
            d[key] = int(float(d[key]))
    return d


def insert_test_run(record: Dict[str, Any]) -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        columns = ["timestamp_utc"] + _TEST_COLUMNS
        values = [timestamp_utc] + [record.get(c) for c in _TEST_COLUMNS]
        placeholders = ", ".join("?" for _ in columns)
        cursor = conn.execute(
            f"INSERT INTO test_runs ({', '.join(columns)}) VALUES ({placeholders})", values
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_test_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM test_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_test_row_to_dict(r) for r in rows]
    finally:
        conn.close()
