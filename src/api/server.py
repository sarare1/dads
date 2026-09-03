import os
import torch
import numpy as np
import pandas as pd
import yaml
import onnxruntime as ort
from typing import Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
import asyncio
import json
import time
import hmac
import hashlib
from datetime import datetime, timezone

from src.data.generator import frequency_to_band, normalize_pdw, build_emitter_library, sample_known_pulse, sample_ood_pulse
from src.data.operating_modes import OPERATING_MODES, transition_mode, sample_dwell
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.data.dataset_export import generate_interleaved_pdws
from src.data.replay import (
    find_latest_dataset, load_replay_dataset, row_to_pdw_vector,
    load_classification_replay_dataset, classification_row_to_pdw_vector,
)
from src.models.backbones import DualHeadEWModel
from src.vector_db.qdrant_engine import QdrantEdgeEngine
from src.training.trainer import (
    train_model, evaluate_model, build_distance_fn,
    build_test_dataset_rows, load_classification_csv_for_training
)
from src.processing.deinterleaver import deinterleave
from src.processing.deep_deinterleaver import deep_metric_deinterleave
from src.storage.training_store import (
    init_db, insert_run, get_latest_run, get_all_runs, insert_test_run, get_all_test_runs
)

app = FastAPI(title="DADS - PhD Research Project")

UI_DIR = os.path.join(os.path.dirname(__file__), "../../ui")
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

# --- Auth gate -------------------------------------------------------------
# Deliberately simple, matching the single hardcoded shared credential this app was asked to
# use: no user table, no password hashing library, just one fixed User ID/password pair and a
# signed cookie proving the browser already presented them once. Not meant as a real
# multi-user auth system — it's a look-but-don't-enter gate for a shared demo deployment.
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "sararedigital"
SESSION_COOKIE_NAME = "dads_auth"
_SESSION_SECRET = "dads-phd-research-hardcoded-session-secret"  # static on purpose — see note above
_SESSION_TOKEN = hmac.new(_SESSION_SECRET.encode(), b"authenticated", hashlib.sha256).hexdigest()
_PUBLIC_PATHS = {"/login", "/api/login"}


def _is_authenticated(cookies: dict) -> bool:
    token = cookies.get(SESSION_COOKIE_NAME)
    return token is not None and hmac.compare_digest(token, _SESSION_TOKEN)


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    if path in _PUBLIC_PATHS or path.startswith("/static/"):
        return await call_next(request)
    if not _is_authenticated(request.cookies):
        if path.startswith("/api/"):
            return JSONResponse({"status": "error", "message": "Not authenticated — please sign in."}, status_code=401)
        return RedirectResponse(url="/login")
    return await call_next(request)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")  # interleaved-format uploads only (Dataset page -> Live Simulation/deinterleaving)
TEST_UPLOAD_DIR = os.path.join(DATA_DIR, "classification_uploads")  # classification-format uploads only (Testing page)
MODELS_DIR = os.path.join(DATA_DIR, "models")
CLASSIFICATION_DATASET_DIR = os.path.join(DATA_DIR, "classification_datasets")
CLASSIFICATION_DATASET_CSV = os.path.join(CLASSIFICATION_DATASET_DIR, "classification_dataset.csv")
CLASSIFICATION_DATASET_META = os.path.join(CLASSIFICATION_DATASET_DIR, "classification_dataset_meta.json")
AUTO_TEST_SAMPLE_DIR = os.path.join(DATA_DIR, "auto_test_samples")  # Testing page's "auto-generate fresh sample" — saved so it's inspectable, not silently discarded
AUTO_TEST_SAMPLE_CSV = os.path.join(AUTO_TEST_SAMPLE_DIR, "last_auto_test_sample.csv")
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEST_UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_DATASET_DIR, exist_ok=True)
os.makedirs(AUTO_TEST_SAMPLE_DIR, exist_ok=True)
init_db()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

current_config = load_config()
current_config["model"].setdefault("inference_engine", "pytorch")
current_config.setdefault("vector_db", {})
current_config["vector_db"].setdefault("hnsw_m", 16)
current_config["vector_db"].setdefault("hnsw_ef_construct", 100)
current_config["vector_db"].setdefault("quantization", "scalar")
current_config["hyperparameters"].setdefault("rpl_loss_weight", 0.1)
current_config["hyperparameters"].setdefault("rpl_adversarial", False)

MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, "current_model_weights.pt")

# Resuming a previous session: if a real training run happened before (possibly in an
# earlier process), restore the config it actually used so the reconstructed model
# architecture and the Qdrant collection's distance metric agree with what's already
# persisted on disk, instead of silently falling back to config.yaml's defaults.
_last_run = get_latest_run()
if _last_run is not None:
    current_config["model"]["backbone_type"] = _last_run["backbone_type"]
    current_config["model"]["inference_engine"] = _last_run["inference_engine"]
    current_config["model"]["num_classes"] = _last_run["num_classes"]
    current_config["hyperparameters"]["distance_metric"] = _last_run["distance_metric"]
    current_config["vector_db"]["hnsw_m"] = _last_run["hnsw_m"]
    current_config["vector_db"]["hnsw_ef_construct"] = _last_run["hnsw_ef_construct"]
    current_config["vector_db"]["quantization"] = _last_run["quantization"]
    if _last_run.get("rpl_loss_weight") is not None:
        current_config["hyperparameters"]["rpl_loss_weight"] = _last_run["rpl_loss_weight"]
    if _last_run.get("rpl_adversarial") is not None:
        current_config["hyperparameters"]["rpl_adversarial"] = _last_run["rpl_adversarial"]

vector_db = QdrantEdgeEngine(
    num_classes=current_config["model"]["num_classes"],
    embed_dim=current_config["model"]["embedding_dim"],
    metric=current_config["hyperparameters"]["distance_metric"],
    hnsw_m=current_config["vector_db"]["hnsw_m"],
    hnsw_ef_construct=current_config["vector_db"]["hnsw_ef_construct"],
    quantization=current_config["vector_db"]["quantization"],
)
prototypes_trained = vector_db.has_trained_prototypes()

model = DualHeadEWModel(
    backbone_type=current_config["model"]["backbone_type"],
    input_dim=current_config["model"]["input_dim"],
    embed_dim=current_config["model"]["embedding_dim"],
    num_classes=current_config["model"]["num_classes"]
)
if prototypes_trained and os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
model.eval()

onnx_session = None
_onnx_path = os.path.join(MODELS_DIR, "current_model.onnx")
if prototypes_trained and os.path.exists(_onnx_path):
    try:
        onnx_session = ort.InferenceSession(_onnx_path, providers=["CPUExecutionProvider"])
    except Exception:
        onnx_session = None

onnx_fallback_warned = False
training_status = {"is_training": False, "progress": None, "last_metrics": None, "error": None}
_training_task = None


def export_onnx(target_model: DualHeadEWModel):
    """Exports the current model to ONNX for the 'ONNX Runtime CPU' inference engine option."""
    global onnx_session
    target_model.eval()
    dummy = torch.randn(1, current_config["model"]["input_dim"])
    onnx_path = os.path.join(MODELS_DIR, "current_model.onnx")
    torch.onnx.export(
        target_model, dummy, onnx_path,
        input_names=["pdw"], output_names=["probs", "embeddings"],
        dynamic_axes={"pdw": {0: "batch"}, "probs": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def run_inference_pipeline(pdw_vec: np.ndarray) -> dict:
    """Runs one normalized PDW vector through the current model + Qdrant prototypes + fusion
    logic. Shared by /ws/telemetry (dataset replay) and /api/infer (manual single-shot input)
    so both paths compute results identically — no duplicated logic between the two."""
    global onnx_fallback_warned
    engine = current_config["model"].get("inference_engine", "pytorch")

    if engine == "onnx" and onnx_session is not None:
        ort_inputs = {"pdw": pdw_vec.reshape(1, -1).astype(np.float32)}
        probs_out, embed_out = onnx_session.run(None, ort_inputs)
        probs_np = probs_out.squeeze(0)
        embed_np = embed_out.squeeze(0)
    else:
        if engine == "onnx" and onnx_session is None and not onnx_fallback_warned:
            print("WARNING: ONNX inference engine selected but no trained export exists yet. "
                  "Falling back to PyTorch until a training run completes.")
            onnx_fallback_warned = True
        with torch.no_grad():
            input_tensor = torch.from_numpy(pdw_vec).unsqueeze(0)
            probs, embeddings = model(input_tensor)
            probs_np = probs.squeeze(0).numpy()
            embed_np = embeddings.squeeze(0).numpy()

    top_class_id = int(np.argmax(probs_np))
    top_confidence = float(probs_np[top_class_id])

    nearest_label, metric_dist, matched_meta = vector_db.search_nearest(embed_np)

    cls_thresh = current_config["hyperparameters"]["classifier_confidence_threshold"]
    ood_thresh = current_config["hyperparameters"]["ood_distance_threshold"]

    if top_confidence >= cls_thresh and metric_dist <= ood_thresh:
        verdict = "VERIFIED_KNOWN_TARGET"
        status_color = "GREEN"
        ecm_action = "MONITOR_ONLY"
    elif top_confidence >= cls_thresh and metric_dist > ood_thresh:
        verdict = "UNKNOWN_THREAT_ANOMALY"
        status_color = "ORANGE"
        ecm_action = "AUTOMATED_JAMMER_ENGAGED"
    else:
        verdict = "OOD_NOISE_REJECTED"
        status_color = "RED"
        ecm_action = "ELINT_LOGGING"

    top3_idx = np.argsort(probs_np)[-3:][::-1]
    top3_probs = [
        {"class": f"RADAR_FAMILY_{idx:02d}", "prob": round(float(probs_np[idx]), 4)}
        for idx in top3_idx
    ]

    return {
        "classifier": {
            "top_class": f"RADAR_FAMILY_{top_class_id:02d}",
            "confidence": round(top_confidence, 4),
            "top3": top3_probs
        },
        "qdrant_db": {
            "nearest_prototype": nearest_label,
            "freq_band": matched_meta.get("freq_band"),
            "distance": metric_dist,
            "threshold": ood_thresh,
            "is_ood": metric_dist > ood_thresh,
            "prototypes_trained": prototypes_trained
        },
        "fusion_verdict": {
            "verdict": verdict,
            "color": status_color,
            "ecm_trigger": ecm_action
        },
        "inference_engine": engine
    }


class ConfigUpdateModel(BaseModel):
    backbone_type: str
    learning_rate: float
    triplet_margin: float
    classifier_confidence_threshold: float
    ood_distance_threshold: float
    distance_metric: str
    inference_engine: str = "pytorch"
    qdrant_hnsw_m: int = 16
    qdrant_ef_construct: int = 100
    qdrant_quantization: str = "scalar"
    rpl_loss_weight: float = 0.1
    rpl_adversarial: bool = False


class DatasetRanges(BaseModel):
    """Physical parameter ranges the 20 radar-family archetypes are drawn from. Defaults
    match the original hardcoded constants exactly, so leaving these untouched reproduces
    prior behavior — only the dataset_seed (drawn fresh each run) changes by default."""
    freq_min_mhz: float = 2000.0
    freq_max_mhz: float = 18000.0
    pw_min_us: float = 0.5
    pw_max_us: float = 50.0
    pri_min_us: float = 10.0
    pri_max_us: float = 1000.0
    rise_min_ns: float = 5.0
    rise_max_ns: float = 150.0

    def to_ranges_dict(self) -> dict:
        return {
            "freq_range": (self.freq_min_mhz, self.freq_max_mhz),
            "pw_range": (self.pw_min_us, self.pw_max_us),
            "pri_range": (self.pri_min_us, self.pri_max_us),
            "rise_range": (self.rise_min_ns, self.rise_max_ns),
        }

    def to_columns(self) -> dict:
        return self.dict()


class TrainRequest(BaseModel):
    epochs: int = 20


class GenerateClassificationRequest(BaseModel):
    num_classes: int = 20
    num_holdout_classes: int = 3
    samples_per_class: int = 200
    noise_pct: float = 0.10
    ranges: DatasetRanges = DatasetRanges()

def serve_page(filename: str):
    with open(os.path.join(UI_DIR, filename), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

class LoginRequest(BaseModel):
    username: str
    password: str

@app.get("/login")
async def get_login_page():
    return serve_page("login.html")

@app.post("/api/login")
async def login(req: LoginRequest, request: Request):
    if req.username == ADMIN_USERNAME and req.password == ADMIN_PASSWORD:
        resp = JSONResponse({"status": "success"})
        resp.set_cookie(
            key=SESSION_COOKIE_NAME, value=_SESSION_TOKEN,
            httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7,
            secure=(request.url.scheme == "https"),
        )
        return resp
    return JSONResponse({"status": "error", "message": "Invalid User ID or password."}, status_code=401)

@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"status": "success"})
    resp.delete_cookie(SESSION_COOKIE_NAME)
    return resp

@app.get("/")
async def get_training_page():
    return serve_page("index.html")

@app.get("/validation")
async def get_validation_page():
    return serve_page("validation.html")

@app.get("/inference")
async def get_inference_page():
    return serve_page("inference.html")

@app.get("/dataset")
async def get_dataset_page():
    return serve_page("dataset.html")

@app.get("/analytics")
async def get_analytics_page():
    return serve_page("analytics.html")

@app.get("/live")
async def get_live_page():
    return serve_page("live.html")

@app.get("/help")
async def get_help_page():
    return serve_page("help.html")

@app.get("/api/config")
async def get_current_config():
    return JSONResponse(content=current_config)

@app.post("/api/config")
async def update_config(cfg: ConfigUpdateModel):
    """Never touches the live model or its trained prototypes — only stores the requested
    config. `backbone_type` is the one field that eventually requires a different model
    architecture, but that rebuild is deliberately deferred to the moment Train Model is
    actually clicked (see _run_training_job's own backbone-mismatch check below), exactly
    like the existing num_classes-mismatch handling already works. This means Apply
    Configuration can never destroy a trained model — the currently trained weights keep
    serving Live Simulation/Inference/Testing right up until you choose to retrain, even
    after picking a different backbone here."""
    global current_config, vector_db
    current_config["model"]["backbone_type"] = cfg.backbone_type
    current_config["model"]["inference_engine"] = cfg.inference_engine
    current_config["hyperparameters"]["learning_rate"] = cfg.learning_rate
    current_config["hyperparameters"]["triplet_margin"] = cfg.triplet_margin
    current_config["hyperparameters"]["classifier_confidence_threshold"] = cfg.classifier_confidence_threshold
    current_config["hyperparameters"]["ood_distance_threshold"] = cfg.ood_distance_threshold
    current_config["hyperparameters"]["distance_metric"] = cfg.distance_metric
    current_config["hyperparameters"]["rpl_loss_weight"] = cfg.rpl_loss_weight
    current_config["hyperparameters"]["rpl_adversarial"] = cfg.rpl_adversarial
    current_config["vector_db"]["hnsw_m"] = cfg.qdrant_hnsw_m
    current_config["vector_db"]["hnsw_ef_construct"] = cfg.qdrant_ef_construct
    current_config["vector_db"]["quantization"] = cfg.qdrant_quantization

    # Qdrant re-indexing is safe to apply immediately — it re-indexes the SAME already-trained
    # prototype vectors under a new metric/HNSW config, it doesn't touch the model's weights
    # or discard anything, so it doesn't need to wait for a retrain the way an architecture
    # change does.
    vector_db.update_metric(cfg.distance_metric)
    vector_db.update_indexing(
        hnsw_m=cfg.qdrant_hnsw_m,
        hnsw_ef_construct=cfg.qdrant_ef_construct,
        quantization=cfg.qdrant_quantization,
    )

    backbone_pending = cfg.backbone_type != model.backbone_type
    message = (
        f"Configuration saved. Backbone is now set to '{cfg.backbone_type}' but the currently "
        f"trained '{model.backbone_type}' model keeps serving Live Simulation/Inference/Testing "
        f"until you click Train Model — that's when the new architecture actually takes effect."
        if backbone_pending else
        "Configuration saved. Trained model and prototypes are unaffected — no need to retrain."
    )
    return JSONResponse({"status": "success", "message": message})

async def _run_training_job(req: TrainRequest):
    """Runs in the background so the client never holds a long-lived HTTP request open —
    the frontend only ever talks to /api/train/status, which stays fast and cheap to poll."""
    global training_status, prototypes_trained, model, vector_db, onnx_session
    training_status["is_training"] = True
    training_status["progress"] = {"epoch": 0, "epochs": req.epochs, "loss": None}
    training_status["error"] = None
    start_time = time.time()
    try:
        if not (os.path.exists(CLASSIFICATION_DATASET_CSV) and os.path.exists(CLASSIFICATION_DATASET_META)):
            raise RuntimeError("No classification dataset found — generate one on the Dataset page first.")

        with open(CLASSIFICATION_DATASET_META, "r") as f:
            meta = json.load(f)
        df = pd.read_csv(CLASSIFICATION_DATASET_CSV)

        num_classes = meta["num_classes"]
        dataset_seed = meta["dataset_seed"]
        dataset = load_classification_csv_for_training(
            df, meta["holdout_class_ids"], split_seed=dataset_seed
        )

        current_config["model"]["num_classes"] = num_classes
        needs_rebuild = (
            num_classes != model.classifier_head.out_features or
            current_config["model"]["backbone_type"] != model.backbone_type
        )
        if needs_rebuild:
            # Either the generated dataset defines a different num_classes than the currently
            # loaded model, or Apply Configuration set a different backbone_type than what's
            # currently loaded (deliberately deferred until now — see update_config) — rebuild
            # to match right before training. Existing prototypes/weights were computed in the
            # old architecture and are no longer valid regardless of which of the two changed.
            model = DualHeadEWModel(
                backbone_type=current_config["model"]["backbone_type"],
                input_dim=current_config["model"]["input_dim"],
                embed_dim=current_config["model"]["embedding_dim"],
                num_classes=num_classes,
            )
            model.eval()
            vector_db.num_classes = num_classes
            onnx_session = None
            prototypes_trained = False

        def progress_cb(epoch, epochs, loss):
            training_status["progress"] = {"epoch": epoch, "epochs": epochs, "loss": loss}

        metrics = await asyncio.to_thread(
            train_model, model, dataset, current_config, req.epochs, 64, progress_cb
        )

        centroids = metrics.pop("centroids")
        class_freqs = dataset["class_frequencies_mhz"]
        metadata = {c: {"freq_band": frequency_to_band(class_freqs[c])} for c in centroids}
        vector_db.upsert_prototypes(centroids, metadata)
        prototypes_trained = True
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
        await asyncio.to_thread(export_onnx, model)

        training_status["last_metrics"] = metrics

        hp = current_config["hyperparameters"]
        vdb = current_config["vector_db"]
        insert_run({
            "backbone_type": current_config["model"]["backbone_type"],
            "distance_metric": hp["distance_metric"],
            "inference_engine": current_config["model"]["inference_engine"],
            "hnsw_m": vdb["hnsw_m"],
            "hnsw_ef_construct": vdb["hnsw_ef_construct"],
            "quantization": vdb["quantization"],
            "num_classes": num_classes,
            "num_holdout_classes": meta["num_holdout_classes"],
            "samples_per_class": meta["samples_per_class"],
            "noise_pct": meta.get("noise_pct", 0.0),
            "epochs": req.epochs,
            "learning_rate": hp["learning_rate"],
            "triplet_margin": hp["triplet_margin"],
            "classifier_confidence_threshold": hp["classifier_confidence_threshold"],
            "ood_distance_threshold": hp["ood_distance_threshold"],
            "rpl_loss_weight": hp.get("rpl_loss_weight", 0.1),
            "rpl_adversarial": metrics.get("rpl_adversarial", False),
            "closed_set_accuracy": metrics["closed_set_accuracy"],
            "open_set_auroc": metrics["open_set_auroc"],
            "open_set_auroc_rpl": metrics.get("open_set_auroc_rpl"),
            "final_loss": metrics["loss_curve"][-1],
            "training_duration_seconds": round(time.time() - start_time, 2),
            "loss_curve": metrics["loss_curve"],
            "dataset_seed": dataset_seed,
            **meta["ranges"],
        })
    except Exception as e:
        training_status["error"] = str(e)
    finally:
        training_status["is_training"] = False

@app.post("/api/train")
async def train(req: TrainRequest):
    global _training_task
    if training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "Training already in progress"}, status_code=409)
    _training_task = asyncio.create_task(_run_training_job(req))
    return JSONResponse({"status": "started"})

@app.get("/api/train/status")
async def train_status():
    status = dict(training_status)
    if status["last_metrics"] is None and not status["is_training"]:
        # Survives process restarts/page navigation — the in-memory dict is empty right after
        # a fresh start, but the last real run is still on disk.
        latest = get_latest_run()
        if latest is not None:
            status["last_metrics"] = latest
    return JSONResponse({**status, "prototypes_trained": prototypes_trained})

@app.get("/api/train/history")
async def train_history(limit: int = 50):
    return JSONResponse({"status": "success", "runs": get_all_runs(limit)})

def _reused_seed_and_ranges(latest_run: Optional[dict]):
    """Extracts the exact dataset_seed + physical ranges + noise level a training run used, so
    Testing reconstructs the identical population it was trained on — including how much
    unstructured OOD noise was mixed in, not just the seed/ranges. Falls back to the original
    fixed defaults if there's no run yet, or it predates these columns (pre-migration)."""
    defaults = DatasetRanges()
    if latest_run is None:
        return None, defaults.to_ranges_dict(), 0.0
    seed = latest_run.get("dataset_seed")
    range_fields = defaults.dict()
    for key in range_fields:
        if latest_run.get(key) is not None:
            range_fields[key] = latest_run[key]
    ranges = DatasetRanges(**range_fields).to_ranges_dict()
    noise_pct = latest_run.get("noise_pct") if latest_run.get("noise_pct") is not None else 0.0
    return seed, ranges, noise_pct

@app.post("/api/dataset/generate_classification")
async def generate_classification_dataset(req: GenerateClassificationRequest):
    """The one canonical classification dataset Training and Testing both consume — raw PDW
    units + ground-truth class labels (true_class_id = -1 for holdout/unknown rows). A real,
    per-generation hyperparameter, not hardcoded: `num_classes` here decides how many radar
    families exist at all, and _run_training_job rebuilds the model's classifier head to match
    whatever this dataset specifies before training — the dataset is the source of truth, not
    config.yaml. Distinct from the Dataset page's interleaved deinterleaving CSV, whose
    emitter_class labels don't correspond to this classifier's classes at all. Draws a fresh
    random dataset_seed each call — real diversity, not the same fixed population every time —
    and persists it (+ ranges + holdout ids) to a sidecar meta.json so Training/Testing can
    reconstruct the exact same population later."""
    num_classes = req.num_classes
    result = build_test_dataset_rows(
        num_classes, req.num_holdout_classes, req.samples_per_class,
        ranges=req.ranges.to_ranges_dict(), noise_pct=req.noise_pct,
    )
    pd.DataFrame(result["rows"]).to_csv(CLASSIFICATION_DATASET_CSV, index=False)

    meta = {
        "dataset_seed": result["dataset_seed"],
        "num_classes": num_classes,
        "num_holdout_classes": req.num_holdout_classes,
        "samples_per_class": req.samples_per_class,
        "noise_pct": req.noise_pct,
        "holdout_class_ids": result["holdout_class_ids"],
        "known_class_ids": result["known_class_ids"],
        "ranges": req.ranges.to_columns(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(CLASSIFICATION_DATASET_META, "w") as f:
        json.dump(meta, f, indent=2)

    return FileResponse(CLASSIFICATION_DATASET_CSV, media_type="text/csv", filename="classification_dataset.csv")

class TestRequest(BaseModel):
    samples_per_class: int = 100
    dataset_source: Optional[str] = None   # "generated" | "uploaded"; None = auto-generate fresh sample
    dataset_filename: Optional[str] = None

_TEST_REQUIRED_COLUMNS = ["carrier_freq_mhz", "pulse_width_us", "pri_us", "rise_time_ns", "true_class_id"]

@app.post("/api/test")
async def run_test(req: TestRequest):
    sample_filename = None
    dataset_seed_value = None
    if req.dataset_filename:
        directory = TEST_UPLOAD_DIR if req.dataset_source == "uploaded" else CLASSIFICATION_DATASET_DIR
        path = os.path.join(directory, os.path.basename(req.dataset_filename))
        if not os.path.exists(path):
            return JSONResponse({"status": "error", "message": "Test dataset file not found."}, status_code=404)

        df = pd.read_csv(path)
        missing = [c for c in _TEST_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return JSONResponse({
                "status": "error",
                "message": f"This file isn't a classification dataset — missing columns {missing}. "
                           f"Use 'Generate Classification Dataset' on the Dataset page to get the right format."
            }, status_code=400)

        vectors = np.stack([
            normalize_pdw(r["carrier_freq_mhz"], r["pulse_width_us"], r["pri_us"], r["rise_time_ns"])
            for r in df.to_dict("records")
        ]).astype(np.float32)
        labels = df["true_class_id"].astype(int).to_numpy()
        known_mask = labels >= 0

        dataset = {
            "val_x": vectors[known_mask], "val_y": labels[known_mask],
            "openset_x": vectors[~known_mask], "openset_y": labels[~known_mask],
            "known_class_ids": sorted(set(labels[known_mask].tolist())),
            "holdout_class_ids": sorted(set(labels[~known_mask].tolist())) if (~known_mask).any() else [],
        }
        # The raw CSV collapses every holdout/noise row to the same true_class_id = -1 sentinel
        # (there's no other way to represent "unknown" in this format), so counting distinct
        # values in the column above always reports at most 1 holdout class regardless of how
        # many were actually excluded from training. The canonical Dataset-page file has a
        # sidecar meta.json with the real pre-collapse holdout_class_ids — use it for accurate
        # Known/Holdout reporting when evaluating that exact file (doesn't change scoring at
        # all, since evaluate_model only needs known-vs-not-known, not which original class).
        if (req.dataset_source != "uploaded"
                and os.path.basename(path) == os.path.basename(CLASSIFICATION_DATASET_CSV)
                and os.path.exists(CLASSIFICATION_DATASET_META)):
            with open(CLASSIFICATION_DATASET_META, "r") as f:
                meta = json.load(f)
            if meta.get("known_class_ids") and meta.get("holdout_class_ids"):
                dataset["known_class_ids"] = meta["known_class_ids"]
                dataset["holdout_class_ids"] = meta["holdout_class_ids"]
            dataset_seed_value = meta.get("dataset_seed")
        dataset_source_label = f"{req.dataset_source or 'generated'}:{os.path.basename(path)}"
    else:
        latest_run = get_latest_run()
        if latest_run is None:
            return JSONResponse({"status": "error", "message": "No completed training run yet — train a model first."}, status_code=404)

        num_classes = current_config["model"]["num_classes"]
        num_holdout = latest_run["num_holdout_classes"]
        reused_seed, reused_ranges, reused_noise_pct = _reused_seed_and_ranges(latest_run)

        # Same generator the Dataset page uses (raw physical-unit rows, not opaque normalized
        # vectors) — the previous "ephemeral, in-memory only, never saved" version of this path
        # meant the actual test sample could never be inspected after the fact. Saving it here
        # means it can be viewed/downloaded like any other dataset, and it removes a second,
        # largely-duplicate dataset-building function (build_classification_dataset) that only
        # this one call site ever used.
        sample_result = build_test_dataset_rows(
            num_classes, num_holdout, req.samples_per_class,
            dataset_seed=reused_seed, ranges=reused_ranges, noise_pct=reused_noise_pct,
        )
        sample_df = pd.DataFrame(sample_result["rows"])
        sample_df.to_csv(AUTO_TEST_SAMPLE_CSV, index=False)
        sample_filename = os.path.basename(AUTO_TEST_SAMPLE_CSV)

        # All known rows go to "val" (val_fraction=1.0) — none of this is ever trained on, it's
        # a fresh sample drawn purely for evaluation, so there's no train/val split to make.
        dataset = load_classification_csv_for_training(
            sample_df, sample_result["holdout_class_ids"], val_fraction=1.0,
            split_seed=int(np.random.randint(1_000_000)),
        )
        dataset_source_label = f"auto-generated fresh sample (dataset_seed={reused_seed})"
        dataset_seed_value = reused_seed

    distance_fn = build_distance_fn(current_config["hyperparameters"]["distance_metric"])
    metrics = evaluate_model(model, dataset, distance_fn)
    metrics.pop("centroids", None)

    num_known_pulses = int(dataset["val_x"].shape[0])
    num_openset_pulses = int(dataset["openset_x"].shape[0])

    record = {
        "backbone_type": current_config["model"]["backbone_type"],
        "distance_metric": current_config["hyperparameters"]["distance_metric"],
        "inference_engine": current_config["model"]["inference_engine"],
        "hnsw_m": current_config["vector_db"]["hnsw_m"],
        "hnsw_ef_construct": current_config["vector_db"]["hnsw_ef_construct"],
        "quantization": current_config["vector_db"]["quantization"],
        "dataset_source": dataset_source_label,
        "num_known_classes": metrics["num_known_classes"],
        "num_holdout_classes": metrics["num_holdout_classes"],
        "closed_set_accuracy": metrics["closed_set_accuracy"],
        "open_set_auroc": metrics["open_set_auroc"],
        "num_known_pulses": num_known_pulses,
        "num_openset_pulses": num_openset_pulses,
        "total_pulses": num_known_pulses + num_openset_pulses,
        "dataset_seed": dataset_seed_value,
    }
    insert_test_run(record)

    return JSONResponse({"status": "success", "metrics": metrics, "run": record, "sample_filename": sample_filename})

@app.get("/api/test/history")
async def test_history(limit: int = 50):
    return JSONResponse({"status": "success", "runs": get_all_test_runs(limit)})

@app.get("/api/test/sample")
async def download_auto_test_sample():
    """The exact rows the most recent 'auto-generate fresh sample' Test run was scored
    against — physical PDW units + true_class_id/true_label included (necessary to compute
    accuracy/AUROC at all), but that label column is only ever used for scoring after the
    model's predictions come back, never fed to the model as an input feature."""
    if not os.path.exists(AUTO_TEST_SAMPLE_CSV):
        return JSONResponse({"status": "error", "message": "No auto-generated test sample yet — run a Test with 'auto-generate fresh sample' first."}, status_code=404)
    return FileResponse(AUTO_TEST_SAMPLE_CSV, media_type="text/csv", filename="last_auto_test_sample.csv")

class InferRequest(BaseModel):
    carrier_freq_mhz: float
    pulse_width_us: float
    pri_us: float
    rise_time_ns: float

@app.post("/api/infer")
async def infer_single_pulse(req: InferRequest):
    pdw_vec = normalize_pdw(
        freq=req.carrier_freq_mhz,
        pw=req.pulse_width_us,
        pri=req.pri_us,
        rise=req.rise_time_ns,
        toa_ns=0,
    )
    result = run_inference_pipeline(pdw_vec)
    return JSONResponse({"status": "success", "result": result})

_LABEL_COLUMN_CANDIDATES = ["true_label", "emitter_class", "class_id"]


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Single shared candidate list for the ground-truth/class label column across both
    dataset shapes (classification and interleaved) — previously inspect_dataset and
    dataset_analytics each had their own slightly different candidate list."""
    return next((c for c in _LABEL_COLUMN_CANDIDATES if c in df.columns), None)


@app.get("/api/dataset/inspect")
async def inspect_dataset(source: str = "generated", filename: str = None):
    directory = {
        "uploaded": UPLOAD_DIR,
        "classification": CLASSIFICATION_DATASET_DIR,
        "auto_test_sample": AUTO_TEST_SAMPLE_DIR,
    }.get(source, GENERATED_DIR)
    if filename is None:
        csvs = [f for f in os.listdir(directory) if f.endswith(".csv")]
        if not csvs:
            return JSONResponse({"status": "error", "message": "No dataset found"}, status_code=404)
        filename = max(csvs, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    path = os.path.join(directory, os.path.basename(filename))
    if not os.path.exists(path):
        return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)

    df = pd.read_csv(path)
    label_col = _detect_label_column(df)
    class_counts = df[label_col].value_counts().to_dict() if label_col else {}
    features = _resolve_numeric_features(df)

    return JSONResponse({
        "status": "success",
        "filename": os.path.basename(filename),
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "label_column": label_col,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "histograms": {
            "frequency_mhz": _histogram(df[features["frequency_mhz"]].to_numpy(dtype=float), bins=10) if "frequency_mhz" in features else None,
            "pulse_width": _histogram(df[features["pulse_width"]].to_numpy(dtype=float), bins=10) if "pulse_width" in features else None,
        }
    })

_NUMERIC_FEATURE_ALIASES = {
    "frequency_mhz": ["frequency_mhz", "carrier_freq_mhz"],
    "pulse_width": ["pw_ns", "pulse_width_us"],
    "pri_us": ["pri_us"],
    "rise_time_ns": ["rise_time_ns"],
    "aoa_deg": ["aoa_deg"],
}


def _resolve_numeric_features(df: pd.DataFrame) -> dict:
    """Maps canonical feature names to whichever column actually present carries that quantity —
    the classification and interleaved datasets use different column names/units for the same
    physical measurements, so callers work in canonical names regardless of dataset shape."""
    resolved = {}
    for canonical, aliases in _NUMERIC_FEATURE_ALIASES.items():
        col = next((c for c in aliases if c in df.columns and pd.api.types.is_numeric_dtype(df[c])), None)
        if col:
            resolved[canonical] = col
    return resolved


def _histogram(values: np.ndarray, bins: int = 12) -> Optional[dict]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None
    counts, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "counts": [int(c) for c in counts]}


def _tukey_box_stats(values: np.ndarray) -> Optional[dict]:
    """Standard Tukey boxplot: box at Q1/median/Q3, whiskers extended to the furthest point
    still within 1.5*IQR of the box, everything beyond flagged as an outlier."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = values[(values >= lo_fence) & (values <= hi_fence)]
    whisker_lo = float(inliers.min()) if len(inliers) else float(values.min())
    whisker_hi = float(inliers.max()) if len(inliers) else float(values.max())
    outliers = sorted(values[(values < whisker_lo) | (values > whisker_hi)].tolist())
    return {
        "min": whisker_lo, "q1": float(q1), "median": float(median), "q3": float(q3), "max": whisker_hi,
        "outliers": outliers[:25],
    }


def _pca_2d(matrix: np.ndarray, labels: Optional[List[str]] = None, max_points: int = 800) -> Optional[dict]:
    """Standardizes and projects `matrix` to 2D via PCA — the standard first-look for whether
    classes are actually separable in a given feature space, rather than assuming it from
    summary statistics alone. Subsamples for plotting if there are many more rows than
    max_points (PCA itself still fits on the full matrix)."""
    valid = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid]
    if labels is not None:
        labels = [l for l, v in zip(labels, valid) if v]
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    standardized = (matrix - matrix.mean(axis=0)) / std
    n_components = min(2, matrix.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(standardized)
    if n_components == 1:
        projected = np.concatenate([projected, np.zeros_like(projected)], axis=1)

    idx = np.arange(len(projected))
    if len(idx) > max_points:
        idx = np.random.RandomState(0).choice(idx, size=max_points, replace=False)

    points = [{"x": float(projected[i, 0]), "y": float(projected[i, 1]),
               "label": str(labels[i]) if labels is not None else None} for i in idx]
    explained = pca.explained_variance_ratio_.tolist()
    return {"points": points, "explained_variance_ratio": [round(float(v), 4) for v in explained]}


def _silhouette(matrix: np.ndarray, labels: List[str]) -> Optional[float]:
    """Single-number complement to the PCA scatter: how well-separated classes actually are
    in a feature space (+1 = tight, well-separated clusters; 0 = overlapping; negative =
    likely mislabeled/worse than random). Needs at least 2 classes with 2+ samples each."""
    valid = ~np.isnan(matrix).any(axis=1)
    matrix, labels = matrix[valid], [l for l, v in zip(labels, valid) if v]
    unique = set(labels)
    if len(unique) < 2 or matrix.shape[0] < 4:
        return None
    counts = {u: labels.count(u) for u in unique}
    if any(c < 2 for c in counts.values()):
        return None
    try:
        return round(float(silhouette_score(matrix, labels)), 4)
    except Exception:
        return None


@app.get("/api/analytics")
async def dataset_analytics(source: str = "classification", filename: str = None,
                             max_boxplot_classes: int = 15, max_scatter_rows: int = 600):
    """One-shot EDA payload for the Analytics page: summary statistics, a Pearson correlation
    matrix, per-feature histograms, per-class Tukey boxplots, and a row sample for the scatter
    explorer — computed once per dataset selection so the page's plot buttons just toggle
    which already-fetched view is shown, rather than re-hitting the server per click."""
    directory = {
        "uploaded": UPLOAD_DIR,
        "classification": CLASSIFICATION_DATASET_DIR,
        "generated": GENERATED_DIR,
        "auto_test_sample": AUTO_TEST_SAMPLE_DIR,
    }.get(source, CLASSIFICATION_DATASET_DIR)

    if filename is None:
        csvs = [f for f in os.listdir(directory) if f.endswith(".csv")]
        if not csvs:
            return JSONResponse({"status": "error", "message": "No dataset found"}, status_code=404)
        filename = max(csvs, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    path = os.path.join(directory, os.path.basename(filename))
    if not os.path.exists(path):
        return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)

    df = pd.read_csv(path)
    label_col = _detect_label_column(df)
    class_counts_full = df[label_col].value_counts() if label_col else pd.Series(dtype=int)

    features = _resolve_numeric_features(df)
    if not features:
        return JSONResponse({"status": "error", "message": "No recognized numeric PDW columns in this file."}, status_code=400)
    sub = df[list(features.values())].rename(columns={v: k for k, v in features.items()})

    # Standard derived radar quantities — real ESM analysts work with these at least as often
    # as raw PW/PRI. Only computable for the classification dataset (the interleaved dataset
    # has no direct per-row PRI, only the whole-stream inter-pulse-interval already handled
    # below), consistent with how PRI is already treated differently between the two shapes.
    if "pri_us" in sub.columns and "pulse_width" in sub.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            sub["duty_cycle_pct"] = (sub["pulse_width"] / sub["pri_us"] * 100).replace([np.inf, -np.inf], np.nan)
            sub["prf_hz"] = (1e6 / sub["pri_us"]).replace([np.inf, -np.inf], np.nan)

    describe = {}
    for col in sub.columns:
        vals = sub[col].to_numpy(dtype=float)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        describe[col] = {
            "count": int(len(valid)), "missing": int(len(vals) - len(valid)),
            "mean": float(np.mean(valid)), "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(np.min(valid)), "p25": float(np.percentile(valid, 25)),
            "median": float(np.median(valid)), "p75": float(np.percentile(valid, 75)),
            "max": float(np.max(valid)),
        }

    corr_df = sub.corr(numeric_only=True).round(4)
    correlation = {
        "columns": list(corr_df.columns),
        "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in corr_df.to_numpy()],
    }

    histograms = {}
    for col in sub.columns:
        h = _histogram(sub[col].to_numpy(dtype=float))
        if h:
            histograms[col] = h
    # Interleaved streams have no direct PRI column (PRI is only meaningful per-emitter, unknown
    # pre-deinterleaving) — the raw inter-pulse-interval histogram across the whole intercepted
    # stream is the real-world ESM stand-in analysts use instead.
    if "toa_ns" in df.columns and "pri_us" not in features:
        toa_sorted = np.sort(df["toa_ns"].dropna().to_numpy(dtype=float))
        if len(toa_sorted) > 1:
            h = _histogram(np.diff(toa_sorted))
            if h:
                histograms["inter_pulse_interval_ns"] = h

    boxplot = {}
    classes_truncated = False
    if label_col:
        top_classes = class_counts_full.index.tolist()
        if len(top_classes) > max_boxplot_classes:
            top_classes = top_classes[:max_boxplot_classes]
            classes_truncated = True
        label_values = df[label_col].to_numpy()
        for col in sub.columns:
            col_values = sub[col].to_numpy(dtype=float)
            classes_here, stats_here = [], []
            for cls in top_classes:
                stats = _tukey_box_stats(col_values[label_values == cls])
                if stats:
                    classes_here.append(str(cls))
                    stats_here.append(stats)
            if stats_here:
                boxplot[col] = {"classes": classes_here, "stats": stats_here}

    sample_df = sub.copy()
    if label_col:
        sample_df["_label"] = df[label_col].astype(str).to_numpy()
    if len(sample_df) > max_scatter_rows:
        sample_df = sample_df.sample(n=max_scatter_rows, random_state=0)
    scatter_sample = sample_df.where(pd.notnull(sample_df), None).to_dict("records")

    # Frequency band breakdown — the classic first-pass ESM/EW analysis (how much of the
    # intercepted spectrum falls in which band), using the same L/S/C/X/Ku/K/Ka mapping the
    # live inference pipeline already uses for its Qdrant payload metadata.
    band_distribution = None
    if "frequency_mhz" in sub.columns:
        bands = sub["frequency_mhz"].dropna().apply(frequency_to_band)
        band_distribution = {str(k): int(v) for k, v in bands.value_counts().items()}

    # PRI modulation pattern breakdown — only meaningful now that the generator actually
    # produces more than one pattern (see src/data/pri_patterns.py); shows which of the six
    # standard types (constant/jittered/staggered/sliding/wobulated/dwell_switch) the current
    # dataset's families/emitters are using.
    pri_pattern_distribution = None
    if "pri_pattern" in df.columns:
        pri_pattern_distribution = {str(k): int(v) for k, v in df["pri_pattern"].value_counts().items()}

    # Class balance as a single number, not just the bar chart — Shannon entropy (normalized
    # to [0,1], 1.0 = perfectly even) and the max/min count ratio, standard dataset-health
    # metrics for any classification task.
    class_diversity = None
    if label_col and len(class_counts_full) > 1:
        counts = class_counts_full.to_numpy(dtype=float)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        max_entropy = float(np.log(len(counts)))
        class_diversity = {
            "shannon_entropy_normalized": round(entropy / max_entropy, 4) if max_entropy > 0 else None,
            "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
            "num_classes": int(len(counts)),
        }

    # PCA + silhouette in the RAW physical feature space — directly visualizes/quantifies
    # whether classes are actually separable before any model gets involved, the standard
    # first question in any classification EDA.
    label_values_list = df[label_col].astype(str).tolist() if label_col else None
    pca_raw = _pca_2d(sub.to_numpy(dtype=float), label_values_list)
    silhouette_raw = _silhouette(sub.to_numpy(dtype=float), label_values_list) if label_values_list else None

    # PCA + silhouette in the model's LEARNED embedding space — the actual research question
    # this whole platform exists to answer (does the trained model separate known classes
    # well?), made directly visual rather than only inferable from AUROC/accuracy numbers.
    # Only possible for the classification-format dataset (matching PDW schema + labels the
    # model was trained on) with an actually-trained model available.
    pca_embedding, silhouette_embedding = None, None
    if source == "classification" and prototypes_trained and label_col and \
            all(c in df.columns for c in ["carrier_freq_mhz", "pulse_width_us", "pri_us", "rise_time_ns"]):
        eval_rows = df if len(df) <= max_scatter_rows else df.sample(n=max_scatter_rows, random_state=0)
        vectors = np.stack([
            normalize_pdw(r["carrier_freq_mhz"], r["pulse_width_us"], r["pri_us"], r["rise_time_ns"])
            for r in eval_rows.to_dict("records")
        ]).astype(np.float32)
        with torch.no_grad():
            _, embeddings = model(torch.from_numpy(vectors))
        embed_np = embeddings.numpy()
        embed_labels = eval_rows[label_col].astype(str).tolist()
        pca_embedding = _pca_2d(embed_np, embed_labels)
        silhouette_embedding = _silhouette(embed_np, embed_labels)

    # Pulse train timeline — the classic real-time ESM console view (time-of-arrival vs.
    # frequency), only meaningful for the interleaved dataset which has genuine ToA ordering.
    # This is exactly what makes deinterleaving visually obvious: overlapping pulse trains
    # from different emitters, and PRI pattern shapes (staggered/sliding/etc.) become visible
    # as literal geometric patterns along the time axis.
    pulse_train_timeline = None
    if "toa_ns" in df.columns and "frequency_mhz" in df.columns:
        timeline_df = df.sort_values("toa_ns")
        if len(timeline_df) > max_scatter_rows:
            timeline_df = timeline_df.iloc[np.linspace(0, len(timeline_df) - 1, max_scatter_rows).astype(int)]
        if label_col:
            timeline_label_col = label_col
        elif "emitter_instance_id" in df.columns:
            timeline_label_col = "emitter_instance_id"
        else:
            timeline_label_col = None
        pulse_train_timeline = [
            {"toa_ns": float(r["toa_ns"]), "frequency_mhz": float(r["frequency_mhz"]),
             "label": str(r[timeline_label_col]) if timeline_label_col else None}
            for r in timeline_df.to_dict("records")
        ]

    return JSONResponse({
        "status": "success",
        "filename": os.path.basename(filename),
        "total_rows": int(len(df)),
        "label_column": label_col,
        "class_counts": {str(k): int(v) for k, v in class_counts_full.items()},
        "features": list(sub.columns),
        "describe": describe,
        "correlation": correlation,
        "histograms": histograms,
        "boxplot": boxplot,
        "boxplot_classes_truncated": classes_truncated,
        "scatter_sample": scatter_sample,
        "band_distribution": band_distribution,
        "pri_pattern_distribution": pri_pattern_distribution,
        "class_diversity": class_diversity,
        "pca_raw": pca_raw,
        "silhouette_raw": silhouette_raw,
        "pca_embedding": pca_embedding,
        "silhouette_embedding": silhouette_embedding,
        "embedding_available": prototypes_trained,
        "pulse_train_timeline": pulse_train_timeline,
    })


class DeinterleaveRequest(BaseModel):
    method: str = "classical"  # "classical" (DBSCAN over raw features) | "deep_metric" (triplet-loss embedding + DBSCAN)
    eps: float = 0.5
    min_samples: int = 5
    epochs: int = 30  # deep_metric only — embedding network training epochs


@app.post("/api/deinterleave")
async def run_deinterleave(req: DeinterleaveRequest = DeinterleaveRequest()):
    path = os.path.join(GENERATED_DIR, "synthetic_radar_pdws_v2.csv")
    if not os.path.exists(path):
        return JSONResponse({"status": "error", "message": "Generate an interleaved dataset first"}, status_code=404)

    df = pd.read_csv(path)
    try:
        if req.method == "deep_metric":
            result = await asyncio.to_thread(
                deep_metric_deinterleave, df, req.eps, req.min_samples, req.epochs
            )
        else:
            result = deinterleave(df, eps=req.eps, min_samples=req.min_samples)
            result.pop("cluster_labels", None)
            result["method"] = "classical"
    except ValueError as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    return JSONResponse({"status": "success", **result})

class InterleavedRanges(BaseModel):
    """Physical parameter ranges the interleaved-dataset emitters are drawn from. Units match
    generate_interleaved_pdws' native scale (PW/PRI/rise in nanoseconds, not microseconds —
    distinct from the classifier's DatasetRanges)."""
    freq_min_mhz: float = 3000.0
    freq_max_mhz: float = 11000.0
    pw_min_ns: float = 400.0
    pw_max_ns: float = 1300.0
    pri_min_ns: float = 100000.0
    pri_max_ns: float = 700000.0
    aoa_min_deg: float = 0.0
    aoa_max_deg: float = 180.0
    rise_min_ns: float = 2.0
    rise_max_ns: float = 65.0


class GenerateInterleavedRequest(BaseModel):
    simulation_time_ns: int = 50000000
    num_emitters: int = 4
    num_classes: int = 3
    seed: Optional[int] = None
    ranges: InterleavedRanges = InterleavedRanges()

INTERLEAVED_DATASET_META = os.path.join(GENERATED_DIR, "synthetic_radar_pdws_v2_meta.json")

@app.post("/api/dataset/generate")
async def generate_dataset(req: GenerateInterleavedRequest):
    filepath = os.path.join(GENERATED_DIR, "synthetic_radar_pdws_v2.csv")
    r = req.ranges
    result = generate_interleaved_pdws(
        filename=filepath,
        simulation_time_ns=req.simulation_time_ns,
        num_emitters=req.num_emitters,
        num_classes=req.num_classes,
        seed=req.seed,
        freq_range=(r.freq_min_mhz, r.freq_max_mhz),
        pw_range=(r.pw_min_ns, r.pw_max_ns),
        pri_range=(r.pri_min_ns, r.pri_max_ns),
        aoa_range=(r.aoa_min_deg, r.aoa_max_deg),
        rise_range=(r.rise_min_ns, r.rise_max_ns),
    )
    with open(INTERLEAVED_DATASET_META, "w") as f:
        json.dump(result, f, indent=2)

    return FileResponse(
        filepath,
        media_type="text/csv",
        filename="synthetic_radar_pdws_v2.csv"
    )

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Interleaved-format PDW upload (Dataset page) — consumed by Live Simulation and deinterleaving.
    Kept in a separate directory from classification-format test uploads so the two can't collide."""
    contents = await file.read()
    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(contents)
    return JSONResponse({"status": "success", "filename": safe_name, "size_bytes": len(contents)})

@app.post("/api/test/upload")
async def upload_test_dataset(file: UploadFile = File(...)):
    """Classification-format test dataset upload (Testing page) — separate from the interleaved
    upload directory so it never gets picked up by Live Simulation's dataset replay."""
    contents = await file.read()
    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(TEST_UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(contents)
    return JSONResponse({"status": "success", "filename": safe_name, "size_bytes": len(contents)})

MAX_LIVE_FEED_SECONDS = 900  # hard server-side cap (15 min) regardless of what the client requests


async def _stream_live_feed(websocket: WebSocket, num_classes: int, duration_seconds: int,
                             freq_range, pw_range, pri_range, rise_range):
    """Genuinely continuous, unlabeled real-time generation — no CSV, nothing persisted,
    nothing pre-computed. Each pulse is drawn fresh and pushed the instant it's generated,
    exactly mirroring how a real ESM operator actually experiences a live feed: sensor
    readings and the model's own best guess, with no oracle access to ground truth at all.
    This is deliberately NOT the same as replaying a dataset with hidden labels — the true
    class is never even computed here in a way that's exposed; it's only used internally to
    draw a physically-plausible sample and to give a small set of simulated emitters
    persistent state (so a "war mode" search -> track -> lock-on escalation is something that
    can actually be observed unfolding over the session), never sent to the client or scored.
    """
    duration_seconds = min(max(duration_seconds, 10), MAX_LIVE_FEED_SECONDS)
    seed = int(np.random.randint(1_000_000))
    library = build_emitter_library(num_classes, seed=seed, freq_range=freq_range, pw_range=pw_range,
                                     pri_range=pri_range, rise_range=rise_range)
    rng = np.random.RandomState()  # unseeded — genuine fresh entropy per pulse, not a reproducible sequence

    # A handful of persistent simulated emitters, each with its own evolving operating mode —
    # without this, every pulse would be an independent draw with no possibility of observing
    # any single emitter escalate over time, which is exactly what a real live feed shows.
    num_tracks = min(5, num_classes)
    tracks = [
        {"class_id": int(rng.randint(num_classes)), "mode_idx": 0, "dwell": sample_dwell(rng)}
        for _ in range(num_tracks)
    ]

    start_time = time.time()
    pulse_counter = 0
    try:
        while (time.time() - start_time) < duration_seconds:
            pulse_counter += 1
            if rng.random() < 0.08:
                sample = sample_ood_pulse(rng)
            else:
                track = tracks[int(rng.randint(num_tracks))]
                track["dwell"] -= 1
                if track["dwell"] <= 0:
                    track["mode_idx"] = transition_mode(rng, track["mode_idx"])
                    track["dwell"] = sample_dwell(rng)
                sample = sample_known_pulse(library[track["class_id"]], rng, mode=OPERATING_MODES[track["mode_idx"]])

            pdw_vec = normalize_pdw(sample["freq"], sample["pw"], sample["pri"],
                                     sample["rise"], toa_ns=time.time_ns() % 1_000_000_000)
            result = run_inference_pipeline(pdw_vec)

            raw_meta = {
                "carrier_freq_mhz": round(float(sample["freq"]), 2),
                "pulse_width_us": round(float(sample["pw"]), 3),
                "pri_us": round(float(sample["pri"]), 2),
                "rise_time_ns": round(float(sample["rise"]), 1),
                # No true_label / class_id / operating_mode / is_correct — this mode never
                # exposes ground truth at all, matching genuine real-time operation.
            }
            elapsed = round(time.time() - start_time, 1)
            payload = {
                "pulse_id": pulse_counter, "total_rows": None,
                "elapsed_seconds": elapsed, "duration_seconds": duration_seconds,
                "pdw_meta": raw_meta, **result,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.4)

        await websocket.send_text(json.dumps({
            "stream_complete": True, "reason": "duration_elapsed",
            "total_rows": pulse_counter, "dataset_filename": f"live feed ({duration_seconds}s session)",
        }))
        await websocket.close()
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket, source: str = "interleaved",
                               num_classes: int = 20, duration_seconds: int = 300,
                               freq_min_mhz: float = 2000.0, freq_max_mhz: float = 18000.0,
                               pw_min_us: float = 0.5, pw_max_us: float = 50.0,
                               pri_min_us: float = 10.0, pri_max_us: float = 1000.0,
                               rise_min_ns: float = 5.0, rise_max_ns: float = 150.0):
    """Replays a PDW dataset pulse-by-pulse through the live inference pipeline, stopping
    automatically at the last row — Pulse ID is always bounded by that dataset's real row
    count, never an unbounded synthetic stream.

    `source="interleaved"` (default) replays the Dataset page's interleaved/deinterleaving
    dataset in real time-of-arrival order — its emitter_class labels live in a different,
    unrelated label space from the classifier's actual trained classes, so predictions here
    are NOT checkable against ground truth (demo/visualization value only).

    `source="classification"` instead replays the Classification Dataset — the exact schema
    and label space the model was actually trained on — so each prediction gets a real,
    verifiable `is_correct` flag: for known rows, whether the top predicted class matches the
    true label; for held-out/noise rows, whether the fusion verdict correctly did NOT confirm
    them as a known target. This is what makes the live feed an actual evaluation, not just
    a visualization.

    `source="live_feed"` generates genuinely continuous, unlabeled pulses on the fly (nothing
    read from or written to disk), for up to `duration_seconds` (hard-capped at 15 minutes
    server-side) — the closest thing to actually standing in front of a real receiver, with
    population ranges/class count configurable via the Advanced Population Ranges panel.
    """
    if not _is_authenticated(websocket.cookies):
        await websocket.close(code=1008)  # policy violation
        return
    await websocket.accept()

    if source == "live_feed":
        await _stream_live_feed(
            websocket, num_classes, duration_seconds,
            (freq_min_mhz, freq_max_mhz), (pw_min_us, pw_max_us), (pri_min_us, pri_max_us),
            (rise_min_ns, rise_max_ns),
        )
        return

    use_classification = source == "classification"

    if use_classification:
        dataset_path = CLASSIFICATION_DATASET_CSV
        if not os.path.exists(dataset_path):
            await websocket.send_text(json.dumps({
                "error": "No classification dataset found. Generate one on the Dataset page first."
            }))
            await websocket.close()
            return
        try:
            df = load_classification_replay_dataset(dataset_path)
        except ValueError as e:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
            return
    else:
        dataset_path = find_latest_dataset(GENERATED_DIR, UPLOAD_DIR)
        if dataset_path is None:
            await websocket.send_text(json.dumps({
                "error": "No dataset found. Generate or upload one first, then Start Live Simulation."
            }))
            await websocket.close()
            return
        try:
            df = load_replay_dataset(dataset_path)
        except ValueError as e:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
            return

    total_rows = len(df)

    try:
        prev_toa_ns = None
        for pulse_counter, (_, row) in enumerate(df.iterrows(), start=1):
            if use_classification:
                pdw_vec, raw_meta = classification_row_to_pdw_vector(row)
            else:
                pdw_vec, raw_meta = row_to_pdw_vector(row, prev_toa_ns)
                prev_toa_ns = row["toa_ns"]

            result = run_inference_pipeline(pdw_vec)

            if use_classification:
                predicted_label = result["classifier"]["top_class"]
                if raw_meta["class_id"] >= 0:
                    is_correct = predicted_label == raw_meta["true_label"]
                else:
                    is_correct = result["fusion_verdict"]["verdict"] != "VERIFIED_KNOWN_TARGET"
                raw_meta["is_correct"] = is_correct

            payload = {
                "pulse_id": pulse_counter,
                "total_rows": total_rows,
                "pdw_meta": raw_meta,
                **result
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.4)

        await websocket.send_text(json.dumps({
            "stream_complete": True,
            "total_rows": total_rows,
            "dataset_filename": os.path.basename(dataset_path)
        }))
        await websocket.close()
    except WebSocketDisconnect:
        pass
