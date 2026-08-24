# SOSA-Aligned Open-Set Radar Emitter Recognition Platform

A research platform for open-set radar emitter classification, real-time ESM (Electronic
Support Measures) simulation, and pulse deinterleaving — built for PhD research into
SOSA-aligned electronic warfare signal processing. Every metric, dataset, and model in this
system is real and computed live (no hardcoded/simulated results): real PyTorch training,
real embedded Qdrant vector search, real open-set evaluation, real clustering-based
deinterleaving.

**Explicit design constraints:** PyTorch + ONNX Runtime only (no TensorRT — no NVIDIA GPU on
the target dev machine); embedded/local `qdrant-client` only (no Docker, no Qdrant server);
synthetic-data-only (no real VITA49 capture or radar hardware).

---

## 1. What this system does

At its core, this platform trains a neural network to identify which of N known radar
"families" a single intercepted pulse belongs to, while also recognizing when a pulse belongs
to **none** of them (open-set recognition) — and separately, demonstrates deinterleaving
(reconstructing which pulses came from which physical emitter out of a jumbled, overlapping
stream) and a real-time ESM operator console.

These are the six pages, each backing a distinct part of the pipeline:

| Page | Route | Purpose |
|---|---|---|
| Training | `/` | Configure and train the classifier; view run history |
| Testing | `/validation` | Evaluate a trained model against held-out/fresh data |
| Inference | `/inference` | Manually submit one PDW and see the model's live decision |
| Dataset | `/dataset` | Generate the Classification Dataset and the Interleaved Dataset (two independent pipelines) |
| Analytics | `/analytics` | EDA — statistics, correlation, PCA/separability, ESM-domain views |
| Live Simulation | `/live` | Real-time operator console: replay a dataset or generate a genuinely continuous unlabeled feed |
| Help | `/help` | In-app glossary for every metric/control on every page |

---

## 2. Architecture overview

```
                    ┌─────────────────────────────────────────────┐
                    │              Dataset page (UI)               │
                    │  generates two INDEPENDENT files:            │
                    │  - classification_dataset.csv (+ meta.json)  │
                    │  - synthetic_radar_pdws_v2.csv (+ meta.json) │
                    └───────────────┬───────────────┬──────────────┘
                                    │                │
                     Classifier pipeline      Deinterleaving pipeline
                                    │                │
              ┌─────────────────────┴───┐   ┌────────┴─────────────┐
              │  Training page           │   │  Dataset page's       │
              │  -> DualHeadEWModel      │   │  "Run Deinterleaving  │
              │  -> QdrantEdgeEngine     │   │  Analysis" button      │
              │  -> SQLite run history   │   │  (classical / deep-   │
              └───────────┬───────────────┘   │  metric embedding)   │
                          │                    └───────────────────────┘
        ┌─────────────────┼──────────────────────┐
        │                 │                       │
   Testing page      Inference page        Live Simulation
   (re-evaluate      (single manual        (replay either dataset,
   trained model)     PDW -> verdict)       OR a live, continuous,
                                             unlabeled feed)
```

The classifier and the deinterleaver are **two entirely separate models/pipelines** that
happen to share the Dataset page. The classifier never sees the Interleaved Dataset; the
deinterleaver's embedding network never sees the Classification Dataset.

---

## 3. Technical stack

- **Backend**: FastAPI + Uvicorn (`--reload` dev server), vanilla Python — no ORM, no task queue.
- **Frontend**: Vanilla HTML/JS (no framework/build step), Chart.js (CDN) for all charts.
- **ML**: PyTorch (model + training loop), scikit-learn (KMeans, silhouette, PCA, DBSCAN), ONNX Runtime (portable inference).
- **Vector DB**: `qdrant-client` in **embedded/local mode** (`QdrantClient(path=...)`) — a real Qdrant instance persisted to disk, no server process.
- **Persistence**: SQLite (stdlib `sqlite3`) for training/test run history; flat CSV/JSON files for datasets.
- **No GPU dependency anywhere in the running app** — CPU-only PyTorch and ONNX Runtime.

---

## 4. Model architecture — `DualHeadEWModel`

```
PDW input (6-dim, normalized) -> Backbone -> ┬─> Classifier Head  -> softmax probs (num_classes)
                                              └─> Metric Head      -> L2-normalized embedding (128-dim)
```

**Input features** (`src/data/generator.py::normalize_pdw`): carrier frequency, pulse width,
PRI, RSSI, time-of-arrival, rise time — each independently scaled into a roughly `[0,1]`
range before entering the network.

**Backbone options** (`src/models/backbones.py`, selectable per training run):
- `cnn1d` — two 1D-conv layers over the 6-dim feature vector.
- `transformer` — a 2-layer Transformer encoder (4 heads, d_model=32) treating the 6 features as a length-6 sequence.
- `hybrid` (default) — both of the above in parallel, concatenated and projected to the final embedding — validated against a 2024 published "1D-CNN-Transformer for Radar Emitter Identification" design built for the same edge-deployment reasons.

**Two output heads**:
- **Classifier head** — a linear layer + softmax over `num_classes` (a real hyperparameter of whatever dataset was generated on the Dataset page, not hardcoded — the model is rebuilt to match at the moment Train Model is clicked, never before).
- **Metric head** — a small MLP projecting to a 128-dim, L2-normalized embedding, trained via in-batch triplet loss (same-class pulses pulled together, different-class pushed apart) using whichever distance metric (`cosine` / `euclidean` / `dot`) the Qdrant collection is configured with.

Total training loss per batch: `cross_entropy(known rows) + triplet_loss + rpl_loss_weight * (RPL auxiliary loss)` (see §6).

**Known limitation**: `config.yaml`'s `model.dropout` value is currently unused — no `nn.Dropout` layer exists in any backbone. It has no effect on training; treat it as reserved, not implemented.

---

## 5. Open-set recognition — two independently-scored methods

Every training run reports **both** of the following, computed from the same embeddings, so
they're directly comparable:

### 5.1 Embedding-centroid distance threshold (the method the live system actually acts on)
- After training, each known class's embeddings are clustered into up to **3 sub-centroids**
  via K-means (not a single mean) — `src/training/trainer.py::_class_sub_centroids`.
- **Why sub-centroids, not one mean per class**: classes now genuinely span multiple
  correlated operating modes (§7.3) with different PW/PRI/frequency profiles per mode. A
  single mean embedding can land in the empty space *between* a class's real sub-clusters,
  making genuine same-class samples look falsely anomalous. K-means sub-centroids fix this —
  verified directly (a 12-known-class run produced exactly 36 Qdrant points, 3 per class).
- These sub-centroids are upserted into Qdrant (`QdrantEdgeEngine.upsert_prototypes`) — one
  point per sub-centroid, all sharing that class's label in their payload. `search_nearest`
  needed **no changes** to benefit: it already just returns whichever point is nearest,
  regardless of how many points share a label.
- **Live decision logic** (`run_inference_pipeline` in `server.py`): combines classifier
  softmax confidence with Qdrant distance-to-nearest-prototype into one of three verdicts —
  `VERIFIED_KNOWN_TARGET` (green), `UNKNOWN_THREAT_ANOMALY` (orange — confident classifier but
  far from any prototype, a possible novel/spoofed emitter), `OOD_NOISE_REJECTED` (red — low
  confidence regardless of distance).

### 5.2 Reciprocal Points Learning / ARPL (`src/models/open_set.py`)
A second, independently-trained open-set method, reported as `open_set_auroc_rpl` alongside
the threshold method's `open_set_auroc` — never replacing it.

- **RPL** (Chen et al., ECCV 2020): instead of representing a class by what it IS (a
  centroid), learns a *reciprocal point* per class representing what it ISN'T. Distance to a
  class's reciprocal point IS that class's logit (farther = more confident) — trained via
  cross-entropy on those logits. Open-set score = the confidence gap between the top-1 and
  top-2 farthest reciprocal points (a self-normalizing quantity, chosen after an earlier
  radius-based margin-loss version was found to directly contradict the classification
  objective — caught by insisting on empirical validation before trusting the metric).
- **Full ARPL** (Chen et al., TPAMI 2021, opt-in via `rpl_adversarial`): adds a
  `ConfusingSampleGenerator` — a small generator network adversarially trained to synthesize
  embeddings that fool the RPL head into confident (large-gap) acceptance, while the RPL head
  is simultaneously trained to keep such synthesized samples ambiguous (small gap). A genuine
  min-max adversarial game, played directly in embedding space (no image decoder needed, since
  this is tabular PDW data, not vision).
- Toggle in Training page's "Reciprocal Points (RPL / ARPL)" panel: **RPL Loss Weight**
  (default 0.1) and **Enable Adversarial Training (ARPL)** (off by default — retrain with it
  on for a genuine RPL-vs-ARPL comparison, both visible side by side in Run History).

---

## 6. Dataset generation — two independent pipelines

### 6.1 Classification Dataset (`src/training/trainer.py::build_test_dataset_rows`)
Generated from the Dataset page's blue **Classifier Pipeline** panel. The *only* dataset the
classifier trains/tests on.

- **Real hyperparameters, not hardcoded**: Number of Classes, Holdout Classes, Samples per
  Known Class, Noise Percentage, and 10 population-range fields (frequency/PW/PRI/RSSI/rise) —
  all set per generation, persisted to `classification_dataset_meta.json`.
- **Open-set protocol**: `Holdout Classes` are entirely excluded from training and evaluated
  only as "unknown" — the standard open-set recognition benchmark methodology, not just noisy
  known-class samples.
- **Noise injection**: additionally mixes in unstructured OOD noise pulses (label
  `UNKNOWN_NOISE_OOD`, distinct from holdout classes' `UNKNOWN_HOLDOUT`) — real intercepted
  environments contain both novel-but-structured emitter types and outright junk signals.
- **Correlated operating modes** (`src/data/operating_modes.py`): each sample is drawn under
  one of 4 modes (search/acquisition/track/illumination) with frequency/PW/PRI shifting
  *together* (e.g. track = higher frequency + narrower pulse + shorter PRI), not
  independently — matching real multifunction radar behavior and 2025 published simulation
  methodology.
- **PRI modulation patterns** (`src/data/pri_patterns.py`): each class is assigned one of the
  six standard radar PRI types (constant / jittered / staggered / sliding / wobulated /
  dwell-and-switch) — previously the generator only ever produced "jittered" regardless of
  what it claimed. Classes are now distinguishable by the *shape* of their PRI distribution,
  not just its mean — the literature-backed discriminant PRI-transform deinterleaving
  (CDIF/SDIF) is built around.
- **OOD realism**: unknown/noise samples deliberately overlap the low end of known parameter
  ranges rather than living in a disjoint band — a trivially-separable-by-frequency-alone open
  set understates real difficulty.
- Ground-truth `pri_pattern` and `operating_mode` columns are included in the CSV for
  inspection/analytics — never used as model input features, only for diagnostics.

### 6.2 Interleaved Dataset (`src/data/dataset_export.py::generate_interleaved_pdws`)
Generated from the Dataset page's purple **Deinterleaving Pipeline** panel. Used *only* by the
deinterleaving buttons and Live Simulation's dataset-replay mode.

- Simulates `num_emitters` physical emitters assigned across `num_classes` distinct types
  (when classes < emitters, multiple physical units share a type — the real source of
  deinterleaving difficulty, since telling them apart requires subtler cues than frequency
  alone).
- Each emitter's PRI follows its class's assigned pattern as a **genuine temporal sequence**
  (`generate_pri_sequence`), not just a marginal distribution — staggered cycles through fixed
  positions in order, sliding ramps continuously, dwell-and-switch holds long blocks.
- **Real mode-switching over time**: each emitter dwells in one operating mode for 15-60
  pulses (`sample_dwell`), then transitions (`transition_mode` — escalation-biased 50%, de-
  escalation 30%, random re-scan 20%) — a genuine, observable "war mode" search → track →
  lock-on progression, verified directly (a 200ms simulated emitter showed a real
  `search → acquisition → track → illumination → track → ...` sequence with measurably
  different frequency/PW per mode).
- Output includes `pri_pattern`, `operating_mode`, and the hidden ground-truth
  `emitter_instance_id`/`emitter_class` (used only for scoring deinterleaving results, never
  fed to the clustering algorithm itself).

---

## 7. Deinterleaving — two comparable methods

Both read the Interleaved Dataset and are scored identically (Adjusted Rand Index against the
hidden `emitter_instance_id`), so results are directly comparable on the same data.

- **Classical** (`src/processing/deinterleaver.py`): DBSCAN over standardized
  `[frequency, pulse width, angle-of-arrival]` — a feature-space clustering baseline (not a
  full PRI-transform/CDIF-SDIF deinterleaver).
- **Deep Metric Learning** (`src/processing/deep_deinterleaver.py`): trains a small MLP
  embedding network via triplet loss on the dataset's own ground truth (only usable on
  synthetic data with known `emitter_instance_id` — real captures with no ground truth can
  only use Classical), then runs the same DBSCAN in the *learned* embedding space instead of
  raw features — mirroring a March 2025 published transformer-based approach, using a
  lightweight MLP instead of a full transformer to stay CPU-fast.
- Verified result on a mode-switching-enabled scenario: Classical ARI 0.36 vs. Deep Metric
  1.0 — genuine PRI/mode-driven realism actually made the gap *larger*, showing the learned
  approach's advantage most where classical clustering is genuinely weaker.

---

## 8. Live Simulation — three replay modes

Redesigned around real EW operator-console research (AN/SLQ-32(V) shipboard console human-
factors studies; ISA-101 real-time display hierarchy) — a scrolling **Live Threat Log** (last
30 detections) and situational-awareness stat tiles are the primary view; the classifier
confidence chart and Qdrant distance trend are secondary/diagnostic, not competing for
attention with the actual verdict.

| Mode | Source | Ground truth shown? |
|---|---|---|
| Interleaved Dataset | Replays the Interleaved Dataset file in real ToA order | No — its labels live in an unrelated space from the classifier's trained classes |
| Classification Dataset | Replays the Classification Dataset file (shuffled) | **Yes** — real `Correct ✅ / Incorrect ❌` per pulse and a live accuracy tile |
| Live Feed | Generates pulses continuously, on the fly — nothing read from or written to disk | No, by design — mirrors genuine real-time ESM operation where no ground truth ever exists |

**Live Feed** specifics: hard-capped at 15 minutes server-side regardless of what's requested;
configurable via the **Advanced: Population Ranges** modal (population size, session
duration, all 10 physical ranges); maintains up to 5 persistent simulated emitter "tracks"
with real, evolving operating-mode state (so an escalating emitter is something you can
actually watch happen), while never sending class/mode/track identity to the client.

**Known caveat on "Classification Dataset" replay**: ~80% of that file was actually used to
train the currently-loaded model (the internal train/val split happens inside training, not
before), so high accuracy there partly reflects memorization, not proven generalization. For a
genuinely held-out check, use Testing's "auto-generate fresh sample" instead.

---

## 9. Analytics page

One-shot EDA endpoint (`/api/analytics`) computing everything at once per dataset selection,
so the page's plot buttons just toggle between already-fetched views:

- **Overview**: summary statistics (pandas-`describe()`-style), Pearson correlation heatmap, class balance + diversity (Shannon entropy, imbalance ratio).
- **Radar/ESM analysis**: frequency band breakdown (L/S/C/X/Ku/K/Ka), PRI pattern distribution, and a **Pulse Train Timeline** (ToA vs. frequency scatter) — the classic real-time ESM console view, for the interleaved dataset only.
- **Class Separability**: 2D PCA of the *raw physical features* vs. 2D PCA of the model's *actual learned embeddings* for the same rows, each with a silhouette score. On real project data this showed raw silhouette ≈ −0.05 (barely separated) vs. embedding silhouette ≈ 0.70 (well-separated) — a direct, visual, quantitative answer to "did training actually work," not just an accuracy number.
- **Distributions/Box Plots**: per-feature histograms and per-class Tukey boxplots, including derived **Duty Cycle** and **PRF (Pulse Repetition Frequency)** as first-class features.
- **Scatter Explorer**: any two features plotted against each other, colored by class.

---

## 10. Persistence model

| What | Where | Survives restart? |
|---|---|---|
| Trained model weights | `data/models/current_model_weights.pt` | Yes |
| ONNX export | `data/models/current_model.onnx` | Yes |
| Qdrant prototypes (multi-centroid) | `data/qdrant_storage/` (embedded Qdrant) | Yes |
| Training run history (full config + metrics) | `data/training_history.db` (SQLite) | Yes |
| Test run history | same SQLite DB, `test_runs` table | Yes |
| Classification Dataset + meta | `data/classification_datasets/` | Yes (until regenerated) |
| Interleaved Dataset + meta | `data/generated/` | Yes (until regenerated) |
| Auto-generated test sample | `data/auto_test_samples/` | Yes (overwritten each auto-generate) |
| Live Feed pulses | **Nowhere — never written to disk** | N/A by design |

At server startup, the last training run's full config (backbone, distance metric, num_classes,
Qdrant indexing, RPL settings) is restored from SQLite *before* the model/Qdrant client are
constructed, so a restart never mismatches a persisted model's actual architecture.

**Apply Configuration never destroys a trained model**: only `backbone_type` changes require a
different architecture, and that rebuild is deliberately deferred to the exact moment Train
Model is clicked — not the moment you pick a new backbone in the dropdown. Every other setting
(thresholds, distance metric, Qdrant indexing, RPL knobs) applies immediately without
retraining.

---

## 11. API reference

| Method | Route | Purpose |
|---|---|---|
| GET | `/`, `/validation`, `/inference`, `/dataset`, `/analytics`, `/live`, `/help` | Page routes |
| GET / POST | `/api/config` | Read / update live hyperparameters (backbone deferred, see §10) |
| POST | `/api/train` | Start a background training run against the Classification Dataset |
| GET | `/api/train/status`, `/api/train/history` | Poll progress / list past runs |
| POST | `/api/dataset/generate_classification` | Generate the Classification Dataset |
| POST | `/api/dataset/generate` | Generate the Interleaved Dataset |
| POST | `/api/dataset/upload` | Upload an interleaved-format file |
| POST | `/api/test/upload` | Upload a classification-format file |
| GET | `/api/dataset/inspect` | Lightweight chart data for the Dataset page |
| GET | `/api/analytics` | Full EDA payload for the Analytics page (§9) |
| POST | `/api/test` | Evaluate the trained model (uploaded / generated / auto-fresh-sample) |
| GET | `/api/test/history`, `/api/test/sample` | Past test runs / download the last auto-generated sample |
| POST | `/api/infer` | Single manual PDW → live verdict |
| POST | `/api/deinterleave` | Run Classical or Deep Metric deinterleaving |
| WS | `/ws/telemetry?source=interleaved\|classification\|live_feed` | Live Simulation's three replay modes (§8) |

---

## 12. Project structure

```
src/
  api/server.py            FastAPI app — all routes, config state, inference pipeline
  data/
    generator.py            Classification-dataset per-pulse sampling, PDW normalization
    dataset_export.py        Interleaved-dataset generation (mode-switching pulse trains)
    operating_modes.py        Shared search/acquisition/track/illumination mode logic
    pri_patterns.py            Six standard PRI modulation types (marginal + temporal)
    replay.py                   Dataset-replay adapters for Live Simulation
  models/
    backbones.py              CNN1D / Transformer / Hybrid backbones + DualHeadEWModel
    open_set.py                 Reciprocal Points Learning + adversarial (ARPL) generator
  training/trainer.py        Dataset building, training loop, evaluation, AUROC
  processing/
    deinterleaver.py          Classical DBSCAN deinterleaving
    deep_deinterleaver.py       Deep-metric-learning deinterleaving
  vector_db/qdrant_engine.py  Embedded Qdrant wrapper (multi-centroid prototypes)
  storage/training_store.py  SQLite training/test run history
ui/                         One HTML file per page (vanilla JS, Chart.js via CDN)
config/config.yaml          Startup defaults (overridden by the last real training run)
data/                       All generated/persisted artifacts (see §10)
```

---

## 13. Known limitations (documented deliberately, not hidden)

- **Qdrant embedded mode**: real SDK calls and config (HNSW, quantization), but embedded mode
  doesn't guarantee the same ANN performance/concurrency behavior as a full Qdrant server —
  fine for a single-process research demo, not a production deployment claim.
- **ARPL scope**: implements RPL's core mechanism plus a genuine adversarial generator game in
  embedding space; the original vision-domain ARPL paper's image-feature decoder isn't
  reproduced (no natural equivalent for tabular PDW data).
- **`model.dropout` config value is currently inert** — no dropout layer exists in any backbone.
- **Deep-metric deinterleaving** only works on synthetic data with known ground truth; real
  captures with no `emitter_instance_id` can only use the Classical method.
- **Live Simulation's Classification Dataset replay** partially replays data the model was
  trained on (§8) — not a clean held-out evaluation.
- No TensorRT / GPU support anywhere by design — CPU-only PyTorch + ONNX Runtime throughout.

---

## 14. Setup & run

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

Opens `http://localhost:8000` automatically. First run: go to **Dataset** and generate a
Classification Dataset, then go to **Training** and click Train Model — everything else
(Testing, Inference, Analytics, Live Simulation) depends on that first trained model existing.
