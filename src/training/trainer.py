import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

from src.data.generator import build_emitter_library, sample_known_pulse, sample_ood_pulse, normalize_pdw
from src.models.open_set import ReciprocalPointsHead, ConfusingSampleGenerator, adversarial_rejection_loss, rpl_auroc

DEFAULT_RPL_LOSS_WEIGHT = 0.1  # keeps the auxiliary RPL objective from dominating the primary
                               # classifier + triplet training it's being compared against
ADV_MARGIN = 0.1  # how far past its learned radius a confusing sample must be pushed


def _select_holdout_ids(num_classes: int, num_holdout_classes: int, dataset_seed: int) -> List[int]:
    """Deterministically picks which classes are held out from a single seed, so the exact
    same known/unknown split can be reconstructed later (e.g. by /api/test) just by knowing
    dataset_seed + num_holdout_classes — no separate list needs to be stored."""
    if num_holdout_classes <= 0:
        return []
    rng = np.random.RandomState(dataset_seed)
    return sorted(rng.choice(num_classes, size=num_holdout_classes, replace=False).tolist())


def build_test_dataset_rows(
    num_classes: int,
    num_holdout_classes: int,
    samples_per_class: int = 100,
    dataset_seed: int = None,
    ranges: Dict[str, Tuple[float, float]] = None,
    noise_pct: float = 0.0,
) -> Dict[str, Any]:
    """Builds a downloadable, human-editable classification dataset: raw physical PDW units
    (matching the Inference page's fields) plus ground-truth labels — the one canonical
    dataset both training and testing consume, generated only from the Dataset page.
    `true_class_id` is -1 for holdout/unknown rows — evaluated as open-set anomalies, never
    seen during training.

    `noise_pct` additionally injects unstructured OOD noise pulses (also true_class_id=-1,
    but labeled UNKNOWN_NOISE_OOD rather than UNKNOWN_HOLDOUT) alongside the holdout classes —
    real-world intercepts include both genuinely novel emitter types AND unstructured
    junk/noise, so the open-set evaluation pool models both, not just held-out classes.
    Noise rows never enter training (load_classification_csv_for_training routes every
    true_class_id=-1 row to the openset-only evaluation pool), so this only affects how
    realistic/thorough the open-set evaluation is, not what the classifier trains on.

    Pass the exact `dataset_seed` (and `ranges`) a previous run used to regenerate a dataset
    matching that run's actual population — otherwise this draws its own fresh random one.

    Returns `{"rows": [...], "dataset_seed": int, "known_class_ids": [...],
    "holdout_class_ids": [...]}` — the seed and split are needed by the caller to persist a
    meta.json alongside the CSV, so training/testing can reconstruct the exact same split.
    """
    if dataset_seed is None:
        dataset_seed = int(np.random.randint(1_000_000))
    ranges = ranges or {}
    holdout_class_ids = _select_holdout_ids(num_classes, num_holdout_classes, dataset_seed)

    rng = np.random.RandomState(dataset_seed)
    library = build_emitter_library(num_classes, seed=dataset_seed, **ranges)
    known_ids = [c for c in range(num_classes) if c not in holdout_class_ids]
    rows = []

    for c in known_ids:
        for _ in range(samples_per_class):
            s = sample_known_pulse(library[c], rng)
            rows.append({
                "carrier_freq_mhz": round(float(s["freq"]), 2),
                "pulse_width_us": round(float(s["pw"]), 3),
                "pri_us": round(float(s["pri"]), 2),
                "rssi_dbm": round(float(s["rssi"]), 1),
                "rise_time_ns": round(float(s["rise"]), 1),
                "true_class_id": c,
                "true_label": f"RADAR_FAMILY_{c:02d}",
                "pri_pattern": str(library[c]["pri_pattern"]),
                "operating_mode": s["mode"],
            })

    for c in holdout_class_ids:
        for _ in range(max(1, samples_per_class // 4)):
            s = sample_known_pulse(library[c], rng)
            rows.append({
                "carrier_freq_mhz": round(float(s["freq"]), 2),
                "pulse_width_us": round(float(s["pw"]), 3),
                "pri_us": round(float(s["pri"]), 2),
                "rssi_dbm": round(float(s["rssi"]), 1),
                "rise_time_ns": round(float(s["rise"]), 1),
                "true_class_id": -1,
                "true_label": "UNKNOWN_HOLDOUT",
                "pri_pattern": str(library[c]["pri_pattern"]),
                "operating_mode": s["mode"],
            })

    n_noise = int(len(known_ids) * samples_per_class * noise_pct)
    for _ in range(n_noise):
        s = sample_ood_pulse(rng)
        rows.append({
            "carrier_freq_mhz": round(float(s["freq"]), 2),
            "pulse_width_us": round(float(s["pw"]), 3),
            "pri_us": round(float(s["pri"]), 2),
            "rssi_dbm": round(float(s["rssi"]), 1),
            "rise_time_ns": round(float(s["rise"]), 1),
            "true_class_id": -1,
            "true_label": "UNKNOWN_NOISE_OOD",
            "pri_pattern": "n/a (unstructured noise)",
            "operating_mode": "n/a (unstructured noise)",
        })

    return {
        "rows": rows,
        "dataset_seed": dataset_seed,
        "known_class_ids": known_ids,
        "holdout_class_ids": holdout_class_ids,
    }


def load_classification_csv_for_training(
    df,
    holdout_class_ids: List[int],
    val_fraction: float = 0.2,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """Converts an already-loaded classification-dataset DataFrame (raw physical PDW units +
    true_class_id, as produced by the Dataset page's Generate Classification Dataset) into the
    same dict shape train_model/evaluate_model already consume. Training no longer generates
    its own data in-memory — it trains on whatever the Dataset page produced, split 80/20 into
    train/val per known row; true_class_id == -1 rows are the held-out open-set pool.
    """
    rng = np.random.RandomState(split_seed)
    vectors = np.stack([
        normalize_pdw(r["carrier_freq_mhz"], r["pulse_width_us"], r["pri_us"], r["rise_time_ns"])
        for r in df.to_dict("records")
    ]).astype(np.float32)
    labels = df["true_class_id"].astype(int).to_numpy()

    known_mask = labels >= 0
    known_idx = np.where(known_mask)[0]
    rng.shuffle(known_idx)
    n_val = max(1, int(len(known_idx) * val_fraction))
    val_idx, train_idx = known_idx[:n_val], known_idx[n_val:]
    openset_idx = np.where(~known_mask)[0]

    known_class_ids = sorted(set(labels[known_mask].tolist()))
    class_frequencies_mhz = {
        c: float(df.loc[labels == c, "carrier_freq_mhz"].mean()) for c in known_class_ids
    }

    return {
        "train_x": vectors[train_idx], "train_y": labels[train_idx],
        "val_x": vectors[val_idx], "val_y": labels[val_idx],
        "openset_x": vectors[openset_idx], "openset_y": labels[openset_idx],
        "known_class_ids": known_class_ids,
        "holdout_class_ids": holdout_class_ids,
        "class_frequencies_mhz": class_frequencies_mhz,
    }


def build_distance_fn(metric: str):
    """Builds the embedding-distance function matching the configured Qdrant metric, so the
    metric head's training geometry matches what QdrantEdgeEngine searches with. Shared by
    train_model and any standalone evaluation (e.g. the /api/test endpoint)."""
    if metric == "euclidean":
        return lambda x, y: nn.functional.pairwise_distance(x, y)
    return lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y)


def _in_batch_triplets(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mines (anchor, positive, negative) index triplets from a batch: anchor/positive share
    a label, negative doesn't. Skips OOD-labeled (-1) rows as anchors since they have no
    same-class positive by construction."""
    anchors, positives, negatives = [], [], []
    labels_np = labels.numpy()
    for i, label in enumerate(labels_np):
        if label < 0:
            continue
        same = np.where(labels_np == label)[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        diff = np.where(labels_np != label)[0]
        if len(diff) == 0:
            continue
        anchors.append(i)
        positives.append(np.random.choice(same))
        negatives.append(np.random.choice(diff))
    idx = (torch.tensor(anchors, dtype=torch.long),
           torch.tensor(positives, dtype=torch.long),
           torch.tensor(negatives, dtype=torch.long))
    return idx


def train_model(model: nn.Module, dataset: Dict[str, Any], config: Dict[str, Any],
                 epochs: int = 30, batch_size: int = 64,
                 progress_cb=None) -> Dict[str, Any]:
    """Trains the dual-head model with real losses: cross-entropy on the classifier head
    (known classes only) and in-batch triplet loss on the metric head, using the configured
    distance metric so the embedding geometry matches what QdrantEdgeEngine searches with.
    Returns evaluation metrics computed on held-out data, not asserted numbers.

    `progress_cb(epoch, epochs, avg_loss)` is called after each epoch if provided, so callers
    (e.g. the API layer) can surface real-time training status instead of a black box.
    """
    lr = config["hyperparameters"]["learning_rate"]
    margin = config["hyperparameters"]["triplet_margin"]
    metric = config["hyperparameters"]["distance_metric"]
    embed_dim = config["model"]["embedding_dim"]
    num_classes = config["model"]["num_classes"]
    rpl_loss_weight = config["hyperparameters"].get("rpl_loss_weight", DEFAULT_RPL_LOSS_WEIGHT)
    rpl_adversarial = config["hyperparameters"].get("rpl_adversarial", False)

    distance_fn = build_distance_fn(metric)

    # RPL is trained jointly (same optimizer, same batches) as a genuine second open-set
    # method — its loss shapes the shared embedding space alongside the triplet loss, and its
    # own AUROC is reported next to the existing centroid-threshold AUROC for direct comparison.
    rpl_head = ReciprocalPointsHead(num_classes, embed_dim)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(rpl_head.parameters()), lr=lr)

    # Full ARPL: a generator adversarially trained alongside the classifier — it tries to
    # synthesize embeddings that fool the reciprocal-point classifier into accepting them as a
    # known class, while the classifier (via adversarial_rejection_loss, added to the main
    # optimizer step below) is trained to keep rejecting them. Only constructed/trained when
    # explicitly enabled, since the adversarial game is inherently less stable than RPL alone.
    generator, gen_optimizer = None, None
    if rpl_adversarial:
        generator = ConfusingSampleGenerator(num_classes, embed_dim)
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    ce_loss_fn = nn.CrossEntropyLoss()
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=margin)

    train_x = torch.from_numpy(dataset["train_x"])
    train_y = torch.from_numpy(dataset["train_y"])
    n = train_x.shape[0]

    model.train()
    loss_curve = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = train_x[idx], train_y[idx]
            known_mask = yb >= 0

            # --- main step: model + RPL head, using real data (and, if enabled, rejecting
            # the generator's CURRENT (detached) confusing samples) ---
            optimizer.zero_grad()
            probs, embeddings = model(xb)

            cls_loss = ce_loss_fn(probs[known_mask], yb[known_mask]) if known_mask.any() else torch.tensor(0.0)

            a_idx, p_idx, n_idx = _in_batch_triplets(yb)
            if len(a_idx) > 0:
                trip_loss = triplet_loss_fn(embeddings[a_idx], embeddings[p_idx], embeddings[n_idx])
            else:
                trip_loss = torch.tensor(0.0)

            rpl_dists = rpl_head.distances(embeddings)
            rpl_ce_loss = ce_loss_fn(rpl_head.logits(rpl_dists)[known_mask], yb[known_mask]) \
                if known_mask.any() else torch.tensor(0.0)

            adv_reject_loss = torch.tensor(0.0)
            target_classes = yb[known_mask]
            if rpl_adversarial and len(target_classes) > 0:
                with torch.no_grad():
                    fake_embeddings = generator(target_classes, num_classes)
                adv_reject_loss = adversarial_rejection_loss(rpl_head, fake_embeddings, ADV_MARGIN)

            loss = cls_loss + trip_loss + rpl_loss_weight * (rpl_ce_loss + adv_reject_loss)
            loss.backward()
            optimizer.step()

            # --- generator step: try to fool the CURRENT (just-updated) rpl_head into
            # confidently accepting fresh confusing samples as their target class ---
            if rpl_adversarial and len(target_classes) > 0:
                gen_optimizer.zero_grad()
                fake_embeddings = generator(target_classes, num_classes)
                fake_logits = rpl_head.logits(rpl_head.distances(fake_embeddings))
                gen_loss = ce_loss_fn(fake_logits, target_classes)
                gen_loss.backward()
                gen_optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = round(epoch_loss / max(1, n_batches), 5)
        loss_curve.append(avg_loss)
        if progress_cb is not None:
            progress_cb(epoch + 1, epochs, avg_loss)

    model.eval()
    rpl_head.eval()
    metrics = evaluate_model(model, dataset, distance_fn, rpl_head=rpl_head)
    metrics["loss_curve"] = loss_curve
    metrics["rpl_adversarial"] = rpl_adversarial
    return metrics


def _class_sub_centroids(class_emb: torch.Tensor, max_subclusters: int) -> List[torch.Tensor]:
    """K-means sub-centroids for one class's embeddings, instead of a single mean — a class
    whose true population spans multiple operating modes (search/track/etc., each pulling
    embeddings in a different direction) can have a mean that lands in a low-density gap
    *between* the real sub-clusters, making genuine same-class samples look anomalously far
    from "their own" centroid. Falling back to a single centroid when there's too little data
    to cluster meaningfully (KMeans needs more points than clusters to be well-posed)."""
    n = class_emb.shape[0]
    if n < max_subclusters * 2:
        return [nn.functional.normalize(class_emb.mean(dim=0), dim=0)]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=max_subclusters, n_init=10, random_state=0).fit(class_emb.detach().numpy())
    return [nn.functional.normalize(torch.from_numpy(c.astype(np.float32)), dim=0) for c in km.cluster_centers_]


def evaluate_model(model: nn.Module, dataset: Dict[str, Any], distance_fn,
                    rpl_head: Optional["ReciprocalPointsHead"] = None,
                    max_subclusters: int = 3) -> Dict[str, Any]:
    """Computes closed-set accuracy on known validation classes and an embedding-distance-based
    open-set AUROC against held-out (never-trained-on) classes.

    Each known class contributes up to `max_subclusters` centroids (via K-means), not one mean
    — see `_class_sub_centroids`. The open-set score is still "distance to the single nearest
    known centroid," just drawn from a richer, multi-modal-aware reference set; Qdrant stores
    these the same way (multiple points sharing one class label), so `search_nearest` needs no
    changes at all to benefit from this.

    `rpl_head`, when provided (only by train_model's own final call, right after jointly
    training one), additionally reports `open_set_auroc_rpl` — the Reciprocal Points Learning
    method's own AUROC on the exact same val/openset split, for direct comparison against the
    centroid-threshold `open_set_auroc`. Standalone callers (e.g. /api/test re-evaluating an
    already-trained model against a fresh sample) have no persisted RPL head to reuse, so they
    simply omit it and get only the original metric — RPL is a training-time-only comparison,
    not a second persisted model.
    """
    with torch.no_grad():
        val_x = torch.from_numpy(dataset["val_x"])
        val_y = dataset["val_y"]
        probs, val_emb = model(val_x)
        preds = probs.argmax(dim=-1).numpy()
        known_mask = val_y >= 0
        closed_set_accuracy = float(np.mean(preds[known_mask] == val_y[known_mask])) if known_mask.any() else 0.0

        centroids: Dict[int, List[torch.Tensor]] = {}
        val_y_t = torch.from_numpy(val_y)
        for c in dataset["known_class_ids"]:
            class_emb = val_emb[val_y_t == c]
            if class_emb.shape[0] > 0:
                centroids[c] = _class_sub_centroids(class_emb, max_subclusters)

        all_sub_centroids = [sc for subs in centroids.values() for sc in subs]

        def min_distance_to_centroids(embeddings: torch.Tensor) -> np.ndarray:
            dists = torch.stack([distance_fn(embeddings, sc.unsqueeze(0).expand_as(embeddings))
                                  for sc in all_sub_centroids], dim=1)
            return dists.min(dim=1).values.numpy()

        auroc = None
        auroc_rpl = None
        if dataset["openset_x"].shape[0] > 0 and all_sub_centroids:
            openset_x = torch.from_numpy(dataset["openset_x"])
            _, openset_emb = model(openset_x)
            known_scores = min_distance_to_centroids(val_emb[known_mask])
            unknown_scores = min_distance_to_centroids(openset_emb)
            auroc = _auroc(known_scores, unknown_scores)

            if rpl_head is not None:
                rpl_known_scores = rpl_head.ood_scores(val_emb[known_mask])
                rpl_unknown_scores = rpl_head.ood_scores(openset_emb)
                auroc_rpl = rpl_auroc(rpl_known_scores, rpl_unknown_scores)

    return {
        "closed_set_accuracy": round(closed_set_accuracy, 4),
        "open_set_auroc": round(auroc, 4) if auroc is not None else None,
        "open_set_auroc_rpl": round(auroc_rpl, 4) if auroc_rpl is not None else None,
        "num_known_classes": len(dataset["known_class_ids"]),
        "num_holdout_classes": len(dataset["holdout_class_ids"]),
        "centroids": {int(c): [sc.numpy().tolist() for sc in subs] for c, subs in centroids.items()},
    }


def _auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Rank-based AUROC: probability a random unknown sample scores as more anomalous
    (higher distance) than a random known sample. Avoids adding sklearn just for this."""
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[labels == 1].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)
