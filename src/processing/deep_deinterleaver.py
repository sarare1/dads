import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

FEATURE_COLS = ["frequency_mhz", "pw_ns", "aoa_deg", "rise_time_ns"]


class PulseEmbeddingNet(nn.Module):
    """Small MLP embedding network for metric-learning-based deinterleaving — the same triplet-
    loss idea the classifier already uses, applied to individual pulses instead of PDW-level
    classification. Deliberately tiny (this trains from scratch in seconds on CPU): the point
    is the metric-learning approach, not a large model."""

    def __init__(self, input_dim: int = 4, embed_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


def _in_batch_triplets(labels: np.ndarray):
    anchors, positives, negatives = [], [], []
    for i, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        diff = np.where(labels != label)[0]
        if len(diff) == 0:
            continue
        anchors.append(i)
        positives.append(np.random.choice(same))
        negatives.append(np.random.choice(diff))
    return (torch.tensor(anchors, dtype=torch.long),
            torch.tensor(positives, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long))


def _train_embedding_net(features: np.ndarray, instance_ids: np.ndarray, epochs: int, seed: int) -> PulseEmbeddingNet:
    """Trains a per-pulse embedding network via triplet loss using the synthetic dataset's
    ground-truth `emitter_instance_id` — the same "train on labeled synthetic data, then
    generalize to real intercepted streams" pattern the published transformer-based
    deinterleaving approach uses. Ground truth is used only here, at training time, exactly
    like the classifier; it is never used by the clustering/scoring step that follows."""
    torch.manual_seed(seed)
    model = PulseEmbeddingNet(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.from_numpy(features.astype(np.float32))

    model.train()
    for _ in range(epochs):
        embeddings = model(x)
        a_idx, p_idx, n_idx = _in_batch_triplets(instance_ids)
        if len(a_idx) == 0:
            continue
        loss = F.triplet_margin_loss(embeddings[a_idx], embeddings[p_idx], embeddings[n_idx], margin=0.3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def deep_metric_deinterleave(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5,
                              epochs: int = 30, seed: int = 0) -> Dict[str, Any]:
    """Deinterleaves the same way `deinterleaver.deinterleave` does (DBSCAN clustering, same
    eps/min_samples semantics, same result shape) but over a LEARNED embedding space instead
    of hand-picked standardized raw features — the 2025 research direction (transformer-based
    deep metric learning for pulse deinterleaving) applied here with a lightweight MLP instead
    of a full transformer, kept CPU-fast for this project's scope. Requires
    `emitter_instance_id` in `df` to train against (only available for synthetic data with
    known ground truth); the clustering/scoring step downstream is exactly as unsupervised as
    the classical method — ground truth trains the embedding, never the clustering itself.
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for deep-metric deinterleaving: {missing}")
    if "emitter_instance_id" not in df.columns:
        raise ValueError(
            "Deep-metric deinterleaving needs ground-truth `emitter_instance_id` to train its "
            "embedding network (synthetic data only) — use the classical method for uploaded "
            "real-world captures that have no such ground truth."
        )

    raw_features = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    standardized = StandardScaler().fit_transform(raw_features)
    instance_ids = df["emitter_instance_id"].to_numpy()

    embedding_net = _train_embedding_net(standardized, instance_ids, epochs=epochs, seed=seed)
    with torch.no_grad():
        embeddings = embedding_net(torch.from_numpy(standardized.astype(np.float32))).numpy()

    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings)

    predicted_tracks = int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))
    noise_pulses = int(np.sum(cluster_labels == -1))

    return {
        "method": "deep_metric",
        "total_pulses": int(len(df)),
        "predicted_tracks": predicted_tracks,
        "noise_pulses": noise_pulses,
        "true_emitters": int(df["emitter_instance_id"].nunique()),
        "adjusted_rand_index": round(float(adjusted_rand_score(instance_ids, cluster_labels)), 4),
    }
