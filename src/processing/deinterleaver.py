import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


def deinterleave(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
    """Reconstructs per-emitter pulse tracks from a TOA-interleaved PDW stream.

    Clusters pulses in standardized [frequency_mhz, pw_ns, aoa_deg] feature space with
    DBSCAN — the classical deinterleaving discriminants (frequency, pulse width, angle of
    arrival). This is a feature-space clustering baseline, not a PRI-transform (CDIF/SDIF)
    deinterleaver; it's a legitimate simpler approach appropriate for this system's scope,
    not a claim of matching full classical PRI-transform techniques.

    Ground truth `emitter_instance_id` (if present, e.g. from generate_interleaved_pdws) is
    used only to score the result via Adjusted Rand Index, never used by the clustering
    itself — the algorithm is genuinely unsupervised.
    """
    feature_cols = ["frequency_mhz", "pw_ns", "aoa_deg"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for deinterleaving: {missing}")

    features = StandardScaler().fit_transform(df[feature_cols].to_numpy())
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)

    predicted_tracks = int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))
    noise_pulses = int(np.sum(cluster_labels == -1))

    result = {
        "total_pulses": int(len(df)),
        "predicted_tracks": predicted_tracks,
        "noise_pulses": noise_pulses,
        "cluster_labels": cluster_labels.tolist(),
    }

    if "emitter_instance_id" in df.columns:
        true_labels = df["emitter_instance_id"].to_numpy()
        result["true_emitters"] = int(df["emitter_instance_id"].nunique())
        result["adjusted_rand_index"] = round(float(adjusted_rand_score(true_labels, cluster_labels)), 4)
    else:
        result["true_emitters"] = None
        result["adjusted_rand_index"] = None

    return result
