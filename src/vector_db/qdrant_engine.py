import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    BinaryQuantization, BinaryQuantizationConfig,
)

_METRIC_MAP = {
    "cosine": Distance.COSINE,
    "euclidean": Distance.EUCLID,
    "dot": Distance.DOT,
}

_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "../../data/qdrant_storage")


class QdrantEdgeEngine:
    """
    Wraps the real `qdrant-client` SDK in embedded/local mode (no Docker/server required) —
    genuine collection creation, HNSW config, quantization config, and vector search against
    an on-disk store persisted under data/qdrant_storage/ (so prototypes survive dev-server
    reloads). Before training has ever run, seeds the collection with the same seeded-random
    prototype vectors the original simulation used, so the live dashboard keeps working out
    of the box; `upsert_prototypes` replaces them with real learned class centroids once a
    training run completes.

    Honesty note: local-mode quantization/HNSW are the real SDK calls and config objects from
    the spec, but qdrant-client's embedded engine does not guarantee the same on-disk
    compression/ANN behavior as the full Qdrant server — treat this as "real API, approximated
    runtime," not a claim of server-grade performance.
    """

    def __init__(self, num_classes: int = 20, embed_dim: int = 128, metric: str = "cosine",
                 collection_name: str = "radar_prototypes", hnsw_m: int = 16,
                 hnsw_ef_construct: int = 100, quantization: str = "scalar"):
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.metric = metric
        self.collection_name = collection_name
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.quantization = quantization

        os.makedirs(_STORAGE_DIR, exist_ok=True)
        self.client = QdrantClient(path=_STORAGE_DIR)

        self._attach_or_create()

    def _attach_or_create(self):
        """Reuses an already-persisted collection as-is (which may hold real trained
        centroids from a previous process) if one exists and matches this embed_dim;
        otherwise creates a fresh collection seeded with random placeholder prototypes."""
        if self.client.collection_exists(self.collection_name):
            try:
                info = self.client.get_collection(self.collection_name)
                if info.config.params.vectors.size == self.embed_dim:
                    return  # reuse whatever is already on disk
            except Exception:
                pass
        self._recreate_collection()
        self._seed_random_prototypes()

    def has_trained_prototypes(self) -> bool:
        """Checks whether the currently attached collection holds real learned centroids
        (payload marked trained=True) rather than random placeholders — used at server
        startup to correctly restore the in-memory prototypes_trained flag."""
        points = self.client.scroll(collection_name=self.collection_name, limit=1, with_payload=True)[0]
        return bool(points) and points[0].payload.get("trained", False)

    def _quantization_config(self):
        if self.quantization == "binary":
            return BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
        )

    def _recreate_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embed_dim, distance=_METRIC_MAP[self.metric]),
            hnsw_config=HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
            quantization_config=self._quantization_config(),
        )

    def _seed_random_prototypes(self):
        """Fallback prototypes so search works before any real training has run."""
        rng = np.random.RandomState(100)
        points = []
        self._label_meta = {}
        for c in range(self.num_classes):
            vec = rng.randn(self.embed_dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            label = f"RADAR_FAMILY_{c:02d}"
            payload = {"label": label, "freq_band": None, "trained": False}
            points.append(PointStruct(id=c, vector=vec.tolist(), payload=payload))
            self._label_meta[c] = payload
        self.client.upsert(collection_name=self.collection_name, points=points)

    def upsert_prototypes(self, centroids: Dict[int, list], metadata: Optional[Dict[int, Dict[str, Any]]] = None):
        """Replaces prototypes with real learned class centroids from a training run. Each
        class may contribute multiple sub-centroid points sharing the same class label
        (K-means sub-clusters — see trainer.py's _class_sub_centroids) rather than exactly
        one mean point per class; `search_nearest`'s plain nearest-neighbor lookup needs no
        changes at all to benefit — it already just returns whichever point is closest,
        regardless of how many points share a label. A full recreate (not a partial upsert)
        since a fresh training run always fully replaces the previous prototype set, and the
        number of sub-centroids per class can differ run to run."""
        self._recreate_collection()
        points = []
        point_id = 0
        for class_id, vectors in centroids.items():
            payload = {"label": f"RADAR_FAMILY_{class_id:02d}", "trained": True}
            if metadata and class_id in metadata:
                payload.update(metadata[class_id])
            # Backward-compatible with a caller still passing one flat vector per class.
            vector_list = vectors if (len(vectors) > 0 and isinstance(vectors[0], (list, tuple))) else [vectors]
            for vector in vector_list:
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
                point_id += 1
        self.client.upsert(collection_name=self.collection_name, points=points)

    def update_metric(self, metric: str):
        """Qdrant collections are metric-fixed at creation, so switching distance metrics
        means recreating the collection and re-upserting whatever prototypes currently exist."""
        if metric == self.metric:
            return
        existing = self.client.scroll(collection_name=self.collection_name, limit=self.num_classes,
                                       with_vectors=True, with_payload=True)[0]
        self.metric = metric
        self._recreate_collection()
        if existing:
            points = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in existing]
            self.client.upsert(collection_name=self.collection_name, points=points)
        else:
            self._seed_random_prototypes()

    def update_indexing(self, hnsw_m: int = None, hnsw_ef_construct: int = None, quantization: str = None):
        """Recreates the collection with new HNSW/quantization params, preserving current points.
        No-ops if nothing actually changed, matching update_metric's existing behavior — avoids
        rebuilding the collection on every Apply Configuration click when indexing params
        weren't touched."""
        unchanged = (
            (hnsw_m is None or hnsw_m == self.hnsw_m) and
            (hnsw_ef_construct is None or hnsw_ef_construct == self.hnsw_ef_construct) and
            (quantization is None or quantization == self.quantization)
        )
        if unchanged:
            return
        existing = self.client.scroll(collection_name=self.collection_name, limit=self.num_classes,
                                       with_vectors=True, with_payload=True)[0]
        if hnsw_m is not None:
            self.hnsw_m = hnsw_m
        if hnsw_ef_construct is not None:
            self.hnsw_ef_construct = hnsw_ef_construct
        if quantization is not None:
            self.quantization = quantization
        self._recreate_collection()
        if existing:
            points = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in existing]
            self.client.upsert(collection_name=self.collection_name, points=points)
        else:
            self._seed_random_prototypes()

    def search_nearest(self, query_vector: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Real qdrant-client vector search. Converts the returned similarity score into the
        existing 'distance' semantics (larger = more anomalous) per metric, so the fusion
        logic in server.py that compares against ood_distance_threshold is unaffected."""
        query_vec = query_vector.flatten()
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec.tolist(),
            limit=1,
            with_payload=True,
        )
        if not result.points:
            return "NO_MATCH", float("inf"), {}

        top = result.points[0]
        if self.metric == "euclidean":
            distance = float(top.score)
        else:  # cosine or dot: qdrant score is a similarity, convert to a distance
            distance = 1.0 - float(top.score)

        payload = top.payload or {}
        return payload.get("label", "UNKNOWN"), round(distance, 4), payload
