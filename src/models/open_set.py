import torch
import torch.nn as nn
import torch.nn.functional as F


class ReciprocalPointsHead(nn.Module):
    """Reciprocal Points Learning (RPL) for open-set recognition — Chen et al., "Learning Open
    Set Network with Discriminative Reciprocal Points" (ECCV 2020), the mechanism behind the
    adversarial reciprocal point learning (ARPL) line of work cited in current (2025) radar
    open-set literature. A genuine second open-set method trained alongside the existing
    embedding-centroid-threshold approach, not a replacement for it — every training run
    reports both AUROC numbers so the two are directly comparable.

    Core idea: instead of representing each known class by what it IS (a centroid, as the
    existing threshold method does), RPL learns a "reciprocal point" per class representing
    everything that ISN'T that class. A sample's distance to class k's reciprocal point is its
    classification logit for class k (farther = more confidently class k, since a genuine
    class-k sample should look nothing like class k's "not-k" region).

    Open-set score: the confidence GAP between the farthest (top-1) and second-farthest (top-2)
    reciprocal point distances. A well-separated known sample stands out clearly against one
    specific class's reciprocal point (large gap); an unknown sample doesn't cleanly stand out
    against any single class (small gap, ambiguous across all of them). This is a relative,
    self-normalizing quantity — unlike bounding one class's raw distance with a learned radius
    (an earlier version of this module did that and it directly fought the classification
    objective, which wants that same distance to be large — a real bug caught by checking the
    resulting AUROC empirically, not assumed correct from the formula alone), a margin between
    two distances in the same row can't run away unbounded, so no separate radius parameter
    is needed to keep it well-behaved.
    """

    def __init__(self, num_classes: int, embed_dim: int, gamma: float = 8.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reciprocal_points = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)

    def distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean distance from each embedding to every class's reciprocal point.
        Shape [batch, num_classes]."""
        return torch.cdist(embeddings, self.reciprocal_points, p=2) ** 2

    def logits(self, dists: torch.Tensor) -> torch.Tensor:
        """Farther from o_k => more confidently class k, so distance itself (scaled) is the
        classification logit — the inverted geometry that is RPL's central trick."""
        return self.gamma * dists

    def confidence_gap(self, dists: torch.Tensor) -> torch.Tensor:
        """Top-1 minus top-2 distance per sample — how clearly one class's reciprocal point
        stands out as farthest (confidently known) versus ambiguous across several (unknown)."""
        top2 = dists.topk(2, dim=-1).values
        return top2[:, 0] - top2[:, 1]

    def ood_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Per-sample open-set anomaly score, same convention as the existing centroid-distance
        score elsewhere in this app: larger = more likely unknown."""
        dists = self.distances(embeddings)
        return -self.confidence_gap(dists)


class ConfusingSampleGenerator(nn.Module):
    """The 'A' in ARPL — Adversarial Reciprocal Points Learning (Chen et al., TPAMI 2021),
    the full extension of the RPL mechanism above. A small generator network synthesizes
    embeddings targeted at a known class, adversarially trained to fool the reciprocal-point
    classifier into confidently accepting them as that one class (a large confidence gap),
    while the classifier is simultaneously trained to keep such synthesized samples ambiguous
    (a small gap) — the min-max game that tightens the open-set boundary beyond what training
    on real data alone achieves.

    The original ARPL paper operates on image features with a convolutional generator/decoder;
    this PDW system has no natural image-style decoder to generate through, so the adversarial
    game is played directly in the model's own embedding space (feature-level confusing
    samples) — a faithful, appropriately-scoped adaptation of the same mechanism to tabular
    radar PDW data rather than a claim of reproducing the original vision architecture.
    """

    def __init__(self, num_classes: int, embed_dim: int, noise_dim: int = 16):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, target_classes: torch.Tensor, num_classes: int) -> torch.Tensor:
        noise = torch.randn(target_classes.shape[0], self.noise_dim)
        one_hot = F.one_hot(target_classes, num_classes).float()
        raw = self.net(torch.cat([noise, one_hot], dim=-1))
        return F.normalize(raw, p=2, dim=-1)  # same L2-normalized space the model's embeddings live in


def adversarial_rejection_loss(rpl_head: ReciprocalPointsHead, fake_embeddings: torch.Tensor,
                                margin: float = 0.1) -> torch.Tensor:
    """The classifier's half of the adversarial game: keeps the confidence gap for the
    generator's current confusing samples small (ambiguous / not clearly any one class),
    directly opposing the generator's own objective of making that gap large."""
    dists = rpl_head.distances(fake_embeddings)
    gap = rpl_head.confidence_gap(dists)
    return F.relu(gap - margin).pow(2).mean()


def rpl_auroc(known_scores: torch.Tensor, unknown_scores: torch.Tensor) -> float:
    """Same rank-based AUROC as the existing threshold method's `_auroc`, applied to RPL's
    own anomaly score — kept separate (not imported from trainer.py) so this module has no
    dependency on the training loop, only torch/numpy."""
    import numpy as np
    known = known_scores.detach().numpy()
    unknown = unknown_scores.detach().numpy()
    scores = np.concatenate([known, unknown])
    labels = np.concatenate([np.zeros_like(known), np.ones_like(unknown)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
