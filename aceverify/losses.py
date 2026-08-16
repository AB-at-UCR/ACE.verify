"""Training objective for cross-generator generalization.

Plain BCE lets the classifier latch onto whichever artifact the training
generator happens to leave behind, which is exactly what fails on unseen
generators. Three terms push back:

* label smoothing -- caps the logit magnitude so the model cannot become
  arbitrarily confident about a single generator's fingerprint;
* supervised contrastive -- shapes the embedding so all fakes cluster together
  regardless of how they were produced, instead of only being linearly separable;
* artifact sparsity -- forces the per-patch artifact map to be localized on fakes
  and silent on reals, rather than firing on global colour/compression statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """SupCon (Khosla et al., 2020) over an L2-normalized embedding."""
    features = F.normalize(embeddings.float(), dim=-1)
    labels = labels.view(-1)
    batch = features.shape[0]
    if batch < 2:
        return features.new_zeros(())

    similarity = features @ features.t() / temperature
    # Subtracting the row max keeps exp() finite; it cancels in the log-softmax.
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(batch, dtype=torch.bool, device=features.device)
    positive = (labels[:, None] == labels[None, :]) & ~self_mask

    exp_sim = torch.exp(similarity).masked_fill(self_mask, 0.0)
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_zeros(())

    mean_log_prob = (log_prob * positive).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob.mean()


def artifact_sparsity_loss(
    patch_logits: torch.Tensor, labels: torch.Tensor, top_k: float = 0.1
) -> torch.Tensor:
    """Boundary-aware regularizer on the per-patch artifact map.

    ``patch_logits`` is ``[B, T, P]``. Fakes should light up a small number of
    patches strongly; reals should stay near zero everywhere.
    """
    scores = patch_logits.float().flatten(1)
    labels = labels.view(-1, 1).float()

    k = max(1, int(scores.shape[1] * top_k))
    top = scores.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    detection = F.binary_cross_entropy_with_logits(top, labels)

    # Penalize a diffuse response on fakes: a low-entropy patch distribution means
    # the evidence is concentrated on a seam rather than spread over the whole face.
    probs = torch.softmax(scores, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1, keepdim=True)
    normalized_entropy = entropy / torch.log(
        torch.tensor(float(scores.shape[1]), device=scores.device)
    )
    concentration = (labels * normalized_entropy).sum() / labels.sum().clamp(min=1.0)

    quiet = ((1.0 - labels) * scores.abs().mean(dim=1, keepdim=True)).sum() / (
        1.0 - labels
    ).sum().clamp(min=1.0)

    return detection + 0.1 * concentration + 0.1 * quiet


class DeepfakeCriterion(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.05,
        contrastive_weight: float = 0.2,
        artifact_weight: float = 0.1,
        temperature: float = 0.1,
        pos_weight=None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        self.artifact_weight = artifact_weight
        self.temperature = temperature
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))

    def forward(self, logits, labels, aux=None):
        logits = logits.float()
        labels = labels.float().view_as(logits)

        targets = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        classification = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

        parts = {"bce": classification.detach()}
        total = classification

        if aux:
            embedding = aux.get("embedding")
            if self.contrastive_weight > 0 and embedding is not None:
                contrastive = supervised_contrastive_loss(embedding, labels, self.temperature)
                total = total + self.contrastive_weight * contrastive
                parts["supcon"] = contrastive.detach()

            patch_logits = aux.get("patch_logits")
            if self.artifact_weight > 0 and patch_logits is not None:
                artifact = artifact_sparsity_loss(patch_logits, labels)
                total = total + self.artifact_weight * artifact
                parts["artifact"] = artifact.detach()

        return total, parts
