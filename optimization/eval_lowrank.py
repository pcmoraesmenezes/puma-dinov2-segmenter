"""Low-rank factorization (SVD) on the backbone's Linear layers — WITHOUT fine-tuning after.

Mechanism: W (out x in) ~= A @ B via truncated SVD (rank r). Reduces parameters/FLOPs
when r < (in*out)/(in+out). Without fine-tuning, accuracy degradation is expected —
that's the result, not a bug.
"""
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "serving"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_model import load_backbone, load_head, ModelBundle
from eval_utils import evaluate_method, get_val_ids
from utils.logger import get_logger

logger = get_logger(__name__)

RANK_RATIO = 0.25  # rank = 25% of min(in_features, out_features)


class LowRankLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        W = linear.weight.data  # (out, in)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        rank = min(rank, S.shape[0])

        A = U[:, :rank] * S[:rank]  # (out, rank)
        B = Vh[:rank, :]  # (rank, in)

        self.first = nn.Linear(linear.in_features, rank, bias=False)
        self.first.weight.data = B.contiguous()

        self.second = nn.Linear(rank, linear.out_features, bias=linear.bias is not None)
        self.second.weight.data = A.contiguous()
        if linear.bias is not None:
            self.second.bias.data = linear.bias.data.clone()

        self.rank = rank
        self.orig_shape = (linear.out_features, linear.in_features)

    def forward(self, x):
        return self.second(self.first(x))


def apply_low_rank(module: nn.Module, rank_ratio: float) -> int:
    """Recursively replaces every nn.Linear with LowRankLinear. Returns how many were replaced."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            rank = max(1, int(min(child.in_features, child.out_features) * rank_ratio))
            setattr(module, name, LowRankLinear(child, rank))
            count += 1
        else:
            count += apply_low_rank(child, rank_ratio)
    return count


def make_bundle(device):
    backbone = load_backbone(device)
    head, _ = load_head(device)

    n_replaced = apply_low_rank(backbone, RANK_RATIO)
    logger.info(f"Low-rank applied: {n_replaced} Linear layers replaced (rank_ratio={RANK_RATIO})")

    return ModelBundle(backbone, head, device)


def run():
    device = torch.device("cpu")
    bundle = make_bundle(device)
    val_ids = get_val_ids()

    def predict_fn(tensor):
        return bundle.predict(tensor, upscale=True)

    return evaluate_method(predict_fn, f"Low-Rank SVD (r={RANK_RATIO})", val_ids, resolution=1036)


if __name__ == "__main__":
    run()
