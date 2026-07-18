"""STRUCTURED pruning (channels actually removed, not just zeroed) on the head.

Scope intentionally reduced: the head is only ~13% of total latency (measured
earlier), so the gain here is necessarily small — measured anyway, to close the
comparison with real data instead of skipping the technique.
"""
import os
import sys
import copy

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

KEEP_RATIO = 0.5  # keeps 50% of channels per layer, ranked by importance (L1 of the filters)


def _prune_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, keep_ratio: float):
    out_channels = conv.out_channels
    keep = max(1, int(out_channels * keep_ratio))
    importance = conv.weight.data.abs().mean(dim=(1, 2, 3))
    keep_idx = torch.topk(importance, keep).indices.sort().values

    new_conv = nn.Conv2d(conv.in_channels, keep, kernel_size=conv.kernel_size, padding=conv.padding)
    new_conv.weight.data = conv.weight.data[keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx].clone()

    new_bn = nn.BatchNorm2d(keep)
    new_bn.weight.data = bn.weight.data[keep_idx].clone()
    new_bn.bias.data = bn.bias.data[keep_idx].clone()
    new_bn.running_mean = bn.running_mean[keep_idx].clone()
    new_bn.running_var = bn.running_var[keep_idx].clone()

    return new_conv, new_bn, keep_idx


def prune_segmentation_head(head, keep_ratio: float):
    head = copy.deepcopy(head)
    conv1, bn1, relu1, conv2, bn2, relu2 = head.refiner

    new_conv1, new_bn1, keep_idx1 = _prune_conv_bn(conv1, bn1, keep_ratio)

    # conv2 receives conv1's pruned output — its INPUT channels must be sliced first
    conv2_sliced = nn.Conv2d(len(keep_idx1), conv2.out_channels, kernel_size=conv2.kernel_size, padding=conv2.padding)
    conv2_sliced.weight.data = conv2.weight.data[:, keep_idx1].clone()
    if conv2.bias is not None:
        conv2_sliced.bias.data = conv2.bias.data.clone()

    new_conv2, new_bn2, keep_idx2 = _prune_conv_bn(conv2_sliced, bn2, keep_ratio)

    new_classifier = nn.Conv2d(len(keep_idx2), head.classifier.out_channels, kernel_size=1)
    new_classifier.weight.data = head.classifier.weight.data[:, keep_idx2].clone()
    new_classifier.bias.data = head.classifier.bias.data.clone()

    head.refiner = nn.Sequential(new_conv1, new_bn1, nn.ReLU(inplace=True), new_conv2, new_bn2, nn.ReLU(inplace=True))
    head.classifier = new_classifier
    return head


def make_bundle(device):
    backbone = load_backbone(device)
    head, _ = load_head(device)

    head.tissue_head = prune_segmentation_head(head.tissue_head, KEEP_RATIO)
    head.nuclei_head = prune_segmentation_head(head.nuclei_head, KEEP_RATIO)
    head.eval()

    logger.info(f"Structured pruning applied to the head (keep_ratio={KEEP_RATIO})")
    return ModelBundle(backbone, head, device)


def run():
    device = torch.device("cpu")
    bundle = make_bundle(device)
    val_ids = get_val_ids()

    def predict_fn(tensor):
        return bundle.predict(tensor, upscale=True)

    return evaluate_method(predict_fn, f"Structured Pruning — head only (keep={KEEP_RATIO})", val_ids, resolution=1036)


if __name__ == "__main__":
    run()
