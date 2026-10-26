# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Sequence

import torch
from torch import nn


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def RMSLELoss(pred, actual, valid_mask):
    # valid_mask = (actual > 0.01) & (pred > 0.01)
    return torch.sqrt(((torch.log(pred + 1) - torch.log(actual + 1)) ** 2)[valid_mask].mean() )

def RMSELoss(y, target, valid_mask):
    return torch.sqrt(torch.mean(((target- y) ** 2)[valid_mask]))

def Rel(pred, gt, valid_mask=None):
    """
    Compute Absolute Relative Error (Abs Rel) for depth prediction.

    Args:
        pred (torch.Tensor): predicted depth map, shape (B, H, W)
        gt (torch.Tensor): ground truth depth map, shape (B, H, W)
        mask (torch.Tensor or None): optional binary mask of valid pixels (B, H, W)

    Returns:
        abs_rel (float): average Abs Rel across batch
    """
    # Ensure predictions and GTs are float
    pred = pred.float()
    gt = gt.float()


    # Compute absolute relative error
    abs_rel = torch.abs(pred - gt) / gt
    abs_rel = abs_rel[valid_mask]

    sq_rel = ((pred - gt)** 2) / gt
    sq_rel = sq_rel[valid_mask]

    # valid_mask = (pred > 0.01) & (gt > 0.01)
    log_diff = torch.log(pred[valid_mask]) - torch.log(gt[valid_mask])

    if log_diff.numel() == 0:
        silog = float('nan')
    else:
        silog = log_diff.pow(2).mean() - log_diff.mean().pow(2)

    return (abs_rel.mean() if abs_rel.numel() > 0 else float('nan'),
            sq_rel.mean() if sq_rel.numel() > 0 else float('nan'),
            silog)




def depth_metrics(outputs: torch.Tensor, targets: torch.Tensor):
    # mask = (targets > 0.01) & (targets < 0.99)
    mask = targets > 0.01
    outputs =  torch.nn.functional.interpolate(outputs,targets.shape[2:], mode='bilinear',
                                                align_corners=False)

    rmse = RMSELoss(outputs, targets, mask)
    rmsle = RMSLELoss(outputs, targets, mask)
    absrel, sqrl, silog = Rel(outputs, targets, mask)

    return rmse, rmsle, absrel, sqrl, silog


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)
