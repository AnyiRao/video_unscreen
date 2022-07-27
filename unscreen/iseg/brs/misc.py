import numpy as np
import torch


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)
    return dims


def _compute_iou(pred_mask, gt_mask, ignore_mask=None, keep_ignore=False):
    if ignore_mask is not None:
        pred_mask = torch.where(ignore_mask, torch.zeros_like(pred_mask),
                                pred_mask)

    reduction_dims = get_dims_with_exclusion(gt_mask.dim(), 0)
    union = torch.mean(
        (pred_mask | gt_mask).float(),
        dim=reduction_dims).detach().cpu().numpy()
    intersection = torch.mean(
        (pred_mask & gt_mask).float(),
        dim=reduction_dims).detach().cpu().numpy()
    nonzero = union > 0

    iou = intersection[nonzero] / union[nonzero]
    if not keep_ignore:
        return iou
    else:
        result = np.full_like(intersection, -1)
        result[nonzero] = iou
        return result
