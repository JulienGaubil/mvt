"""Pointmap metrics: AbsRel error, inlier ratio and multi-view consistency."""

import torch

from jaxtyping import Float
from typing import Any, Dict, List, Sequence

from mvt.geometry import (
    per_view_procrustes_align,
    pointmap_alignment,
    symmetric_mutual_correspondence_masks,
    warp_field,
    warp_pc_to_pixel,
)
from mvt.utils import masked_mean_per_sample


def pointmap_absrel_errors(
    gt_pts_views: Sequence[torch.Tensor],
    pred_pts_views: Sequence[torch.Tensor],
    valid_masks: Sequence[torch.Tensor],
    eps: float = 1e-8,
) -> List[Float[torch.Tensor, "batch"]]:
    """Masked per-sample AbsRel pointmap error of each view."""
    errors = []
    for gt_pts, pred_pts, valid_mask in zip(gt_pts_views, pred_pts_views, valid_masks):
        gt_norm = gt_pts.norm(dim=-1)
        rel_error = (pred_pts - gt_pts).norm(dim=-1) / gt_norm.clip(min=eps)
        errors.append(masked_mean_per_sample(rel_error, valid_mask & (gt_norm > eps)))
    return [error.cpu() for error in errors]


def pointmap_absrel_inlier_ratios(
    gt_pts_views: Sequence[torch.Tensor],
    pred_pts_views: Sequence[torch.Tensor],
    valid_masks: Sequence[torch.Tensor],
    threshold: float = 0.03,
    eps: float = 1e-8,
) -> List[Float[torch.Tensor, "batch"]]:
    """Masked per-sample ratio of pixels with an AbsRel error below `threshold`."""
    ratios = []
    for gt_pts, pred_pts, valid_mask in zip(gt_pts_views, pred_pts_views, valid_masks):
        gt_norm = gt_pts.norm(dim=-1)
        rel_error = (pred_pts - gt_pts).norm(dim=-1) / gt_norm.clip(min=eps)
        inliers = (rel_error < threshold).to(dtype=gt_pts.dtype)
        ratios.append(masked_mean_per_sample(inliers, valid_mask & (gt_norm > eps)))
    return [ratio.cpu() for ratio in ratios]


def pointmap_multiview_errors(
    views: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, torch.Tensor]],
    tolerance_px: int = 5,
) -> List[Float[torch.Tensor, "batch"]]:
    """Multi-view consistency error (e_mv) on co-visible pixels.

    View 0 is paired with every other view, and GT geometry gives their mutual
    pixel correspondences. e_mv is computed like the AbsRel error, but the target
    of each view is the other view's prediction warped to its pixel grid, on
    co-visible pixels only.

    Args:
        views: Per-view GT dictionaries with 3D points, camera poses and
            intrinsics, and valid masks.
        preds: Per-view predictions, with `pts3d` in view 0's frame.
        tolerance_px: Maximum round-trip pixel error of a mutual correspondence.

    Returns:
        Per-view tensors of per-sample errors. Each non-reference view holds its
        error with view 0, and view 0 the mean of its errors with all views.
    """
    reference = views[0]
    reference_errors, other_errors = [], []
    for view_idx in range(1, len(views)):
        other = views[view_idx]

        # Mutual GT correspondences between view 0 and this view.
        reference_to_other = warp_pc_to_pixel(
            reference["pts3d"], other["camera_pose"], other["camera_intrinsics"]
        )
        other_to_reference = warp_pc_to_pixel(
            other["pts3d"], reference["camera_pose"], reference["camera_intrinsics"]
        )
        reference_mask, other_mask = symmetric_mutual_correspondence_masks(
            reference_to_other,
            other_to_reference,
            reference["valid_mask"],
            other["valid_mask"],
            tolerance_px=tolerance_px,
        )
        reference_mask = reference_mask & reference["valid_mask"]
        other_mask = other_mask & other["valid_mask"]

        # Target: the other prediction warped to this view (view 0 frame, so identity).
        reference_pts = preds[0]["pts3d"]
        other_pts = preds[view_idx]["pts3d"]
        identity = torch.eye(
            4, dtype=reference_pts.dtype, device=reference_pts.device
        ).expand(len(reference_pts), 4, 4)
        targets = [
            {
                "pts3d": warp_field(other_pts, reference_to_other, reference_mask),
                "camera_pose": identity,
                "valid_mask": reference_mask,
            },
            {
                "pts3d": warp_field(reference_pts, other_to_reference, other_mask),
                "camera_pose": identity,
                "valid_mask": other_mask,
            },
        ]
        target_pts, pred_pts, masks = pointmap_alignment(
            targets, [preds[0], preds[view_idx]]
        )
        reference_error, other_error = pointmap_absrel_errors(
            target_pts, pred_pts, masks
        )
        reference_errors.append(reference_error)
        other_errors.append(other_error)

    return [torch.stack(reference_errors).nanmean(dim=0), *other_errors]


def evaluate_pointmap_batch(
    views: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, torch.Tensor]],
    procrustes_align: bool = False,
    compute_multiview_error: bool = False,
) -> Dict[str, List[torch.Tensor]]:
    """Evaluate pointmap predictions after average-distance normalization.

    Args:
        views: Per-view GT dictionaries. `camera_pose` is c2w and `pts3d` is in a
            shared world frame.
        preds: Per-view predictions, with `pts3d` in the first-view frame.
        procrustes_align: Whether to also compute metrics after per-view rigid
            alignment.
        compute_multiview_error: Whether to compute the multi-view consistency.

    Returns:
        A mapping from metric names to per-view lists of per-sample metrics.
    """
    views, preds = list(views), list(preds)
    if len(views) != len(preds):
        raise ValueError(
            f"Expected as many views as predictions, got {len(views)} and {len(preds)}"
        )
    if len(views) < 2:
        raise ValueError(f"Expected at least 2 views, got {len(views)}")

    gt_pts_views, pred_pts_views, valid_masks = pointmap_alignment(views, preds)
    metrics = {
        "pointmaps_abs_rel": pointmap_absrel_errors(
            gt_pts_views, pred_pts_views, valid_masks
        ),
        "pointmaps_absrel_inlier_thres_003": pointmap_absrel_inlier_ratios(
            gt_pts_views, pred_pts_views, valid_masks, threshold=0.03
        ),
    }
    if compute_multiview_error:
        metrics["multiview_error"] = pointmap_multiview_errors(views, preds)

    if procrustes_align:
        pred_pts_views_proc = per_view_procrustes_align(
            pred_pts_views, gt_pts_views, valid_masks
        )
        metrics["Procrustes_pointmaps_abs_rel"] = pointmap_absrel_errors(
            gt_pts_views, pred_pts_views_proc, valid_masks
        )
        metrics[
            "Procrustes_pointmaps_absrel_inlier_thres_003"
        ] = pointmap_absrel_inlier_ratios(
            gt_pts_views, pred_pts_views_proc, valid_masks, threshold=0.03
        )
    return metrics
