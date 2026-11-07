"""Training losses: DUSt3R's confidence-weighted pointmap loss and VGGT's losses."""

from __future__ import annotations

import torch
import torch.nn as nn

from copy import deepcopy
from typing import Any, Dict, List, Literal, Sequence, Tuple

from mvt import _thirdp  # noqa: F401
from mvt.geometry import (
    normalize_pointmaps_by_avg_distance,
    normalize_views_vggt,
    pointmap_alignment,
    project_reference_frame_gt,
)
from mvt.utils import stack_views
from dust3r.losses import L21
from training.loss import compute_camera_loss, compute_depth_loss, compute_point_loss

Views = Sequence[Dict[str, Any]]
Preds = Dict[str, List[Dict[str, torch.Tensor]]]


def _check_num_views(views: Views, preds: Sequence) -> None:
    if len(views) != len(preds):
        raise ValueError(
            f"Expected as many views as predictions, got {len(views)} and {len(preds)}"
        )
    if len(views) < 2:
        raise ValueError(f"Expected at least 2 views, got {len(views)}")


class PointmapLoss(nn.Module):
    """DUSt3R's confidence-weighted pointmap loss, for any number of views.

    All views are compared in the frame of view 0, after normalizing GT and
    predictions by their average point norm over all views.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        """Initialize the loss.

        Args:
            alpha: Confidence regularization weight.
        """
        super().__init__()
        self.alpha = alpha
        self.criterion = deepcopy(L21)
        self.criterion.reduction = "none"

    def forward(self, views: Views, preds: Preds) -> torch.Tensor:
        """Compute the loss.

        Args:
            views: Per-view GT dictionaries. `camera_pose` is c2w and `pts3d` is
                in a shared world frame.
            preds: MVT predictions, whose "pointmaps" entry holds per-view `pts3d`
                (in the first-view frame) and `conf` tensors.

        Returns:
            Scalar loss.
        """
        views, preds = list(views), list(preds["pointmaps"])
        _check_num_views(views, preds)
        gt_pts_views, pred_pts_views, valid_masks = pointmap_alignment(views, preds)

        total_loss = pred_pts_views[0].new_zeros(())
        for gt_pts, pred_pts, pred, valid_mask in zip(
            gt_pts_views, pred_pts_views, preds, valid_masks
        ):
            point_loss = self.criterion(pred_pts[valid_mask], gt_pts[valid_mask])
            if point_loss.numel() == 0:
                continue
            conf = pred["conf"][valid_mask]
            total_loss = (
                total_loss + (point_loss * conf - self.alpha * torch.log(conf)).mean()
            )
        return total_loss


class DepthLoss(nn.Module):
    """VGGT's depth loss (confidence, regression and gradient terms)."""

    def forward(self, views: Views, preds: Preds) -> torch.Tensor:
        """Compute the loss from VGGT-normalized views and the "depths" predictions."""
        views, depths = list(views), list(preds["depths"])
        _check_num_views(views, depths)
        losses = compute_depth_loss(
            predictions=dict(
                depth=stack_views(depths, "depth").unsqueeze(-1),  # (B, V, H, W, 1)
                depth_conf=stack_views(depths, "conf"),  # (B, V, H, W)
            ),
            batch=dict(
                depths=stack_views(views, "depthmap"),
                point_masks=stack_views(views, "valid_mask"),
            ),
            gradient_loss_fn="grad",
            valid_range=0.98,
        )
        return (
            losses["loss_conf_depth"]
            + losses["loss_reg_depth"]
            + losses["loss_grad_depth"]
        )


class CameraLoss(nn.Module):
    """VGGT's camera pose loss."""

    def forward(self, views: Views, preds: Preds) -> torch.Tensor:
        """Compute the loss from VGGT-normalized views and the "poses" predictions.

        Args:
            views: Per-view GT dictionaries after VGGT normalization, with c2w
                OpenCV `camera_pose` (first view as world frame) and
                `camera_intrinsics` with a centered principal point.
            preds: MVT predictions, whose "poses" entry holds per-view `pose_enc`
                tensors (B, 9) in VGGT's convention (w2c OpenCV cameras).

        Returns:
            Scalar loss.
        """
        views, poses = list(views), list(preds["poses"])
        _check_num_views(views, poses)
        # Extrinsics are (B, V, 3, 4) OpenCV w2c.
        extrinsics = torch.linalg.inv(stack_views(views, "camera_pose"))[..., :3, :]
        losses = compute_camera_loss(
            pred_dict=dict(pose_enc_list=[stack_views(poses, "pose_enc")]),
            batch_data=dict(
                extrinsics=extrinsics,
                intrinsics=stack_views(views, "camera_intrinsics"),
                point_masks=stack_views(views, "valid_mask"),
                images=stack_views(views, "img"),
            ),
            gamma=0.0,
        )
        return losses["loss_camera"]


class VGGTPointmapLoss(nn.Module):
    """VGGT's pointmap loss (confidence, regression and normal terms)."""

    def forward(self, views: Views, preds: Preds) -> torch.Tensor:
        """Compute the loss from the "pointmaps" predictions.

        GT points are projected to the first-view frame and normalized by their
        average norm over all views.
        """
        views, pointmaps = list(views), list(preds["pointmaps"])
        _check_num_views(views, pointmaps)
        gt_pts_views, valid_masks = project_reference_frame_gt(views)
        gt_pts_views = normalize_pointmaps_by_avg_distance(gt_pts_views, valid_masks)
        losses = compute_point_loss(
            predictions=dict(
                world_points=stack_views(pointmaps, "pts3d"),  # (B, V, H, W, 3)
                world_points_conf=stack_views(pointmaps, "conf"),  # (B, V, H, W)
            ),
            batch=dict(
                world_points=torch.stack(gt_pts_views, dim=1),
                point_masks=torch.stack(valid_masks, dim=1),
            ),
            gradient_loss_fn="normal",
            valid_range=0.98,
        )
        return (
            losses["loss_conf_point"]
            + losses["loss_reg_point"]
            + losses["loss_grad_point"]
        )


class VGGTLoss(nn.Module):
    """Weighted sum of the depth, camera and pointmap losses of the predicted outputs."""

    def __init__(
        self,
        depth_weight: float = 1.0,
        camera_weight: float = 1.0,
        pointmap_weight: float = 1.0,
        pointmap_loss: Literal["vggt", "dust3r"] = "vggt",
    ) -> None:
        """Initialize the loss.

        Args:
            depth_weight: Weight of the depth loss.
            camera_weight: Weight of the camera loss.
            pointmap_weight: Weight of the pointmap loss.
            pointmap_loss: Pointmap loss, VGGT's ("vggt") or DUSt3R's
                confidence-weighted loss ("dust3r").
        """
        super().__init__()
        self.depth_weight = depth_weight
        self.camera_weight = camera_weight
        self.pointmap_weight = pointmap_weight
        self.depth_loss = DepthLoss()
        self.camera_loss = CameraLoss()
        if pointmap_loss == "vggt":
            self.pointmap_loss = VGGTPointmapLoss()
        elif pointmap_loss == "dust3r":
            self.pointmap_loss = PointmapLoss()
        else:
            raise ValueError(f"Loss {pointmap_loss} not recognised")

    def forward(self, views: Views, preds: Preds) -> Tuple[torch.Tensor, dict]:
        """Normalize the GT views as VGGT does and sum the losses of predicted outputs.

        Args:
            views: Per-view GT dictionaries.
            preds: MVT predictions, with any of the "depths", "poses" and
                "pointmaps" entries.

        Returns:
            The total loss, and a dict of detached loss terms.
        """
        device = next(iter(next(iter(preds.values()))[0].values())).device
        views = normalize_views_vggt(views, device)

        terms = {}
        if "depths" in preds:
            terms["depth_loss"] = self.depth_weight * self.depth_loss(views, preds)
        if "poses" in preds:
            terms["cam_loss"] = self.camera_weight * self.camera_loss(views, preds)
        if "pointmaps" in preds:
            terms["point_loss"] = self.pointmap_weight * self.pointmap_loss(
                views, preds
            )
        if not terms:
            raise ValueError(f"Unsupported MVT prediction keys: {list(preds.keys())}")

        total_loss = sum(terms.values())
        details = {key: value.detach().cpu() for key, value in terms.items()}
        details["total_loss"] = total_loss.detach().cpu()
        return total_loss, details
