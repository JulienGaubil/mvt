"""Pointmap normalization, Procrustes alignment, reprojection and GT correspondences."""

import torch

from jaxtyping import Bool, Float, Int
from typing import Any, Dict, List, Sequence, Tuple

from mvt import _thirdp  # noqa: F401
from mvt.utils import masked_mean
from dust3r.utils.geometry import geotrf, inv


def project_reference_frame_gt(
    views: Sequence[Dict[str, Any]],
) -> Tuple[
    List[Float[torch.Tensor, "batch height width xyz=3"]],
    List[Bool[torch.Tensor, "batch height width"]],
]:
    """Project all GT pointmaps into the first view coordinate system.

    Args:
        views: Per-view GT dictionaries. `camera_pose` entries are c2w
            transforms in OpenCV camera coordinates and `pts3d` are pixel-aligned
            3D points in a shared world frame.

    Returns:
        Per-view pointmaps in the first-view frame, and copies of the per-view
        valid masks.
    """
    in_reference = inv(views[0]["camera_pose"])  # first-view OpenCV w2c
    gt_pts_views = [geotrf(in_reference, view["pts3d"]) for view in views]
    valid_masks = [view["valid_mask"].clone() for view in views]
    return gt_pts_views, valid_masks


def get_pointmaps_avg_distance(
    pts_views: Sequence[Float[torch.Tensor, "batch height width xyz=3"]],
    valid_masks: Sequence[Bool[torch.Tensor, "batch height width"]],
) -> Float[torch.Tensor, "batch"]:
    """Average point norm over the valid points of all views, per sample."""
    B = pts_views[0].shape[0]
    all_pts = torch.cat([pts.reshape(B, -1, 3) for pts in pts_views], dim=1)
    all_masks = torch.cat([mask.reshape(B, -1) for mask in valid_masks], dim=1)
    avg_norm = masked_mean(
        all_pts.norm(dim=-1), all_masks.to(all_pts.device), dim=1, keepdim=False
    )
    return avg_norm.clamp_min(1e-8)


def normalize_pointmaps_by_avg_distance(
    pts_views: Sequence[Float[torch.Tensor, "batch height width xyz=3"]],
    valid_masks: Sequence[Bool[torch.Tensor, "batch height width"]],
) -> List[Float[torch.Tensor, "batch height width xyz=3"]]:
    """Rescale pointmaps by their average norm over the valid points of all views."""
    avg_norm = get_pointmaps_avg_distance(pts_views, valid_masks)
    avg_norm = avg_norm.view(-1, *([1] * (pts_views[0].ndim - 1)))  # (B, 1, 1, 1)
    return [pts / avg_norm.to(device=pts.device, dtype=pts.dtype) for pts in pts_views]


def pointmap_alignment(
    views: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, torch.Tensor]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Project GT to view 0 and normalize GT and predictions by their average norm.

    Args:
        views: Per-view GT dictionaries. `camera_pose` is c2w and `pts3d` is in a
            shared world frame.
        preds: Per-view predictions, with `pts3d` in the first-view frame.

    Returns:
        Tuple of (gt_pts_views, pred_pts_views, valid_masks) with the normalized
        GT and predicted pointmaps in the first-view frame.
    """
    gt_pts_views, valid_masks = project_reference_frame_gt(views)
    gt_pts_views = normalize_pointmaps_by_avg_distance(gt_pts_views, valid_masks)
    pred_pts_views = normalize_pointmaps_by_avg_distance(
        [pred["pts3d"] for pred in preds], valid_masks
    )
    return gt_pts_views, pred_pts_views, valid_masks


def normalize_views_vggt(
    views: Sequence[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Apply VGGT's GT normalization, keeping the per-view dictionary format.

    The first view becomes the world frame and the scene is rescaled by its
    average point norm.

    Args:
        views: Per-view GT dictionaries. `camera_pose` is an OpenCV c2w
            transform, `camera_intrinsics` an OpenCV intrinsics matrix, `pts3d`
            is in the original world frame and `depthmap` is per-view depth.
        device: Device of the returned geometry tensors.

    Returns:
        Per-view dictionaries with normalized c2w `camera_pose`, `pts3d` (world
        frame), `pts3d_cam` (camera frame) and `depthmap`.
    """
    views = list(views)
    pts3d_list, valid_masks = project_reference_frame_gt(views)
    avg_norm = get_pointmaps_avg_distance(pts3d_list, valid_masks)
    avg_norm = avg_norm.to(device).to(pts3d_list[0].dtype)

    in_reference = inv(views[0]["camera_pose"]).to(device)
    normalized_views = []
    for view in views:
        B = view["pts3d"].shape[0]
        c2ws = in_reference @ view["camera_pose"].to(device)
        pts3d = geotrf(in_reference, view["pts3d"].to(device))
        pts3d_local = geotrf(inv(c2ws), pts3d)

        c2ws_norm = c2ws.clone()
        c2ws_norm[:, :3, 3] /= avg_norm.view(B, 1)
        normalized_view = dict(view)
        normalized_view["camera_pose"] = c2ws_norm
        normalized_view["pts3d"] = pts3d / avg_norm.view(B, 1, 1, 1)
        normalized_view["pts3d_cam"] = pts3d_local / avg_norm.view(B, 1, 1, 1)
        normalized_view["depthmap"] = view["depthmap"].to(device) / avg_norm.view(
            B, 1, 1
        )
        normalized_view["valid_mask"] = view["valid_mask"].to(device)
        normalized_views.append(normalized_view)

    return normalized_views


def procrustes_alignment(
    source: Float[torch.Tensor, "*batch num_points 3"],
    target: Float[torch.Tensor, "*batch num_points 3"],
    weights: Float[torch.Tensor, "*batch num_points"] | None = None,
) -> Float[torch.Tensor, "*batch num_points 3"]:
    """Align a source point cloud to a target point cloud with a rigid transform.

    Finds the rotation and translation minimizing the weighted squared distance
    between transformed source and target points ("Least-Squares Rigid Motion
    Using SVD", Sorkine-Hornung and Rabinovich).

    Adapted from: https://github.com/dcharatan/flowmap

    Args:
        source: Source point cloud.
        target: Target point cloud.
        weights: Optional per-point weights. Uniform if None.

    Returns:
        The source points aligned to the target.
    """
    *batch, num_points, _ = source.shape
    if weights is None:
        weights = torch.ones(
            *batch, num_points, device=source.device, dtype=source.dtype
        )
    weights_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    # Center the point clouds on their weighted centroids.
    source_centroid = (weights_normalized[..., None] * source).sum(dim=-2)
    target_centroid = (weights_normalized[..., None] * target).sum(dim=-2)
    source_centered = source - source_centroid[..., None, :]
    target_centered = target - target_centroid[..., None, :]

    # Optimal rotation from the SVD of the weighted covariance, without reflection.
    covariance = (target_centered * weights[..., None]).transpose(
        -1, -2
    ) @ source_centered
    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=source.dtype, device=source.device)
    s = s.expand(*batch, 3, 3).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    translation = target_centroid - torch.einsum(
        "...ij,...j->...i", rotation, source_centroid
    )
    return (
        torch.einsum("...ij,...nj->...ni", rotation, source) + translation[..., None, :]
    )


def per_view_procrustes_align(
    pred_pts_views: Sequence[torch.Tensor],
    gt_pts_views: Sequence[torch.Tensor],
    valid_masks: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Align each predicted pointmap to GT with its own rigid fit on valid pixels."""
    aligned = []
    for pred_pts, gt_pts, valid_mask in zip(pred_pts_views, gt_pts_views, valid_masks):
        B = len(pred_pts)
        pred_aligned = procrustes_alignment(
            pred_pts.reshape(B, -1, 3),
            gt_pts.reshape(B, -1, 3),
            weights=valid_mask.reshape(B, -1).float(),
        )
        aligned.append(pred_aligned.reshape_as(pred_pts))
    return aligned


def warp_pc_to_pixel(
    world_pts: Float[torch.Tensor, "batch height width xyz=3"],
    c2w_trgt: Float[torch.Tensor, "batch 4 4"],
    k_trgt: Float[torch.Tensor, "batch 3 3"],
) -> Int[torch.Tensor, "batch height width rowcol=2"]:
    """Project world points to the rounded (row, col) pixels of a target camera.

    Points behind the camera or with a non-finite projection map to (-1, -1).
    """
    world_pts = world_pts.to(torch.float64)
    c2w_trgt = c2w_trgt.to(torch.float64)
    rotation = c2w_trgt[..., :3, :3]
    translation = c2w_trgt[..., :3, 3]
    local_pts = (
        torch.einsum("...xyj,...jk->...xyk", world_pts, rotation)
        + (-torch.einsum("...ij,...j->...i", rotation.transpose(-1, -2), translation))[
            ..., None, None, :
        ]
    )

    projected = torch.einsum(
        "...ij,...xyj->...xyi", k_trgt.to(torch.float64), local_pts
    )
    depths = projected[..., 2]
    projected_uv = projected[..., :2] / depths[..., None]
    valid = (
        torch.isfinite(depths) & (depths > 1e-8) & torch.isfinite(projected_uv).all(-1)
    )
    pixels = torch.full(
        (*local_pts.shape[:-1], 2), -1, dtype=torch.int64, device=local_pts.device
    )
    if torch.any(valid):
        pixels[valid] = torch.round(projected_uv[..., [1, 0]][valid]).to(torch.int64)
    return pixels


def mutual_correspondence_masks(
    view1_map: Int[torch.Tensor, "batch height1 width1 rowcol=2"],
    view2_map: Int[torch.Tensor, "batch height2 width2 rowcol=2"],
    valid_mask1: Bool[torch.Tensor, "batch height1 width1"],
    valid_mask2: Bool[torch.Tensor, "batch height2 width2"],
    tolerance_px: int = 0,
) -> Tuple[Bool[torch.Tensor, "batch height1 width1"], torch.Tensor]:
    """Compute the masks of pixels with a mutual (round-trip) correspondence.

    Args:
        view1_map: Pixel map from view 1 to view 2, as (row, col).
        view2_map: Pixel map from view 2 to view 1, as (row, col).
        valid_mask1: Valid pixels of view 1.
        valid_mask2: Valid pixels of view 2.
        tolerance_px: Maximum round-trip error in pixels.

    Returns:
        The pixels of view 1 whose round trip lands within `tolerance_px`, and
        the pixels of view 2 they map to.
    """
    batch_size, h1, w1, _ = view1_map.shape
    h2, w2 = view2_map.shape[1:3]
    device = view1_map.device
    ii = torch.arange(h1, device=device).view(h1, 1).expand(h1, w1)
    jj = torch.arange(w1, device=device).view(1, w1).expand(h1, w1)
    mask1 = torch.zeros((batch_size, h1, w1), dtype=torch.bool, device=device)
    mask2 = torch.zeros((batch_size, h2, w2), dtype=torch.bool, device=device)
    tol = max(0, int(tolerance_px))

    for b in range(batch_size):
        u, v = view1_map[b, ..., 0], view1_map[b, ..., 1]
        corr = valid_mask1[b] & (u >= 0) & (u < h2) & (v >= 0) & (v < w2)
        if not torch.any(corr):
            continue
        corr = corr & torch.zeros_like(corr).masked_scatter(
            corr, valid_mask2[b, u[corr], v[corr]]
        )
        if not torch.any(corr):
            continue

        back = view2_map[b, u[corr], v[corr]]
        x = torch.full((h1, w1), -1, dtype=torch.int64, device=device)
        y = torch.full((h1, w1), -1, dtype=torch.int64, device=device)
        x[corr], y[corr] = back[:, 0], back[:, 1]
        mutual = corr & ((x - ii).abs() <= tol) & ((y - jj).abs() <= tol)
        mask1[b] = mutual
        mask2[b, u[mutual], v[mutual]] = True

    return mask1, mask2


def symmetric_mutual_correspondence_masks(
    view1_map: torch.Tensor,
    view2_map: torch.Tensor,
    valid_mask1: torch.Tensor,
    valid_mask2: torch.Tensor,
    tolerance_px: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Union of the mutual-correspondence masks obtained from either view.

    Starting from both views avoids the target-side sparsity caused by the
    asymmetric rasterization of `mutual_correspondence_masks`.
    """
    mask1_fwd, mask2_fwd = mutual_correspondence_masks(
        view1_map, view2_map, valid_mask1, valid_mask2, tolerance_px
    )
    mask2_bwd, mask1_bwd = mutual_correspondence_masks(
        view2_map, view1_map, valid_mask2, valid_mask1, tolerance_px
    )
    return mask1_fwd | mask1_bwd, mask2_fwd | mask2_bwd


def warp_field(
    src_field: Float[torch.Tensor, "batch height width channel"],
    trgt_2_src_map: Int[torch.Tensor, "batch height width rowcol=2"],
    trgt_stereo_mask: Bool[torch.Tensor, "batch height width"],
) -> Float[torch.Tensor, "batch height width channel"]:
    """Warp a pixel field from a source view to a target view.

    Args:
        src_field: Pixel field defined on the source view.
        trgt_2_src_map: For each target pixel, the (row, col) of the matching
            source pixel.
        trgt_stereo_mask: True for target pixels visible in both views.

    Returns:
        The source field warped to the target view, NaN outside the stereo mask.
    """
    warped = torch.full_like(src_field, torch.nan)
    batch_idx = torch.arange(src_field.shape[0], device=src_field.device)[
        :, None, None
    ].expand_as(trgt_stereo_mask)
    rows = trgt_2_src_map[..., 0].clamp(0, src_field.shape[1] - 1)
    cols = trgt_2_src_map[..., 1].clamp(0, src_field.shape[2] - 1)
    mask = trgt_stereo_mask
    warped[mask] = src_field[batch_idx[mask], rows[mask], cols[mask]]
    return warped
