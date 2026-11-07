"""Output heads mapping MVT tokens to pointmaps, depth maps and camera poses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float
from typing import Dict, List, Literal, Sequence, Tuple


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> nn.Module:
    """Build an MLP with ReLU activations and a constant hidden dimension.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of linear layers. A single layer gives a linear map.

    Returns:
        The MLP module.
    """
    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    mlp = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 2):
        mlp.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    mlp.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*mlp)


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Interpolate a feature map, chunking the batch to avoid INT_MAX overflows.

    Adapted from: https://github.com/facebookresearch/vggt-omega

    Args:
        x: Feature map of shape (B, C, H, W).
        size: Output spatial size.
        mode: Interpolation mode.
        align_corners: Passed to `F.interpolate`.

    Returns:
        Interpolated feature map of shape (B, C, *size).
    """
    if tuple(x.shape[-2:]) == tuple(size):
        return x

    int_max = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements <= int_max:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    chunks = torch.chunk(x, chunks=(input_elements // int_max) + 1, dim=0)
    interpolated_chunks = [
        F.interpolate(chunk, size=size, mode=mode, align_corners=align_corners)
        for chunk in chunks
    ]
    return torch.cat(interpolated_chunks, dim=0).contiguous()


def _make_dense_resize_layer(channels: int, resize_scale: float) -> nn.Module:
    """Build a learned resize layer (strided conv or transposed conv)."""
    if resize_scale == 1.0:
        return nn.Identity()

    if 0.0 < resize_scale < 1.0:
        inv_scale = 1.0 / resize_scale
        if not inv_scale.is_integer():
            raise ValueError(
                f"Downsample resize_scale must be reciprocal integer, got {resize_scale}."
            )
        scale = int(inv_scale)
        return nn.Conv2d(channels, channels, kernel_size=scale, stride=scale)

    if not resize_scale.is_integer():
        raise ValueError(f"Upsample resize_scale must be integer, got {resize_scale}.")
    scale = int(resize_scale)
    return nn.ConvTranspose2d(channels, channels, kernel_size=scale, stride=scale)


class ResidualConvUnit(nn.Module):
    """DPT residual convolution unit."""

    def __init__(self, features: int, activation: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    """DPT feature fusion block: optional skip fusion, refinement and resize."""

    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        activation = nn.ReLU(inplace=False)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        output = x
        if self.has_residual:
            if residual is None:
                raise ValueError(
                    "FeatureFusionBlock requires a residual tensor when has_residual=True"
                )
            output = output + self.resConfUnit1(residual)

        output = self.resConfUnit2(output)
        output = custom_interpolate(output, size=size)
        return self.out_conv(output)


class MLPHead(nn.Module):
    """MLP for dense pixel prediction from the patch tokens of one layer."""

    def __init__(
        self,
        input_dim: int | Sequence[int],
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        patch_size: int = 16,
    ) -> None:
        """Initialize the MLP head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            num_layers: Number of linear layers in the MLP.
            patch_size: Encoder patch size, used for pixel-shuffle upsampling.
        """
        super().__init__()
        self.patch_size = patch_size

        if not isinstance(input_dim, int):
            assert len(input_dim) == 1
            input_dim = input_dim[0]
        mlp_output_dim = output_dim * (patch_size**2)
        self.proj = make_mlp(input_dim, mlp_output_dim, hidden_dim, num_layers)
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        input_tokens: List[Float[torch.Tensor, "batch channel grid_height grid_width"]],
    ) -> Float[torch.Tensor, "batch channel height width"]:
        """Predict per-pixel values from a single-element list of token grids.

        Args:
            input_tokens: Single-element list with patch features of shape
                (B, C, H_grid, W_grid).

        Returns:
            Pixel-level predictions of shape (B, C_out, H, W).
        """
        if len(input_tokens) != 1:
            raise ValueError(
                f"{len(input_tokens)} block embeddings provided, should be 1."
            )
        feats = self.norm(input_tokens[0].permute(0, 2, 3, 1))
        feats = self.proj(feats).permute(0, 3, 1, 2)
        return F.pixel_shuffle(feats, self.patch_size)  # (B, C_out, H, W)


class DenseHead(nn.Module):
    """Dense prediction head aggregating the tokens of 4 layers.

    Uses DPT multi-layer aggregation while avoiding expensive convolutions at full
    resolution. Adapted from: https://github.com/facebookresearch/vggt-omega
    """

    def __init__(
        self,
        input_dims: int | Sequence[int],
        patch_size: int = 14,
        output_dim: int = 4,
        features: int = 256,
        hidden_dim: Sequence[int] = (256, 512, 1024, 1024),
        refine_upscale: int = 2,
    ) -> None:
        """Initialize the dense head.

        Args:
            input_dims: Feature dimension of each input layer, or a single dimension
                shared by all layers.
            patch_size: Encoder patch size.
            output_dim: Number of output channels.
            features: Fused feature dimension.
            hidden_dim: Projected dimension of each of the 4 input layers.
            refine_upscale: Upsampling factor of the fused feature map w.r.t. the
                token grid, before pixel shuffle.

        Raises:
            ValueError: If `patch_size` is not divisible by `refine_upscale`.
        """
        super().__init__()

        if patch_size % refine_upscale != 0:
            raise ValueError(
                f"DenseHead expects patch_size divisible by {refine_upscale}, "
                f"got patch_size={patch_size}."
            )

        self.patch_size = patch_size
        self.refine_upscale = refine_upscale
        if isinstance(input_dims, int):
            input_dims = len(hidden_dim) * [input_dims]
        assert len(input_dims) == len(hidden_dim)

        # Project and resize the input token grids.
        self.projects = nn.ModuleList(
            [nn.Conv2d(ic, oc, kernel_size=1) for ic, oc in zip(input_dims, hidden_dim)]
        )
        self.resize_layers = nn.ModuleList(
            [
                _make_dense_resize_layer(channels, refine_upscale / 2**level)
                for level, channels in enumerate(hidden_dim)
            ]
        )

        # Per-level 3x3 projections and fusion blocks.
        self.scratch = nn.Module()
        for level, channels in enumerate(hidden_dim, start=1):
            setattr(
                self.scratch,
                f"layer{level}_rn",
                nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=False),
            )
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)

        # Learned upsampling to pixel resolution.
        self.shuffle_factor = patch_size // refine_upscale
        self.proj = nn.Conv2d(
            features, output_dim * (self.shuffle_factor**2), kernel_size=1
        )

    def forward(
        self,
        aggregated_tokens_list: List[
            Float[torch.Tensor, "batch channel_in grid_height grid_width"]
        ],
    ) -> Float[torch.Tensor, "batch channel_out height width"]:
        """Fuse the input token grids and predict at pixel resolution.

        Args:
            aggregated_tokens_list: One token grid per input layer, of shape
                (B, C_in, H_grid, W_grid).

        Returns:
            Pixel-level predictions of shape (B, C_out, H, W).
        """
        if len(aggregated_tokens_list) != len(self.projects):
            raise ValueError(
                f"DenseHead expects {len(self.projects)} feature maps, "
                f"got {len(aggregated_tokens_list)}."
            )

        layer_1, layer_2, layer_3, layer_4 = [
            resize(project(x))
            for x, project, resize in zip(
                aggregated_tokens_list, self.projects, self.resize_layers
            )
        ]

        # Fuse from coarsest to finest, at (H, W) / patch_size * refine_upscale.
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        fused = self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])

        return F.pixel_shuffle(self.proj(fused), self.shuffle_factor)


class PointmapPrediction(nn.Module):
    """Map raw outputs to pointmaps (exponential distance) and confidences."""

    def forward(
        self, pred: Float[torch.Tensor, "batch channel height width"]
    ) -> Dict[str, torch.Tensor]:
        """Postprocess raw (x, y, z, conf) predictions of shape (B, 4, H, W).

        Returns:
            "pts3d" of shape (B, H, W, 3), and "conf" of shape (B, H, W) if
            predicted.
        """
        fmap = pred.permute(0, 2, 3, 1)  # (B, H, W, C)
        pts3d = fmap[..., 0:3]
        d = pts3d.norm(dim=-1, keepdim=True)
        pts3d = pts3d / d.clip(min=1e-8) * torch.expm1(d)

        output = dict(pts3d=pts3d)
        if fmap.shape[-1] > 3:
            output["conf"] = 1 + fmap[..., 3].exp()
        return output


class DepthPrediction(nn.Module):
    """Map raw outputs to depth maps (exponential) and confidences."""

    def forward(
        self, pred: Float[torch.Tensor, "batch channel height width"]
    ) -> Dict[str, torch.Tensor]:
        """Postprocess raw (depth, conf) predictions of shape (B, 2, H, W).

        Returns:
            "depth" of shape (B, H, W), and "conf" of shape (B, H, W) if predicted.
        """
        fmap = pred.permute(0, 2, 3, 1)  # (B, H, W, C)
        output = dict(depth=torch.exp(fmap[..., 0]))
        if fmap.shape[-1] > 1:
            output["conf"] = 1 + fmap[..., 1].exp()
        return output


class PosePrediction(nn.Module):
    """Map raw outputs to VGGT's absT_quaR_FoV pose encoding."""

    def forward(self, pred: Float[torch.Tensor, "batch 9"]) -> Dict[str, torch.Tensor]:
        """Postprocess raw (tx, ty, tz, q0, q1, q2, q3, fov_h, fov_w) predictions.

        Returns:
            "pose_enc" of shape (B, 9). Translation and quaternion channels are
            linear and FoV channels are ReLU-activated. Decoded cameras are OpenCV
            world-to-camera transforms with the first view as the world frame,
            and assume a centered principal point.
        """
        pose_enc = torch.cat([pred[..., :7], F.relu(pred[..., 7:])], dim=-1)
        return dict(pose_enc=pose_enc)


class _DenseOutputHead(nn.Module):
    """Linear (MLP) or dense head followed by a postprocessing module."""

    def __init__(
        self,
        output_dim: int,
        predict: nn.Module,
        feat_dim: int | Sequence[int],
        hidden_dim: int = 768,
        patch_size: int = 16,
        num_layers: int = 1,
        head_type: Literal["mlp", "dense"] = "mlp",
    ) -> None:
        super().__init__()
        if head_type == "mlp":
            self.head = MLPHead(
                feat_dim,
                hidden_dim,
                output_dim,
                patch_size=patch_size,
                num_layers=num_layers,
            )
        elif head_type == "dense":
            self.head = DenseHead(
                input_dims=feat_dim,
                patch_size=patch_size,
                output_dim=output_dim,
                refine_upscale=2,
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        self.predict = predict

    def forward(
        self,
        feats: (
            Float[torch.Tensor, "batch channel grid_height grid_width"]
            | Float[torch.Tensor, "batch layer channel grid_height grid_width"]
        ),
    ) -> Dict[str, torch.Tensor]:
        """Predict from tokens of shape (B, C, H_grid, W_grid), or
        (B, L, C, H_grid, W_grid) for the dense head."""
        feats = list(feats.unbind(1)) if feats.ndim == 5 else [feats]
        return self.predict(self.head(feats))


class PointmapHead(_DenseOutputHead):
    """Pointmap and confidence head."""

    def __init__(self, feat_dim: int | Sequence[int], has_conf: bool = True, **kwargs):
        """Initialize the head.

        Args:
            feat_dim: Token dimension, or one dimension per layer for the dense head.
            has_conf: Whether to predict confidence weights.
            **kwargs: `hidden_dim`, `patch_size`, `num_layers` and `head_type`
                ("mlp" on the last layer, or "dense" on 4 layers).
        """
        super().__init__(3 + has_conf, PointmapPrediction(), feat_dim, **kwargs)


class DepthHead(_DenseOutputHead):
    """Depth and confidence head."""

    def __init__(self, feat_dim: int | Sequence[int], has_conf: bool = True, **kwargs):
        """Initialize the head, with the same arguments as `PointmapHead`."""
        super().__init__(1 + has_conf, DepthPrediction(), feat_dim, **kwargs)


class PoseHead(nn.Module):
    """Camera pose head predicting VGGT's absT_quaR_FoV encoding from camera tokens."""

    def __init__(self, feat_dim: int, hidden_dim: int = 768, num_layers: int = 1):
        """Initialize the head.

        Args:
            feat_dim: Flattened camera-token dimension.
            hidden_dim: Hidden layer dimension of the MLP.
            num_layers: Number of MLP layers.
        """
        super().__init__()
        self.head = make_mlp(feat_dim, 9, hidden_dim, num_layers)
        self.norm = nn.LayerNorm(feat_dim)
        self.predict = PosePrediction()

    def forward(
        self, feats: Float[torch.Tensor, "batch channel token"]
    ) -> Dict[str, torch.Tensor]:
        """Predict "pose_enc" (B, 9) from camera tokens of shape (B, C, N)."""
        feats = self.norm(feats.reshape(len(feats), -1))
        return self.predict(self.head(feats))
