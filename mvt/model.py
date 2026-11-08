"""Multi-view transformer (MVT) trained from scratch.

The MVT alternates single-view and cross-view attention blocks on top of a frozen
CroCoV2 or DINOv2 image encoder, and predicts pointmaps and/or depth and camera poses.
"""

import copy
import torch
import torch.nn as nn
import torchvision.transforms as tvf

from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from jaxtyping import Float, Int
from contextlib import nullcontext
from typing import Any, Dict, Tuple, Literal, Callable, Sequence

from mvt import _thirdp  # noqa: F401
from mvt.datasets import dust3r_inv, imagenet_norm
from mvt.heads import DepthHead, PointmapHead, PoseHead
from mvt.utils import tf32_matmul
from vggt.layers.layer_scale import LayerScale
from vggt.layers.mlp import Mlp
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.models.aggregator import slice_expand_and_flatten

# CroCo enables TF32 matmuls globally at import: restore the setting, the
# forward pass enables them locally.
_ALLOW_TF32 = torch.backends.cuda.matmul.allow_tf32
from models.croco import CroCoNet

torch.backends.cuda.matmul.allow_tf32 = _ALLOW_TF32

# Pretrained CroCoV2 weights, downloaded from https://github.com/naver/croco.
CROCOV2_CHECKPOINT = "pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth"


class ViewAttention(nn.Module):
    """Single-view or cross-view multi-head attention over multi-view tokens.

    Single-view attention attends within each view. Cross-view attention either
    attends over the tokens of all views ("global_attn") or only over the tokens of
    the other views ("cross_attn"). Without weight sharing, each view has its own
    projection weights.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = True,
        rope: Callable | None = None,
        num_views: int | None = 2,
        weight_sharing: bool = False,
        cross_view: bool = False,
        cva_param: Literal["global_attn", "cross_attn"] = "global_attn",
    ) -> None:
        """Initialize the attention layer.

        Args:
            dim: Token dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to include bias in the QKV projection.
            proj_bias: Whether to include bias in the output projection.
            norm_layer: Normalization layer used for QK normalization.
            qk_norm: Whether to apply QK normalization.
            rope: Optional rotary position embedding, applied in single-view
                attention only.
            num_views: Number of views, required without weight sharing.
            weight_sharing: Whether projection weights are shared across views.
            cross_view: Whether the layer attends across views.
            cva_param: Cross-view attention mode, "global_attn" or "cross_attn".

        Raises:
            ValueError: If ``cva_param`` is invalid, or ``num_views`` is missing
                without weight sharing.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.weight_sharing = weight_sharing
        self.cross_view = cross_view
        if cva_param not in ["global_attn", "cross_attn"]:
            raise ValueError(
                f"Cross view param should be global_attn or cross_attn, got {cva_param}"
            )
        self.cva_param = cva_param

        self.rope = rope
        if not self.weight_sharing and num_views is None:
            raise ValueError("num_views is required when weight_sharing=False.")

        if self.weight_sharing:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.proj = nn.Linear(dim, dim, bias=proj_bias)
        else:
            self.qkv = nn.ModuleList(
                [nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(num_views)]
            )
            self.q_norm = nn.ModuleList(
                [
                    norm_layer(self.head_dim) if qk_norm else nn.Identity()
                    for _ in range(num_views)
                ]
            )
            self.k_norm = nn.ModuleList(
                [
                    norm_layer(self.head_dim) if qk_norm else nn.Identity()
                    for _ in range(num_views)
                ]
            )
            self.proj = nn.ModuleList(
                [nn.Linear(dim, dim, bias=proj_bias) for _ in range(num_views)]
            )

        self.num_views = num_views

    def compute_qkv(
        self, x: Float[torch.Tensor, "batch view token channel"]
    ) -> Tuple[
        Float[torch.Tensor, "batch head token head_channel"],
        Float[torch.Tensor, "batch head token head_channel"],
        Float[torch.Tensor, "batch head token head_channel"],
    ]:
        """Compute the (QK-normalized) queries, keys and values.

        Args:
            x: Input tokens of shape (B, V, T, C).

        Returns:
            Queries, keys and values of shape (B', H, N, C_head), with B' = B and
            N = V * T for cross-view attention, and B' = B * V and N = T otherwise.
        """
        B, V, T, C = x.shape

        if self.weight_sharing:
            if self.cross_view:
                x = x.reshape(B, V * T, C)
                N = V * T
            else:
                x = x.reshape(B * V, T, C)
                N = T
                B = B * V

            # Get QKV.
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(
                2, 0, 3, 1, 4
            )  # (qkv batch head token channel)

            # QK-Norm.
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

        else:
            if V != self.num_views:
                raise ValueError(
                    f"Mismatch between model ({self.num_views}) and batch ({V}) num views."
                )

            qkv = torch.stack(
                [self.qkv[v](x[:, v]) for v in range(self.num_views)], dim=1
            )

            qkv = qkv.reshape(B, self.num_views, T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(3)  # (batch view token head channel)

            q = torch.stack(
                [self.q_norm[v](q[:, v]) for v in range(self.num_views)], dim=1
            )
            k = torch.stack(
                [self.k_norm[v](k[:, v]) for v in range(self.num_views)], dim=1
            )

            if self.cross_view:
                q = q.reshape(
                    B, self.num_views * T, self.num_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                k = k.reshape(
                    B, self.num_views * T, self.num_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                v = v.reshape(
                    B, self.num_views * T, self.num_heads, self.head_dim
                ).permute(0, 2, 1, 3)
            else:
                q = q.reshape(
                    B * self.num_views, T, self.num_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                k = k.reshape(
                    B * self.num_views, T, self.num_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                v = v.reshape(
                    B * self.num_views, T, self.num_heads, self.head_dim
                ).permute(0, 2, 1, 3)

        return q, k, v

    def forward(
        self,
        x: Float[torch.Tensor, "batch view token channel"],
        pos: Int[torch.Tensor, "batch view token yx=2"] | None = None,
    ) -> Float[torch.Tensor, "batch view token channel"]:
        """Apply attention.

        Args:
            x: Input tokens of shape (B, V, T, C).
            pos: Optional token positions for RoPE.

        Returns:
            Output tokens of shape (B, V, T, C).
        """
        V = x.shape[1]
        C = x.shape[-1]
        q, k, v = self.compute_qkv(x)
        B, _, N, _ = q.shape

        if self.rope is not None and not self.cross_view and pos is not None:
            pos = pos.reshape(B, N, 2)
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Cross-attention masks intra-view tokens out of the global attention matrix.
        if self.cross_view and self.cva_param == "cross_attn":
            frame_idxs = torch.arange(V).to(x.device)
            mask = torch.ones(
                B, self.num_heads, V, N // V, V, N // V, dtype=bool, device=x.device
            )
            mask[:, :, frame_idxs, :, frame_idxs, :] = False
            mask = mask.reshape(B, self.num_heads, N, N)
        else:
            mask = torch.ones(B, self.num_heads, N, N, device=x.device).bool()

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (batch head token token)
        attn[~mask] = -float("inf")
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        if self.cross_view:
            x = x.reshape(B, V, N // V, C)
        else:
            x = x.reshape(B // V, V, N, C)

        if self.weight_sharing:
            x = self.proj(x)
        else:
            x = torch.stack(
                [self.proj[v](x[:, v]) for v in range(self.num_views)], dim=1
            )

        return x


class ViewAttentionBlock(nn.Module):
    """Pre-norm transformer block with view attention and an optional MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = ViewAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope: Callable | None = None,
        num_views: int | None = 2,
        weight_sharing: bool = False,
        cross_view: bool = False,
        cva_param: Literal["global_attn", "cross_attn"] = "global_attn",
        use_ffn_layers: bool = True,
    ) -> None:
        """Initialize the block.

        Args:
            dim: Token dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dim to token dim.
            qkv_bias: Whether to include bias in the QKV projection.
            proj_bias: Whether to include bias in the output projection.
            ffn_bias: Whether to include bias in the MLP.
            init_values: Layer scale init value. Disables layer scale if None.
            act_layer: MLP activation.
            norm_layer: Normalization layer.
            attn_class: Attention layer class.
            ffn_layer: MLP class.
            qk_norm: Whether to apply QK normalization.
            rope: Optional rotary position embedding.
            num_views: Number of views, required without weight sharing.
            weight_sharing: Whether weights are shared across views.
            cross_view: Whether the block attends across views.
            cva_param: Cross-view attention mode, "global_attn" or "cross_attn".
            use_ffn_layers: Whether to apply the MLP sublayer.
        """
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            rope=rope,
            num_views=num_views,
            weight_sharing=weight_sharing,
            cross_view=cross_view,
            cva_param=cva_param,
        )

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.use_ffn_layers = use_ffn_layers
        if self.use_ffn_layers:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = ffn_layer(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=ffn_bias,
            )
            self.ls2 = (
                LayerScale(dim, init_values=init_values)
                if init_values
                else nn.Identity()
            )
        else:
            # Keep the MLP attributes for block walkers, without unused MLP parameters.
            self.norm2 = nn.Identity()
            self.mlp = nn.Identity()
            self.ls2 = nn.Identity()

    def forward(
        self,
        x: Float[torch.Tensor, "batch view token channel"],
        pos: Int[torch.Tensor, "batch view token yx=2"] | None = None,
    ) -> Float[torch.Tensor, "batch view token channel"]:
        """Apply the block to tokens of shape (B, V, T, C)."""
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        if self.use_ffn_layers:
            x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class MVT(nn.Module):
    """Multi-view transformer on top of a frozen CroCoV2 or DINOv2 image encoder.

    Args:
        embed_dim (int): Dimension of the token embeddings.
        num_mvt_blocks (int): Number of single-view/cross-view block pairs.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. 0 disables RoPE.
        init_values (float): Init scale for layer scale.
        encoder_init (str): Image encoder initialization, either "crocov2" or "dinov2".
        cva_param (str): Cross-view attention mode, either "global_attn" or "cross_attn".
        view_weight_sharing (bool): Whether view-attention weights and heads are
            shared across views. Special tokens remain reference/other-specific,
            regardless of this setting.
        output_param (str): Enabled reconstruction heads: "pointmap", "factorised", or "mixed".
        head_type (str): Output head type, either "linear" or "dense".
        num_views (int | None): Number of views required when view weights are not shared.
        freeze_encoder (bool): Whether to keep the image encoder frozen.
        use_ffn_layers (bool): Whether to apply the MLP/FFN sublayer in each view-attention block.
        crocov2_checkpoint (str): Path of the pretrained CroCoV2 weights.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_mvt_blocks: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        rope_freq: int = 100,
        init_values: float = 0.01,
        encoder_init: Literal["crocov2", "dinov2"] = "dinov2",
        cva_param: Literal["global_attn", "cross_attn"] = "cross_attn",
        view_weight_sharing: bool = True,
        output_param: Literal["pointmap", "factorised", "mixed"] = "pointmap",
        head_type: Literal["linear", "dense"] = "linear",
        num_views: int | None = None,
        freeze_encoder: bool = True,
        use_ffn_layers: bool = True,
        crocov2_checkpoint: str = CROCOV2_CHECKPOINT,
    ):
        super().__init__()

        if not view_weight_sharing and num_views is None:
            raise ValueError("num_views is required when view_weight_sharing=False.")

        self.encoder_init = encoder_init
        self.cva_param = cva_param
        self.view_weight_sharing = view_weight_sharing
        self.output_param = output_param
        self.num_views = num_views
        self.freeze_encoder = freeze_encoder
        self.use_ffn_layers = use_ffn_layers
        self.predict_pointmaps = output_param in ["pointmap", "mixed"]
        self.predict_depth = output_param in ["factorised", "mixed"]
        self.predict_pose = output_param in ["factorised", "mixed"]

        # Load encoder.
        if encoder_init == "crocov2":
            ckpt = torch.load(crocov2_checkpoint, "cpu")
            model = CroCoNet(**ckpt.get("croco_kwargs", {}))
            model.load_state_dict(ckpt["model"], strict=True)
            model = model.eval().to(torch.float32)
            self.patch_size = model.patch_embed.patch_size[0]
        elif encoder_init == "dinov2":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
            self.patch_size = model.patch_embed.proj.kernel_size[0]
        else:
            raise ValueError(f"Unknown encoder_init: {encoder_init}")
        self.encoder = model
        if self.freeze_encoder:
            self.encoder = self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        # Project encoder features to the MVT width if needed.
        self.encoder_embed_dim = (
            self.encoder.embed_dim
            if encoder_init == "dinov2"
            else self.encoder.enc_embed_dim
        )
        self.encoder_proj = (
            nn.Linear(self.encoder_embed_dim, embed_dim)
            if self.encoder_embed_dim != embed_dim
            else nn.Identity()
        )

        # Initialize rotary position embedding if frequency > 0.
        self.rope = (
            RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        )
        self.position_getter = PositionGetter() if self.rope is not None else None

        # MVT.
        self.embed_dim = embed_dim
        self.num_mvt_blocks = num_mvt_blocks
        mvt_blocks = list()
        for _ in range(num_mvt_blocks):
            mvt_blocks.extend(
                [
                    ViewAttentionBlock(
                        embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                        num_views=num_views,
                        weight_sharing=view_weight_sharing,
                        cva_param=cva_param,
                        cross_view=cross_view,
                        use_ffn_layers=use_ffn_layers,
                    )
                    for cross_view in [False, True]
                ]
            )
        self.mvt_blocks = nn.ModuleList(mvt_blocks)

        # Register tokens tell ref/non-ref views apart, even with shared weights.
        self.num_register_tokens = num_register_tokens
        self.num_camera_tokens = 1 if self.predict_pose else 0
        if self.num_register_tokens == 0 and self.num_camera_tokens == 0:
            raise ValueError(
                "MVT requires at least one register or camera token to encode "
                "the reference-view role."
            )
        num_token_roles = 2
        self.register_token = nn.Parameter(
            torch.randn(
                1,
                num_token_roles,
                self.num_register_tokens,
                embed_dim,
            )
        )
        self.camera_token = nn.Parameter(
            torch.randn(
                1,
                num_token_roles,
                self.num_camera_tokens,
                embed_dim,
            )
        )
        nn.init.normal_(self.register_token, std=1e-6)
        nn.init.normal_(self.camera_token, std=1e-6)

        # Output head.
        self.head_type = head_type
        if self.head_type == "dense":
            self.head_block_idxs = [
                num_mvt_blocks // 4 - 1,
                num_mvt_blocks // 2 - 1,
                num_mvt_blocks // 4 * 3 - 1,
                num_mvt_blocks - 1,
            ]
        else:
            self.head_block_idxs = [num_mvt_blocks - 1]

        if self.predict_pointmaps:
            if self.head_type == "dense":
                self.pointmap_head = PointmapHead(
                    4 * [embed_dim],
                    patch_size=self.patch_size,
                    has_conf=True,
                    head_type="dense",
                )
            elif self.head_type == "linear":
                self.pointmap_head = PointmapHead(
                    embed_dim,
                    patch_size=self.patch_size,
                    has_conf=True,
                    num_layers=1,
                    head_type="mlp",
                )
            else:
                raise ValueError(f"Unknown head type: {self.head_type}")
            if not view_weight_sharing:
                self.pointmap_head = nn.ModuleList(
                    [copy.deepcopy(self.pointmap_head) for _ in range(num_views)]
                )

        if self.predict_depth:
            if self.head_type == "dense":
                self.depth_head = DepthHead(
                    4 * [embed_dim],
                    patch_size=self.patch_size,
                    has_conf=True,
                    head_type="dense",
                )
            elif self.head_type == "linear":
                self.depth_head = DepthHead(
                    embed_dim,
                    patch_size=self.patch_size,
                    has_conf=True,
                    num_layers=1,
                    head_type="mlp",
                )
            else:
                raise ValueError(f"Unknown head type: {self.head_type}")
            if not view_weight_sharing:
                self.depth_head = nn.ModuleList(
                    [copy.deepcopy(self.depth_head) for _ in range(num_views)]
                )

        if self.predict_pose:
            self.pose_head = PoseHead(
                embed_dim,
                num_layers=1,
            )
            if not view_weight_sharing:
                self.pose_head = nn.ModuleList(
                    [copy.deepcopy(self.pose_head) for _ in range(num_views)]
                )

        # Map images from DUSt3R to ImageNet normalization.
        self.image_transform = tvf.Compose(
            [
                dust3r_inv,
                imagenet_norm,
            ]
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        **override_kwargs: Any,
    ) -> "MVT":
        """Load an MVT model from a training checkpoint directory or file.

        Args:
            checkpoint: Experiment directory containing ``config.yaml`` and
                ``ckpt.pth``, or a direct path to a checkpoint file.
            map_location: Device mapping passed to ``torch.load``.
            strict: Raise when non-encoder checkpoint keys are missing or
                unexpected. Frozen encoder weights are allowed to be absent
                because ``train.py`` omits them when saving.
            override_kwargs: Constructor kwargs that override the saved model
                config (``cfg.model``, or ``cfg.backbone`` in older checkpoints).

        Returns:
            The MVT model in eval mode.

        Raises:
            FileNotFoundError: If the checkpoint or its config cannot be found.
            RuntimeError: If ``strict`` and non-encoder keys do not match.
        """
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir():
            ckpt_path = checkpoint_path / "ckpt.pth"
            cfg_path = checkpoint_path / "config.yaml"
        else:
            ckpt_path = checkpoint_path
            cfg_path = checkpoint_path.parent / "config.yaml"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        checkpoint_data = torch.load(
            ckpt_path,
            map_location=map_location,
            weights_only=False,
        )
        cfg = checkpoint_data.get("cfg")
        if cfg is None:
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"Could not find saved cfg in checkpoint or at {cfg_path}"
                )
            cfg = OmegaConf.load(cfg_path)

        model_cfg = cfg.model if "model" in cfg else cfg.backbone
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        model_cfg.pop("_target_", None)
        model_cfg.update(override_kwargs)
        model = cls(**model_cfg)

        state_dict = checkpoint_data["model"]
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }

        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible_keys.missing_keys)
        unexpected_keys = list(incompatible_keys.unexpected_keys)

        if strict:
            missing_non_encoder = [
                key for key in missing_keys if not key.startswith("encoder.")
            ]
            if missing_non_encoder or unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading MVT state_dict:\n"
                    f"\tMissing keys: {missing_non_encoder}\n"
                    f"\tUnexpected keys: {unexpected_keys}"
                )

        model.eval()
        return model

    @tf32_matmul()
    def forward(
        self,
        *view_images: Float[torch.Tensor, "batch rgb=3 height width"],
    ) -> Dict[str, Sequence[Dict[str, torch.Tensor]]]:
        """Run the MVT encoder, alternating-view blocks, and enabled output heads.

        Args:
            *view_images: One image tensor per view, each in DUSt3R
                normalization with shape ``(B, 3, H, W)``.

        Returns:
            A dictionary whose keys depend on output_param:

            - "pointmaps": per-view sequence of dictionaries with
              "pts3d" and "conf" tensors shaped (B, H, W, 3) and (B, H, W).
              Pointmaps are predicted directly in the canonical first-view frame.
            - "depths": per-view sequence of dictionaries with "depth"
              and "conf" tensors shaped (B, H, W).
            - "poses": per-view sequence of dictionaries with
              "pose_enc" shaped (B, 9). The encoding follows VGGT's convention:
              translation in the first 3 channels, rotation quaternion in the
              next 4 channels, and vertical/horizontal field-of-view values in
              the last 2 channels. Decoded cameras are OpenCV world-to-camera
              transforms, with the first view as the world frame. Assumes
              centered principal point.
        """

        # Prepare input images.
        images = torch.stack(view_images, dim=1)
        if not self.view_weight_sharing and images.shape[1] != self.num_views:
            raise ValueError(
                f"Mismatch between model ({self.num_views}) and batch ({images.shape[1]}) num views."
            )
        images = self.image_transform(images)  # (batch view channel height width)

        B, V, C_in, H, W = images.shape
        h_token, w_token = H // self.patch_size, W // self.patch_size
        images = images.view(B * V, C_in, H, W)

        # Encode images.
        encoder_context = torch.no_grad() if self.freeze_encoder else nullcontext()
        with encoder_context:
            if self.encoder_init == "dinov2":
                patch_tokens = self.encoder.forward_features(images)[
                    "x_norm_patchtokens"
                ]
            else:
                patch_tokens, _, _ = self.encoder._encode_image(images)

        patch_tokens = self.encoder_proj(patch_tokens)

        # Broadcast special tokens.
        register_token = slice_expand_and_flatten(self.register_token, B, V)
        camera_token = slice_expand_and_flatten(self.camera_token, B, V)
        x = torch.cat(
            [camera_token, register_token, patch_tokens], dim=1
        )  # ((batch view) token channel)

        # Add positional encoding.
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * V, h_token, w_token, device=images.device)

            num_special_tokens = (
                self.register_token.shape[2] + self.camera_token.shape[2]
            )
            if num_special_tokens > 0:
                pos = pos + 1
                pos_special = (
                    torch.zeros(B * V, num_special_tokens, 2).to(x.device).to(pos.dtype)
                )
                pos = torch.cat([pos_special, pos], dim=1)

        _, N, C = x.shape
        x = x.reshape(B, V, N, C)  # (batch view token channel)
        if pos is not None:
            pos = pos.reshape(B, V, N, 2)

        # MVT forward pass: alternates single-view and cross-view attention blocks.
        token_list = list()
        for idx, block in enumerate(self.mvt_blocks):
            x = block(x, pos)
            if (idx % 2 == 1) and (idx // 2 in self.head_block_idxs):
                token_list.append(x)

        # Split special tokens from patch tokens, then reshape.
        num_special_tokens = self.num_register_tokens + self.num_camera_tokens
        patch_token_aggregate, camera_token_aggregate = list(), list()
        for x in token_list:
            camera_token = x[:, :, : self.num_camera_tokens, :].transpose(
                -2, -1
            )  # (batch view channel num_camera_tokens)
            patch_tokens = rearrange(
                x[:, :, num_special_tokens:, :],
                "b v (h w) c -> b v c h w",
                h=h_token,
                w=w_token,
            )  # (batch view channel height_token width_token)

            patch_token_aggregate.append(patch_tokens)
            camera_token_aggregate.append(camera_token)
        patch_token_aggregate = torch.stack(
            patch_token_aggregate, dim=2
        )  # (batch view layer channel height width)
        camera_token_aggregate = camera_token_aggregate[
            -1
        ]  # (batch view channel num_camera_tokens), last-layer camera tokens

        # Output heads.
        output = dict()
        view_patch_tokens = patch_token_aggregate.unbind(1)
        camera_patch_tokens = camera_token_aggregate.unbind(1)

        if self.predict_pointmaps:
            pointmap_heads = (
                V * [self.pointmap_head]
                if self.view_weight_sharing
                else self.pointmap_head
            )

            # Per-view pointmaps in the first-view frame, with confidence maps.
            output["pointmaps"] = [
                pointmap_head(tokens)
                for tokens, pointmap_head in zip(view_patch_tokens, pointmap_heads)
            ]  # list of dicts pts3d: (batch height width xyz=3) and conf: (batch height width)

        if self.predict_depth:
            depth_heads = (
                V * [self.depth_head] if self.view_weight_sharing else self.depth_head
            )

            # List of per-view depth and confidence maps.
            output["depths"] = [
                depth_head(tokens)
                for tokens, depth_head in zip(view_patch_tokens, depth_heads)
            ]  # list of dicts depth: (batch height width) and conf: (batch height width)

        if self.predict_pose:
            pose_heads = (
                V * [self.pose_head] if self.view_weight_sharing else self.pose_head
            )

            # Per-view 9D poses: translation (3), quaternion (4), vertical/horiz. FoV (2)
            # OpenCV world-to-camera, first view as world, centered principal point.
            output["poses"] = [
                pose_head(tokens)
                for tokens, pose_head in zip(camera_patch_tokens, pose_heads)
            ]  # list of dicts pose_enc: (batch pose_dim=9), w2c, OpenCV

        return output
