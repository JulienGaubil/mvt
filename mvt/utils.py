"""Masked means, batch preparation, seeding, DDP and learning-rate schedule helpers."""

from __future__ import annotations

import torch
import random
import numpy as np

from pathlib import Path
from jaxtyping import Bool, Float
from typing import Any, Dict, Iterator, List, Sequence
from contextlib import contextmanager
from torch.distributed import init_process_group

from mvt import _thirdp  # noqa: F401
from dust3r.inference import make_batch_symmetric

@contextmanager
def tf32_matmul(enabled: bool = True) -> Iterator[None]:
    """Set TF32 matmuls (as used by DUSt3R and CroCo) within a block or function."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


# Non-tensor (or CPU-only) view fields that are not moved to the device.
HOST_KEYS = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}


def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int | List[int] | None = None,
    keepdim: bool = True,
) -> torch.Tensor:
    """Masked average of a tensor.

    Args:
        x: Tensor to average.
        mask: Mask to apply to the tensor.
        dim: Dimensions to average over.
        keepdim: Whether to keep the reduced dimensions.

    Returns:
        Masked average of the tensor.
    """
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)

    num_valid = mask.float().sum(dim=dim, keepdim=True).clamp(min=1)
    mask_mean = (
        torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim, keepdim=True) / num_valid
    )

    if not keepdim:
        mask_mean = mask_mean.squeeze(dim)

    return mask_mean


def masked_mean_per_sample(
    values: Float[torch.Tensor, "batch ..."],
    valid_mask: Bool[torch.Tensor, "batch ..."],
) -> Float[torch.Tensor, "batch"]:
    """Reduce values to one masked mean of finite entries per batch item.

    Args:
        values: Batched tensor with any number of non-batch dims.
        valid_mask: Valid mask with the same shape.

    Returns:
        Per-sample masked average, NaN if no valid pixel.
    """
    valid = valid_mask.to(device=values.device, dtype=torch.bool) & torch.isfinite(
        values
    )
    values = torch.where(valid, values, torch.zeros_like(values))
    counts = valid.reshape(len(values), -1).sum(dim=1)
    sums = values.reshape(len(values), -1).sum(dim=1)
    means = sums / counts.clamp_min(1).to(dtype=values.dtype)
    return torch.where(counts > 0, means, torch.full_like(means, float("nan")))


def stack_views(
    views: Sequence[Dict[str, Any]],
    key: str,
    dim: int = 1,
) -> torch.Tensor:
    """Stack the `key` tensor of every per-view dictionary along a new `dim` axis."""
    return torch.stack([view[key] for view in views], dim=dim)


def prepare_batch(
    batch: Sequence[Dict[str, Any]],
    device: int | torch.device,
    make_symmetric: bool = True,
) -> List[Dict[str, Any]]:
    """Prepare a batch of views and move its tensors to device.

    Args:
        batch: Per-view dictionaries collated by the dataloader.
        device: Device or device index to move the tensors to.
        make_symmetric: Whether to symmetrize 2-view batches as in DUSt3R, by
            concatenating the swapped pair (view2, view1) along the batch axis.

    Returns:
        List of per-view dictionaries with tensors on device.
    """
    batch = list(batch)
    if make_symmetric and len(batch) == 2:
        batch = list(make_batch_symmetric(batch))
    for view in batch:
        for name in view.keys():
            if name not in HOST_KEYS:
                view[name] = view[name].to(device, non_blocking=True)
    return batch


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for one run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank_device(rank: int) -> torch.device:
    """Return this rank's torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda", rank)
    return torch.device("cpu")


def ddp_setup(rank: int, world_size: int, rendezvous_file: Path) -> None:
    """Initialize distributed data parallel (DDP) training.

    Args:
        rank: Unique identifier of each process.
        world_size: Total number of processes.
        rendezvous_file: Shared file used by all ranks to initialize DDP.
    """
    init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def sanitize_exp_name(exp_name: str) -> str:
    """Normalize experiment names so they are safe as one directory name."""
    return exp_name.replace(" ", "").replace("/", "_")


def resolve_experiment_path(output_root: str | Path, exp_name: str) -> Path:
    """Create and return a fresh experiment directory under `output_root`.

    Raises:
        FileExistsError: If the path exists and is a file or a non-empty directory.
    """
    exp_path = Path(output_root).expanduser().resolve() / exp_name
    if exp_path.exists():
        if not exp_path.is_dir():
            raise FileExistsError(
                f"Output path exists and is not a directory: {exp_path}"
            )
        if any(exp_path.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {exp_path}. "
                "Refusing to start to avoid mixing artifacts."
            )
    exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path


def cosine_decay_linear_warmup(
    current_step: int,
    max_step: int,
    warmup_step: int,
    min_factor: float = 0.01,
) -> float:
    """Compute the learning-rate multiplier with linear warmup and cosine decay.

    Adapted from: https://github.com/mbanani/probe3d/blob/main/evals/utils/optim.py

    Args:
        current_step: Current training step.
        max_step: Total number of training steps.
        warmup_step: Number of warmup steps with linear increase.
        min_factor: Minimum learning-rate factor at the end of training.

    Returns:
        Learning-rate multiplier in [min_factor, 1.0].
    """
    if max_step <= 0:
        raise ValueError(f"Expected max_step > 0, got {max_step}")

    warmup_step = min(max(int(warmup_step), 0), max_step - 1)
    range_factor = 1 - min_factor

    if warmup_step > 0 and current_step <= warmup_step:
        return range_factor * (current_step / warmup_step) + min_factor
    rel_step = (current_step - warmup_step) / (max_step - warmup_step)
    return range_factor * np.cos(0.5 * rel_step * np.pi) + min_factor
