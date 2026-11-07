"""Habitat (HM3D) and BlendedMVS training datasets, dataloaders and image transforms.

Both datasets wrap DUSt3R's dataset classes. Images are normalized to [-1, 1]
(DUSt3R normalization).
"""

import torch
import numpy as np
import os.path as osp
import torchvision.transforms as tvf

from PIL import ImageFile
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Literal
from torch.utils.data.distributed import DistributedSampler

from mvt import _thirdp  # noqa: F401
from dust3r.datasets.blendedmvs import BlendedMVS
from dust3r.datasets.habitat import Habitat

# Avoid errors when reading truncated image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")

DUST3R_MEAN = [0.5, 0.5, 0.5]
DUST3R_STD = [0.5, 0.5, 0.5]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ImageNet normalization, applied to images in [0, 1].
imagenet_norm = tvf.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Inverse of the DUSt3R normalization, mapping dataset images back to [0, 1].
dust3r_inv = tvf.Normalize(
    mean=[-m / s for m, s in zip(DUST3R_MEAN, DUST3R_STD)],
    std=[1 / s for s in DUST3R_STD],
)


def build_loader(
    cfg: DictConfig,
    split: Literal["train", "val", "test"],
    batch_size: int,
    num_gpus: int = 1,
    n_workers: int = 8,
) -> DataLoader:
    """Build the dataloader of a dataset config split.

    Adapted from: https://github.com/mbanani/probe3d (MIT License, Copyright (c)
    2024 Mohamed El Banani).

    Args:
        cfg: Dataset config whose `_target_` is a dataset builder.
        split: Dataset split.
        batch_size: Batch size per GPU.
        num_gpus: Number of GPUs (>1 enables a DistributedSampler).
        n_workers: Number of dataloader workers.

    Returns:
        The dataloader, shuffled for the train split.
    """
    assert split in ["train", "val", "test"]
    dataset = instantiate(cfg, split=split)

    use_ddp = num_gpus > 1
    sampler = (
        DistributedSampler(dataset, shuffle=(split == "train")) if use_ddp else None
    )
    return DataLoader(
        dataset,
        batch_size,
        num_workers=n_workers,
        shuffle=(split == "train") and not use_ddp,
        sampler=sampler,
    )


def habitat_builder(
    split: Literal["train", "val", "test"], num_views: int = 2, **kwargs: Any
) -> "HabitatPointmap":
    """Build the Habitat dataset of a split.

    The training split has 100k samples. The validation and test splits are the
    two halves of the scenes of DUSt3R's 1k-sample Habitat validation set.

    Args:
        split: Dataset split.
        num_views: Number of views per sample, between 2 and 5.
        **kwargs: Arguments of DUSt3R's Habitat dataset (e.g. `ROOT`, `resolution`).

    Returns:
        The dataset.
    """
    if split == "train":
        return HabitatPointmap(
            split="train", size=100000, aug_crop=0, num_views=num_views, **kwargs
        )
    if split not in ["val", "test"]:
        raise ValueError(f"Invalid split: {split}")

    dataset = HabitatPointmap(
        split="val", size=1000, seed=777, num_views=num_views, **kwargs
    )
    half = len(dataset.scenes) // 2
    dataset.scenes = dataset.scenes[:half] if split == "val" else dataset.scenes[half:]
    return dataset


class HabitatPointmap(Habitat):
    """DUSt3R Habitat dataset returning 2 to 5 views per scene."""

    def __init__(self, *args: Any, num_views: int = 2, **kwargs: Any) -> None:
        """Initialize the dataset.

        Args:
            *args: Positional arguments of DUSt3R's Habitat dataset.
            num_views: Number of views per sample, between 2 and 5.
            **kwargs: Keyword arguments of DUSt3R's Habitat dataset.
        """
        super().__init__(*args, **kwargs)
        if not 2 <= num_views <= 5:
            raise ValueError(f"Expected 2 <= num_views <= 5, got {num_views}")
        self.num_views = num_views

    def _get_views(
        self,
        idx: int,
        resolution: int | tuple[int, int],
        rng: np.random.Generator,
    ) -> List[Dict[str, Any]]:
        """Load the views of a scene, skipping broken frames.

        View 0 is always the first frame of the scene and the other views are
        sampled without replacement from the 4 remaining frames.
        """
        scene = self.scenes[idx]
        data_path, key = osp.split(osp.join(self.ROOT, scene))
        other_indices = rng.choice(
            np.arange(1, 5), size=self.num_views - 1, replace=False
        )

        views = []
        for view_index in [0, *other_indices.tolist()]:
            # Load the view, and use the next one if this one is broken.
            for ii in range(view_index, view_index + 5):
                try:
                    image, depthmap, intrinsics, camera_pose = self._load_one_view(
                        data_path, key, ii % 5, resolution, rng
                    )
                except Exception:
                    pass
                if np.isfinite(camera_pose).all():
                    break

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="Habitat",
                    is_metric_scale=False,
                    is_synthetic=True,
                    label=osp.relpath(data_path, self.ROOT),
                    instance=f"{key}_{view_index}",
                )
            )
        return views


def blendedmvs_builder(
    split: Literal["train", "val", "test"], num_views: int = 2, **kwargs: Any
) -> "BlendedMVSPointmap":
    """Build the BlendedMVS dataset of a split.

    Args:
        split: Dataset split.
        num_views: Number of views per sample. Only 2 views are supported.
        **kwargs: Arguments of DUSt3R's BlendedMVS dataset (e.g. `ROOT`,
            `resolution`).

    Returns:
        The dataset.
    """
    if num_views != 2:
        raise ValueError("BlendedMVSPointmap only supports num_views=2")
    if split == "train":
        return BlendedMVSPointmap(split="train", aug_crop=0, **kwargs)
    if split in ["val", "test"]:
        return BlendedMVSPointmap(split=split, seed=777, **kwargs)
    raise ValueError(f"Invalid split: {split}")


class BlendedMVSPointmap(BlendedMVS):
    """DUSt3R BlendedMVS dataset with scene-disjoint train, val and test splits."""

    def _get_views(
        self,
        pair_idx: int,
        resolution: int | tuple[int, int],
        rng: np.random.Generator,
    ) -> List[Dict[str, Any]]:
        """Load an image pair and add the per-view metadata fields."""
        views = super()._get_views(pair_idx, resolution, rng)
        for view in views:
            view["is_metric_scale"] = False
            view["is_synthetic"] = True
        return views

    def _load_data(self, split: str) -> None:
        """Select the image pairs of a split.

        Scenes with `seq_low % 10 > 0` (90%) are used for training. The other
        scenes are split in two halves for validation and test. Pairs are then
        subsampled to about 100k training and 1k val/test pairs.
        """
        pairs = np.load(osp.join(self.ROOT, "blendedmvs_pairs.npy"))
        if split == "train":
            pairs = pairs[(pairs["seq_low"] % 10) > 0]
            self.pairs = pairs[:: len(pairs) // 100_000]
        elif split in ["val", "test"]:
            pairs = pairs[(pairs["seq_low"] % 10) == 0]
            scenes = np.unique(pairs["seq_low"])
            half = len(scenes) // 2
            split_scenes = scenes[:half] if split == "val" else scenes[half:]
            pairs = pairs[np.isin(pairs["seq_low"], split_scenes)]
            self.pairs = pairs[:: len(pairs) // 1000]
        else:
            raise ValueError(f"Split not recognised {split}")

        # List of all scenes, `seq_low` is unique enough.
        self.scenes = np.unique(self.pairs["seq_low"])
