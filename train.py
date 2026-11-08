"""
Train multi-view transformers (MVTs) from scratch and evaluate them on the test split.

Adapted from: https://github.com/mbanani/probe3d/blob/main/train_depth.py
"""

from __future__ import annotations

import os
import uuid
import hydra
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from datetime import datetime
from typing import Dict, Tuple
from contextlib import nullcontext
from collections import defaultdict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from mvt.datasets import build_loader
from mvt.metrics import evaluate_pointmap_batch
from mvt.utils import (
    cosine_decay_linear_warmup,
    ddp_setup,
    prepare_batch,
    rank_device,
    resolve_experiment_path,
    sanitize_exp_name,
    set_global_seed,
    tf32_matmul,
)


def autocast_context(device: torch.device, enabled: bool):
    """Return a bf16 autocast context on CUDA, or a no-op context otherwise."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def save_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    model: nn.Module,
) -> None:
    """Save the MVT weights, without the encoder when it is frozen."""
    model_unwrapped = model.module if isinstance(model, DDP) else model
    model_state = model_unwrapped.state_dict()
    if getattr(model_unwrapped, "freeze_encoder", False):
        model_state = {
            key: value
            for key, value in model_state.items()
            if not key.startswith("encoder.")
        }

    checkpoint = {
        "cfg": cfg,
        "model": model_state,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint at {checkpoint_path}")


def unpack_loss_output(
    loss_output: torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Split a loss output into the scalar objective and a dict of loss terms."""
    if isinstance(loss_output, tuple):
        loss, details = loss_output
        return loss, dict(details)
    return loss_output, {}


@torch.no_grad()
def evaluate_pointmaps(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device,
    verbose: bool = True,
    loss_fn: torch.nn.Module | None = None,
    procrustes_align: bool = True,
    compute_multiview_error: bool = False,
    use_bf16: bool = False,
) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Evaluate the pointmaps predicted by an MVT, accumulating sums over batches.

    Args:
        model: MVT model.
        dataloader: Batches of dataset samples (used with `prepare_batch`).
        device: Device for inference (e.g. "cpu" or "cuda:0").
        verbose: Show progress bar when True.
        loss_fn: Optional loss function. When provided, the loss and its terms are
            stored under "loss" and "loss_<term>" with only the "global" field.
        procrustes_align: Whether to compute Procrustes-aligned metrics.
        compute_multiview_error: Whether to compute multi-view consistency.
        use_bf16: Whether to run inference under bf16 autocast.

    Returns:
        A dict mapping metric names to {"view_1": (mean, std), ...,
        "global": (mean, std)}. Scalar metrics such as the loss only contain
        "global".
    """
    # Aggregate statistics per metric and view.
    metric_sums = defaultdict(lambda: defaultdict(float))
    metric_sums_sq = defaultdict(lambda: defaultdict(float))
    metric_counts = defaultdict(lambda: defaultdict(float))
    metric_stat_names = defaultdict(set)

    def accumulate(metric_key: str, metric_value: Dict[str, torch.Tensor]) -> None:
        for metric_stats, values in metric_value.items():
            metric_stat_names[metric_key].add(metric_stats)
            values = values.detach().cpu().reshape(-1)
            finite_values = values[~torch.isnan(values)]
            if finite_values.numel() == 0:
                continue
            metric_sums[metric_key][metric_stats] += finite_values.sum().item()
            metric_sums_sq[metric_key][metric_stats] += (
                finite_values.square().sum().item()
            )
            metric_counts[metric_key][metric_stats] += float(finite_values.numel())

    pbar = tqdm(dataloader, desc="Evaluation") if verbose else dataloader
    for batch in pbar:
        views = prepare_batch(batch, device, make_symmetric=True)
        images = [view["img"] for view in views]

        with autocast_context(torch.device(device), use_bf16):
            preds = model(*images)

        if loss_fn is not None:
            loss, loss_details = unpack_loss_output(loss_fn(views, preds))
            accumulate(
                "loss",
                {"global": torch.as_tensor(loss, device=device).reshape(1)},
            )
            for detail_key, detail_value in loss_details.items():
                accumulate(
                    f"loss_{detail_key}",
                    {
                        "global": torch.as_tensor(
                            detail_value,
                            device=device,
                        ).reshape(1)
                    },
                )

        if "pointmaps" not in preds:
            continue

        # Evaluate pointmaps.
        batch_metrics = evaluate_pointmap_batch(
            views,
            preds["pointmaps"],
            procrustes_align=procrustes_align,
            compute_multiview_error=compute_multiview_error,
        )

        # Accumulate batch results.
        for key, value in batch_metrics.items():
            metric_value = {
                f"view_{view_idx + 1}": view_value
                for view_idx, view_value in enumerate(value)
            }
            metric_value["global"] = torch.cat(
                [view_value.reshape(-1) for view_value in value]
            )
            accumulate(key, metric_value)

    # Sum statistics across ranks.
    if dist.is_available() and dist.is_initialized():
        reduce_device = torch.device(device)
        for key in metric_sums.keys():
            for stat_name in metric_stat_names[key]:
                stats = torch.tensor(
                    [
                        metric_sums[key][stat_name],
                        metric_sums_sq[key][stat_name],
                        metric_counts[key][stat_name],
                    ],
                    dtype=torch.float64,
                    device=reduce_device,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                (
                    metric_sums[key][stat_name],
                    metric_sums_sq[key][stat_name],
                    metric_counts[key][stat_name],
                ) = stats.cpu().tolist()

    # Reduce sums to the mean and standard deviation of each metric.
    metrics = {}
    for key in set(metric_sums.keys()):
        metrics[key] = {}
        for stat_name in sorted(metric_stat_names[key]):
            if metric_counts[key][stat_name] == 0:
                metrics[key][stat_name] = (torch.nan, torch.nan)
                continue

            metric_sum = torch.as_tensor(
                metric_sums[key][stat_name], dtype=torch.float64
            )
            metric_sum_sq = torch.as_tensor(
                metric_sums_sq[key][stat_name], dtype=torch.float64
            )
            count = torch.as_tensor(metric_counts[key][stat_name], dtype=torch.float64)
            mean = metric_sum / count
            variance = torch.clamp(metric_sum_sq / count - mean.square(), min=0.0)
            metrics[key][stat_name] = (mean, variance.sqrt())

    return metrics


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    n_epochs: int,
    n_val_per_epoch: int,
    loss_fn: nn.Module,
    rank: int = 0,
    world_size: int = 1,
    valid_loader: torch.utils.data.DataLoader | None = None,
    wandb_run: wandb.Run | None = None,
    checkpoint_path: Path | None = None,
    cfg: DictConfig | None = None,
    use_bf16: bool = False,
    val_every_n_epochs: int = 1,
    grad_clip_norm: float | None = None,
) -> None:
    """Train an MVT model.

    Args:
        model: MVT model.
        train_loader: Training dataloader.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        n_epochs: Number of training epochs.
        n_val_per_epoch: Number of extra validation runs to perform within each
            epoch. This excludes the initial validation and end-of-epoch validation.
        loss_fn: Loss function module.
        rank: DDP rank (0 for single GPU).
        world_size: Total number of GPUs.
        valid_loader: Optional validation dataloader.
        wandb_run: Optional wandb run for logging.
        checkpoint_path: Optional checkpoint path to save after every epoch.
        cfg: Training config to include in saved checkpoints.
        use_bf16: Whether to run the forward pass under bf16 autocast.
        val_every_n_epochs: Run the end-of-epoch validation every n epochs.
        grad_clip_norm: Optional maximum gradient norm.
    """

    def run_validation(label: str) -> None:
        if world_size > 1:
            torch.distributed.barrier()

        if valid_loader is not None:
            val_loss, val_metrics = validate(
                model,
                valid_loader,
                loss_fn,
                verbose=rank == 0,
                wandb_run=wandb_run if rank == 0 else None,
                procrustes_align=False,
                compute_multiview_error=False,
                device=rank_device(rank),
                use_bf16=use_bf16,
            )
            if rank == 0:
                logger.info(f"valid loss {label}   | {val_loss:.4f}")
                for metric in val_metrics:
                    logger.info(f"valid {metric:10s} | {val_metrics[metric]:.4f}")

        if world_size > 1:
            torch.distributed.barrier()

    # Validation before first step.
    run_validation("initial")

    for ep in range(n_epochs):
        if world_size > 1 and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0

        # Define validation steps within epoch.
        n_train_batches = len(train_loader)
        if n_val_per_epoch > 0:
            intra_epoch_val_steps = set(
                torch.linspace(
                    0,
                    n_train_batches,
                    n_val_per_epoch + 2,
                    dtype=torch.int64,
                )[1:-1].tolist()
            )
        else:
            intra_epoch_val_steps = set()

        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()

            # Forward pass.
            views = prepare_batch(batch, rank, make_symmetric=True)
            images = [view["img"] for view in views]

            # Forward and backward passes, with TF32 matmuls like the CroCo encoder.
            with tf32_matmul():
                with autocast_context(rank_device(rank), use_bf16):
                    preds = model(*images)
                    loss, loss_details = unpack_loss_output(loss_fn(views, preds))
                loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=grad_clip_norm,
                )
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.detach().float().item()
            train_loss += loss

            # Log training progress.
            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) lr: {pr_lr:.2e}"
                )
                if wandb_run is not None:
                    log_payload = {
                        "train/train_loss": loss,
                        "lr": pr_lr,
                        "epoch": ep,
                    }
                    log_payload.update(
                        {
                            f"train/{key}": torch.as_tensor(value).float().item()
                            for key, value in loss_details.items()
                        }
                    )
                    wandb_run.log(log_payload)

            # Run validation.
            if (i + 1) in intra_epoch_val_steps:
                run_validation(f"{ep}.{i + 1}/{n_train_batches}")

        train_loss /= len(train_loader)

        # Log and validate end of epoch.
        if rank == 0:
            logger.info(f"train loss {ep}   | {train_loss:.4f}")
        if (ep + 1) % val_every_n_epochs == 0:
            run_validation(str(ep))
        if rank == 0:
            if checkpoint_path is not None and cfg is not None:
                save_checkpoint(checkpoint_path, cfg, model)

        if world_size > 1:
            torch.distributed.barrier()


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    procrustes_align: bool = False,
    compute_multiview_error: bool = False,
    verbose: bool = True,
    wandb_run: wandb.Run | None = None,
    device: str | torch.device | None = None,
    use_bf16: bool = False,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Validate an MVT model on a dataloader.

    Args:
        model: MVT model.
        loader: Validation/test dataloader.
        loss_fn: Loss function module.
        procrustes_align: Whether to compute Procrustes-aligned metrics.
        compute_multiview_error: Whether to compute multi-view consistency.
        verbose: Whether to show progress bar.
        wandb_run: Optional wandb run for logging.
        device: Device to run validation on.
        use_bf16: Whether to run inference under bf16 autocast.

    Returns:
        Tuple of (total_loss, metrics) where metrics maps metric names to mean
        values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unwrap DDP if necessary and restore the training mode afterwards.
    is_training_model = model.training
    model.eval()
    model_unwrapped = model.module if isinstance(model, DDP) else model

    raw_metrics = evaluate_pointmaps(
        model=model_unwrapped,
        dataloader=loader,
        device=device,
        verbose=verbose,
        loss_fn=loss_fn,
        procrustes_align=procrustes_align,
        compute_multiview_error=compute_multiview_error,
        use_bf16=use_bf16,
    )

    if is_training_model:
        model.train()

    # Build wandb dict from metrics.
    total_loss = 0.0
    metrics = {}
    loss_metric = raw_metrics.get("loss")
    if loss_metric is not None:
        total_loss += float(loss_metric["global"][0])

    for metric_name, metric_value in raw_metrics.items():
        if metric_name == "loss":
            continue
        view_keys = sorted(
            [key for key in metric_value.keys() if key.startswith("view_")]
        )
        if view_keys:
            for view_key in view_keys:
                metrics[f"{metric_name}_{view_key}"] = torch.as_tensor(
                    metric_value[view_key][0]
                )
        if "global" in metric_value:
            metrics[metric_name] = torch.as_tensor(metric_value["global"][0])

    if wandb_run is not None:
        wandb_run.log({f"valid/{metric}": metrics[metric] for metric in metrics})

    return total_loss, metrics


def train_model(
    rank: int,
    world_size: int,
    cfg: DictConfig,
    exp_root: str,
    timestamp: str,
    seed: int,
    wandb_run_id: str | None = None,
    rendezvous_file: Path | None = None,
) -> None:
    """Set up data, model and logging, then train and test the MVT.

    Args:
        rank: DDP rank (0 for single GPU or master process).
        world_size: Total number of GPUs.
        cfg: Hydra configuration object.
        exp_root: Parent directory where experiment folders are written.
        timestamp: Timestamp and optional note prefix shared by this launch.
        seed: Random seed for this run.
        wandb_run_id: Optional existing W&B run id to resume.
        rendezvous_file: Shared file used to initialize DDP.
    """
    set_global_seed(seed)
    if world_size > 1:
        if rendezvous_file is None:
            raise ValueError("rendezvous_file is required for multi-GPU training.")
        ddp_setup(rank, world_size, rendezvous_file)

    # Get data loaders.
    train_loader = build_loader(
        cfg.dataset,
        "train",
        cfg.batch_size,
        world_size,
        n_workers=cfg.system.num_workers,
    )
    if cfg.val.enable:
        val_loader = build_loader(
            cfg.dataset,
            "val",
            cfg.batch_size,
            world_size,
            n_workers=cfg.system.num_workers,
        )
    else:
        val_loader = None
    test_loader = build_loader(
        cfg.dataset,
        "test",
        cfg.batch_size,
        world_size,
        n_workers=cfg.system.num_workers,
    )
    _ = train_loader.dataset.__getitem__(0)  # trigger lazy initialization

    # Get model.
    model = instantiate(cfg.model)

    # Setup experiment info.
    train_dset = train_loader.dataset.__class__.__name__
    test_dset = test_loader.dataset.__class__.__name__
    model_info = [
        f"mvt_enc:{model.encoder_init:40s}_cva:{model.cva_param}"
        f"_ws:{model.view_weight_sharing}_out:{model.output_param}"
    ]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{str(cfg.optimizer.lr):>10s}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]
    exp_name = "_".join([timestamp] + model_info + train_info + [f"seed_{seed}"])
    exp_name = sanitize_exp_name(exp_name)

    # Setup logging.
    if rank == 0:
        exp_path = resolve_experiment_path(exp_root, exp_name)
        ckpt_path = exp_path / "ckpt.pth"
        cfg.output_dir = str(exp_path)
        logger_id = logger.add(exp_path / "training.log")
        logger.info(f"Seed run start | seed={seed}")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

        # Setup wandb.
        if wandb_run_id is not None:
            wandb_run = wandb.init(
                id=wandb_run_id,
                resume="must",
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                mode=cfg.wandb.mode,
                reinit=True,
            )
            wandb_run.config.update(
                {"output_dir": str(exp_path)},
                allow_val_change=True,
            )
        else:
            wandb_run = None

        # Dump config.
        OmegaConf.save(cfg, exp_path / "config.yaml")
        logger.info(f"Saved config at {exp_path / 'config.yaml'}")
    else:
        wandb_run = None
        ckpt_path = None

    # Move models to device and set to DDP.
    model = model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Setup optimizer.
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(  # noqa: E731
        epoch,
        cfg.optimizer.n_epochs * len(train_loader),
        cfg.optimizer.warmup_epochs * len(train_loader),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    loss_fn = instantiate(cfg.loss)

    # Train model.
    train(
        model,
        train_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        cfg.val.n_val_per_epoch,
        loss_fn=loss_fn,
        rank=rank,
        world_size=world_size,
        valid_loader=val_loader,
        wandb_run=wandb_run,
        checkpoint_path=ckpt_path,
        cfg=cfg,
        use_bf16=cfg.precision.bf16,
        val_every_n_epochs=cfg.val.every_n_epochs,
        grad_clip_norm=cfg.optimizer.get("grad_clip_norm"),
    )

    # Evaluate on test split.
    if world_size > 1:
        torch.distributed.barrier()

    test_loss, test_metrics = validate(
        model,
        test_loader,
        loss_fn,
        procrustes_align=True,
        compute_multiview_error=True,
        verbose=rank == 0,
        device=rank_device(rank),
        use_bf16=cfg.precision.bf16,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")
        logger.info(f"Final test loss       | {test_loss:.4f}")
        for metric in test_metrics:
            logger.info(f"Final test {metric:10s} | {test_metrics[metric]:.4f}")
        results = ", ".join([f"{test_metrics[_m]:.4f}" for _m in test_metrics])
        if wandb_run is not None:
            wandb_run.log(
                {f"test/{metric}": test_metrics[metric] for metric in test_metrics}
            )

        # Log experiment summary.
        exp_info = ", ".join(model_info + train_info)
        log = f"seed={seed}, {exp_info}, {results} \n"
        with open(Path(exp_root) / f"results_{test_dset}.log", "a") as f:
            f.write(log)

        # Save final checkpoint once more after final test logging.
        save_checkpoint(ckpt_path, cfg, model)
        logger.info(f"Seed run end | seed={seed}")
        if wandb_run is not None:
            wandb.finish()
        logger.remove(logger_id)

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="config", config_path="./configs", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train one MVT per seed in ``system.random_seed``, on one or more GPUs.

    Args:
        cfg: Hydra configuration object.
    """
    seed_values = cfg.system.random_seed
    if OmegaConf.is_config(seed_values):
        seed_values = OmegaConf.to_container(seed_values, resolve=True)
    if isinstance(seed_values, int):
        seed_values = [seed_values]
    elif isinstance(seed_values, list):
        if len(seed_values) == 0:
            raise ValueError("system.random_seed must not be an empty list.")
        seed_values = [int(seed) for seed in seed_values]
    else:
        raise ValueError("system.random_seed must be an int or a list of ints.")

    cfg.system.random_seed = seed_values[0] if len(seed_values) == 1 else seed_values

    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    note_suffix = f"_{cfg.note}" if cfg.note != "" else ""
    timestamp = sanitize_exp_name(f"{timestamp}{note_suffix}")
    if cfg.output_dir is not None:
        exp_root = Path(cfg.output_dir).expanduser().resolve()
    else:
        exp_root = Path(__file__).parent / "mvt_exps"
    exp_root.mkdir(parents=True, exist_ok=True)

    wandb_run_id = None
    if cfg.wandb.mode is not None:
        run_name = f"{timestamp}_multiseed" if len(seed_values) > 1 else timestamp
        init_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            tags=["mvt"],
            reinit=True,
            notes=cfg.wandb.notes,
        )
        init_run.config.update(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "seed_values": seed_values,
            }
        )
        wandb_run_id = init_run.id
        wandb.finish()

    for seed in seed_values:
        seed_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        seed_cfg.system.random_seed = seed

        world_size = seed_cfg.system.num_gpus
        if world_size > 1:
            rendezvous_file = (
                exp_root / f".torch_distributed_init_{uuid.uuid4().hex}"
            ).resolve()
            try:
                mp.spawn(
                    train_model,
                    args=(
                        world_size,
                        seed_cfg,
                        str(exp_root),
                        timestamp,
                        seed,
                        wandb_run_id,
                        rendezvous_file,
                    ),
                    nprocs=world_size,
                )
            finally:
                rendezvous_file.unlink(missing_ok=True)
        else:
            train_model(
                0,
                world_size,
                seed_cfg,
                str(exp_root),
                timestamp,
                seed,
                wandb_run_id,
                None,
            )


if __name__ == "__main__":
    main()
