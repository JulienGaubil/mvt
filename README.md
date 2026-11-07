<h1 align="center">MVT: Multi-View Transformers from Scratch</h1>

<div align="center">
  <strong>A unified codebase to train multi-view transformers for 3D reconstruction</strong>
</div>

---

## 📖 Overview

This repository contains a **unified multi-view transformer (MVT)** implementation for feed-forward 3D reconstruction, trained from scratch on top of a frozen image encoder. A single model covers the design choices of [DUSt3R](https://github.com/naver/dust3r) and [VGGT](https://github.com/facebookresearch/vggt):
- 🖼️ **Encoder**: frozen CroCoV2 or DINOv2 ViT-L, with 4 register tokens per view.
- 🔀 **Cross-view attention**: cross-attention to the other views (DUSt3R) or global attention over all views (VGGT).
- 🔗 **Weight sharing**: separate per-view weights (DUSt3R) or weights shared across views (VGGT).
- 🎯 **Output**: pointmaps in the first-view frame (DUSt3R), depth and camera poses (VGGT), or both; from a linear head on the last block or a dense multi-layer head.
- 🧪 **MLP-free variants**: view-attention blocks without MLP sublayers.
- 📐 **Losses**: DUSt3R's confidence-weighted pointmap loss or VGGT's losses, for two or more views.

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#️-training)
- [Loading a Trained MVT](#-loading-a-trained-mvt)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)

---

## 🚀 Quick Start

**1. Clone the repository and submodules**
```bash
git clone --recursive https://github.com/JulienGaubil/mvt.git
cd mvt
```

**2. Build a conda environment and install dependencies**
```bash
conda create -n mvt python=3.10
conda activate mvt

pip install -r requirements.txt
pip install -r thirdp/dust3r/requirements.txt
pip install -r thirdp/vggt/requirements.txt
```

**3. Download the CroCoV2 encoder**

Download the [CroCoV2](https://github.com/naver/croco) ViT-L encoder weights to `pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth` (or set `model.crocov2_checkpoint`). DINOv2 is loaded through `torch.hub`.
```bash
mkdir -p pretrained_models
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P pretrained_models/
```

**4. Optional: Connect to Weights & Biases**

We use Weights & Biases for experiment logging. Set your username in the `wandb.entity` field of [configs/config.yaml](configs/config.yaml), or run with `wandb.mode=offline`.

All commands below are run from the repository root.


## 📦 Dataset Preparation

All datasets are expected under `data/`; set `dataset.ROOT` for other locations.

| Dataset | Config | Default root |
|---------|--------|--------------|
| HM3D (Habitat) | `dataset=habitat` | `data/habitat` |
| BlendedMVS | `dataset=blendedmvs` | `data/blendedMVS` |

**HM3D.** Follow the [instructions](https://github.com/naver/dust3r/tree/main/datasets_preprocess/habitat) from DUSt3R's codebase to render the Habitat dataset with habitat-sim. We train on 100k samples, and use the two halves of the scenes of DUSt3R's validation set for validation and test.

**BlendedMVS.** Download and preprocess BlendedMVS following DUSt3R's [instructions](https://github.com/naver/dust3r/tree/main/datasets_preprocess). Scenes with `seq_low % 10 > 0` are used for training (~100k pairs), and the remaining scenes are split in two halves for validation and test (~1k pairs each).


## 🏋️ Training

[train.py](train.py) trains an MVT ([mvt/model.py](mvt/model.py), configured in [configs/model/mvt.yaml](configs/model/mvt.yaml)) and evaluates it on the test split. By default, MVTs are trained on HM3D at 224×224 with DUSt3R's pointmap loss, a batch size of 16, and AdamW (learning rate 1e-4, weight decay 0.01, gradient clipping at 1) for 10 epochs, with half an epoch of linear warmup followed by cosine decay.

```bash
# DUSt3R-style MVT: CroCoV2, cross-attention, separate per-view weights.
python train.py model.encoder_init=crocov2 model.cva_param=cross_attn \
    model.view_weight_sharing=false

# VGGT-style MVT: DINOv2, global attention, shared weights.
python train.py model.encoder_init=dinov2 model.cva_param=global_attn \
    model.view_weight_sharing=true

# Dense multi-layer head, and MLP-free variants.
python train.py model.head_type=dense
python train.py model.use_ffn_layers=false
python train.py model.use_ffn_layers=false model.num_mvt_blocks=24 model.embed_dim=1216

# VGGT outputs (depth and camera poses) and losses.
python train.py model.output_param=factorised loss=vggt_loss
```

Other options: `dataset=blendedmvs`, `dataset.num_views=4` with `model.num_views=4` (Habitat supports 2 to 5 views), `system.num_gpus=4` for multi-GPU training, and `system.random_seed=[0,1,2]` for one run per seed. Runs are saved to `mvt_exps/<experiment_name>/` with their config, checkpoint and training log.


## 📂 Loading a Trained MVT

```python
from mvt import MVT

model = MVT.from_pretrained("mvt_exps/<experiment_name>")  # eval mode
preds = model(view1_images, view2_images)  # images (B, 3, H, W) in [-1, 1]

# preds["pointmaps"] holds one dict per view, with "pts3d" (B, H, W, 3) in the
# first-view frame and "conf" (B, H, W).
```


## 📝 Acknowledgements

This codebase builds upon [DUSt3R](https://github.com/naver/dust3r), [CroCo](https://github.com/naver/croco), [VGGT](https://github.com/facebookresearch/vggt), [DINOv2](https://github.com/facebookresearch/dinov2) and [Probe3D](https://github.com/mbanani/probe3d). We thank the authors for open-sourcing their great work.


## 📌 Citation

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{gaubil2026understanding,
  title={{Understanding Multi-View Transformers}},
  author={Gaubil, Julien and Star{\'y}, Michal and Martinez, Louis and Geiger, Andreas and Ovsjanikov, Maks and Tewari, Ayush and Sitzmann, Vincent},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
