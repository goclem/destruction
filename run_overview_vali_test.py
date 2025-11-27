#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse one training run:

- For a given run_name (e.g. grid_search_20251122-065809_lr1e-4_wc0.05_wd0.1)
  1. Find the corresponding log folder under {paths.models}/logs/*/{run_name}
  2. Load hparams.yaml
  3. For each city in hparams["cities"] and each split (train/valid/test):
       * Load labels_prepost_..._{split}_balanced.zarr
       * Compute label distribution (tiles & patches) for:
           - no destruction (label=0 after mapping)
           - destruction (label=1 after mapping)
           - mask (raw label=255)
  4. For each checkpoint in the run's checkpoints/ directory:
       * Load the fine-tuned model weights
       * Evaluate performance (AUC, accuracy) on:
           - validation set (overall + per city)
           - test set (overall + per city)
  5. Save everything into run_valid_vs_test_performance.csv in the run folder.
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import transformers

from destruction_utilities import paths  # assumes this defines paths.data and paths.models


# ---------------------------------------------------------------------
# Model definition (same architecture as in the training script)
# ---------------------------------------------------------------------

class SiameseModel(nn.Module):
    """
    Same architecture as used in training:
    - ViT encoder (facebook/vit-mae-base or a fine-tuned checkpoint)
    - projection head
    - MLP head over concatenated patch features
    """

    def __init__(self, backbone: str, head_hidden: int = 512):
        super().__init__()
        self.encoder = transformers.ViTModel.from_pretrained(backbone)
        D = self.encoder.config.hidden_size
        self.patch_dim = self.encoder.config.image_size // self.encoder.config.patch_size

        d = D // 2
        self.proj = nn.Sequential(
            nn.Linear(D, d), nn.GELU(),
            nn.Linear(d, d)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(4 * d, head_hidden), nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, 1)
        )

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        out = self.encoder(x).last_hidden_state  # [B, 1+N_patches, D]
        return out[:, 1:, :]                     # drop CLS token → [B, N_patches, D]

    def forward(self, X: torch.Tensor):
        """
        X: [B, 2, 3, H, W]
        Returns:
            D  : [B, 1, P, P]  (per-patch distance)
            Yh : [B, 1, P, P]  (per-patch logits)
        """
        x0, x1 = X[:, 0], X[:, 1]       # [B, 3, H, W]

        H0 = self._encode_tokens(x0)    # [B, N_patches, D]
        H1 = self._encode_tokens(x1)    # [B, N_patches, D]

        H0 = self.proj(H0)              # [B, N_patches, d]
        H1 = self.proj(H1)              # [B, N_patches, d]

        # distance for contrastive loss
        D = (H0 - H1).norm(dim=-1)      # [B, N_patches]

        # build pair representation
        Z = torch.cat(
            [H0, H1, torch.abs(H0 - H1), H0 * H1],
            dim=-1
        )                               # [B, N_patches, 4d]
        Yh = self.mlp_head(Z).squeeze(-1)  # [B, N_patches]

        B = X.size(0)
        P = self.patch_dim
        D = D.view(B, 1, P, P)          # [B, 1, P, P]
        Yh = Yh.view(B, 1, P, P)        # [B, 1, P, P]
        return D, Yh


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

class EvalZarrDataset(Dataset):
    """
    Minimal dataset that reads pre-post tiles and labels from zarr.
    Returns raw numpy arrays; preprocessing is done in the collate_fn.
    """

    def __init__(self, images_zarr: str, labels_zarr: str):
        if not os.path.exists(images_zarr):
            raise FileNotFoundError(f"Images zarr not found: {images_zarr}")
        if not os.path.exists(labels_zarr):
            raise FileNotFoundError(f"Labels zarr not found: {labels_zarr}")

        self.images = zarr.open(images_zarr, mode="r")
        self.labels = zarr.open(labels_zarr, mode="r")

        if len(self.images) != len(self.labels):
            raise ValueError(f"Images and labels length mismatch: {len(self.images)} vs {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.images[idx]   # expected shape: [2, 3, H, W]
        y = self.labels[idx]   # expected shape: [1, Hy, Wy] or [Hy, Wy]
        return x, y


def make_collate_fn(processor, label_map: dict, image_size: int):
    """
    Returns a collate_fn that:
    - stacks images, applies the ViT image processor exactly as in training
    - stacks labels, applies label_map (0/1, NaN for 255) as in training
    """

    # parse label_map values to float (NaN for "NaN")
    parsed_label_map = {}
    for k, v in label_map.items():
        key_int = int(k)
        if isinstance(v, str) and v.lower() == "nan":
            parsed_label_map[key_int] = float("nan")
        else:
            parsed_label_map[key_int] = float(v)

    def collate_fn(batch):
        xs, ys = zip(*batch)  # lists of numpy arrays

        X = torch.from_numpy(np.stack(xs, axis=0))  # [B, 2, 3, H, W]
        # Flatten (B*2, 3, H, W) for the processor
        B, T, C, H, W = X.shape
        X_flat = X.view(-1, C, H, W)
        X_proc = processor(X_flat, return_tensors="pt")["pixel_values"]  # [B*2, 3, H, W]
        X_proc = X_proc.view(B, T, C, H, W)  # [B, 2, 3, H, W]

        Y_np = np.stack(ys, axis=0)  # [B, 1, Hy, Wy] or [B, Hy, Wy]
        if Y_np.ndim == 4:
            Y_np = Y_np[:, 0, ...]   # drop channel if present
        Y = torch.from_numpy(Y_np).float()  # [B, Hy, Wy]

        # apply label_map as in training
        for key_int, mapped_val in parsed_label_map.items():
            if np.isnan(mapped_val):
                repl = torch.full_like(Y, float("nan"))
            else:
                repl = torch.full_like(Y, mapped_val)
            Y = torch.where(Y == float(key_int), repl, Y)

        return X_proc, Y

    return collate_fn


# ---------------------------------------------------------------------
# Helpers for run discovery, hparams & label map
# ---------------------------------------------------------------------

def find_run_dir(run_name: str, model_name: str | None) -> tuple[str, str]:
    """
    Find the directory {paths.models}/logs/{model_name}/{run_name}.
    If model_name is None, search all model_name directories.
    Returns (run_dir, model_name).
    """
    logs_root = os.path.join(paths.models, "logs")

    if model_name is not None:
        candidate = os.path.join(logs_root, model_name, run_name)
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"Run directory not found: {candidate}")
        return candidate, model_name

    # search all model_name subdirectories
    candidates = []
    for mname in os.listdir(logs_root):
        mdir = os.path.join(logs_root, mname)
        if not os.path.isdir(mdir):
            continue
        rdir = os.path.join(mdir, run_name)
        if os.path.isdir(rdir):
            candidates.append((rdir, mname))

    if not candidates:
        raise FileNotFoundError(f"No run directory named '{run_name}' found under {logs_root}")
    if len(candidates) > 1:
        print(f"Warning: multiple runs with name {run_name} found; using the first one:")
        for rd, mn in candidates:
            print(f"  - {rd}")
    return candidates[0]


def load_hparams(run_dir: str) -> dict:
    hparams_file = os.path.join(run_dir, "hparams.yaml")
    if not os.path.exists(hparams_file):
        raise FileNotFoundError(f"hparams.yaml not found in run dir: {hparams_file}")
    with open(hparams_file, "r") as f:
        hparams = yaml.safe_load(f)
    return hparams


def parse_label_map_from_hparams(hparams: dict) -> dict:
    """
    hparams['label_map'] was stored with JSON/YAML-friendly values.
    Convert keys to int, values to either float (incl NaN) or str 'NaN' → NaN.
    """
    raw = hparams.get("label_map", {0: 0, 1: 0, 2: 1, 3: 1, 255: "NaN"})
    label_map = {}
    for k, v in raw.items():
        key_int = int(k)
        if isinstance(v, str) and v.lower() == "nan":
            label_map[key_int] = float("nan")
        else:
            label_map[key_int] = float(v)
    return label_map


def parse_buffer_flag(val) -> bool:
    """
    buffer_around_destruction in hparams may be bool or string.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)


# ---------------------------------------------------------------------
# Label distribution computation
# ---------------------------------------------------------------------

def compute_label_stats_for_city_split(
    city: str,
    split: str,
    image_size: int,
    patch_size: int,
    buffer_flag: bool,
    label_map: dict,
) -> dict | None:
    """
    Compute label distribution (tiles & patches) for one city & one split
    using the pre-post balanced labels zarr.

    Returns a dict with:
    - city, split
    - tiles_total, tiles_with_no_destruction, tiles_with_destruction, tiles_with_mask
    - patches_total, patches_no_destruction, patches_destruction, patches_mask
    or None if file not found.
    """
    labels_zarr = (
        f"{paths.data}/{city}/zarr/"
        f"labels_prepost_img{image_size}_pat{patch_size}_buf{buffer_flag}_{split}_balanced.zarr"
    )

    if not os.path.exists(labels_zarr):
        print(f"[WARN] Labels zarr not found for city={city}, split={split}: {labels_zarr}")
        return None

    arr = zarr.open(labels_zarr, mode="r")[:]  # [N, C, Hy, Wy] or [N, Hy, Wy]
    if arr.ndim == 4:
        N, C, Hy, Wy = arr.shape
        raw = arr[:, 0, :, :]  # [N, Hy, Wy]
    elif arr.ndim == 3:
        N, Hy, Wy = arr.shape
        raw = arr
    else:
        raise ValueError(f"Unexpected labels shape for {labels_zarr}: {arr.shape}")

    # raw values are original codes (0,1,2,3,255,...)
    mask_patches = (raw == 255)

    # mapped labels (0 / 1 / NaN) as in training
    Y = raw.astype(np.float32)
    for key_int, mapped_val in label_map.items():
        # NaN stays NaN
        Y = np.where(raw == float(key_int), mapped_val, Y)

    patches_destruction = (Y == 1.0)
    patches_no_destruction = (Y == 0.0)

    patches_total = raw.size
    patches_mask = int(mask_patches.sum())
    patches_destruction_count = int(patches_destruction.sum())
    patches_no_destruction_count = int(patches_no_destruction.sum())

    # tile-level: "has at least one" of that type
    tile_has_mask = mask_patches.reshape(N, -1).any(axis=1)
    tile_has_destruction = patches_destruction.reshape(N, -1).any(axis=1)
    tile_has_no_destruction = patches_no_destruction.reshape(N, -1).any(axis=1)

    stats = {
        "city": city,
        "split": split,
        "tiles_total": int(N),
        "tiles_with_no_destruction": int(tile_has_no_destruction.sum()),
        "tiles_with_destruction": int(tile_has_destruction.sum()),
        "tiles_with_mask": int(tile_has_mask.sum()),
        "patches_total": int(patches_total),
        "patches_no_destruction": patches_no_destruction_count,
        "patches_destruction": patches_destruction_count,
        "patches_mask": patches_mask,
    }
    return stats


def aggregate_label_stats_all_cities(label_stats: dict, split: str) -> dict:
    """
    Sum label stats across cities for a given split.
    label_stats is a dict keyed by (city, split) -> stats_dict.
    """
    keys_to_sum = [
        "tiles_total",
        "tiles_with_no_destruction",
        "tiles_with_destruction",
        "tiles_with_mask",
        "patches_total",
        "patches_no_destruction",
        "patches_destruction",
        "patches_mask",
    ]

    agg = {k: 0 for k in keys_to_sum}
    for (city, sp), stats in label_stats.items():
        if sp != split:
            continue
        for k in keys_to_sum:
            agg[k] += stats.get(k, 0)

    agg["city"] = "ALL"
    agg["split"] = split
    return agg


# ---------------------------------------------------------------------
# Evaluation over validation/test for each checkpoint
# ---------------------------------------------------------------------

def evaluate_split_for_checkpoint(
    model: nn.Module,
    cities: list[str],
    split: str,
    image_size: int,
    patch_size: int,
    buffer_flag: bool,
    batch_size: int,
    label_map: dict,
    device: torch.device,
    processor,
) -> tuple[dict, dict]:
    """
    Evaluate one checkpoint on one split (valid or test).

    Returns:
      global_results : dict with keys ['auc', 'acc', 'n_patches']
      per_city_results : dict[city] -> {'auc', 'acc', 'n_patches'}
    """
    collate_fn = make_collate_fn(processor, label_map, image_size)
    per_city_results: dict[str, dict] = {}

    all_y_true = []
    all_y_score = []

    model.eval()

    for city in cities:
        images_zarr = (
            f"{paths.data}/{city}/zarr/"
            f"images_prepost_img{image_size}_pat{patch_size}_buf{buffer_flag}_{split}_balanced.zarr"
        )
        labels_zarr = (
            f"{paths.data}/{city}/zarr/"
            f"labels_prepost_img{image_size}_pat{patch_size}_buf{buffer_flag}_{split}_balanced.zarr"
        )

        if not (os.path.exists(images_zarr) and os.path.exists(labels_zarr)):
            print(f"[WARN] Missing data for city={city}, split={split} -> skipping in metrics")
            continue

        dataset = EvalZarrDataset(images_zarr, labels_zarr)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        city_y_true = []
        city_y_score = []

        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)  # [B, Hy, Wy] with 0/1/NaN

            with torch.no_grad():
                D, Yh = model(X)  # D, Yh: [B, 1, P, P]

                Ph, Pw = Y.shape[-2], Y.shape[-1]  # target patch grid

                # align predictions to labels as in training (adaptive pooling)
                Dp = F.adaptive_avg_pool2d(D, (Ph, Pw)).squeeze(1)   # [B, Ph, Pw]
                Yhp = F.adaptive_avg_pool2d(Yh, (Ph, Pw)).squeeze(1) # [B, Ph, Pw]

                mask = torch.isnan(Y)  # True where label is NaN (incl. original 255)
                probs = torch.sigmoid(Yhp)

                valid = ~mask
                if not torch.any(valid):
                    continue

                y_true = Y[valid].detach().cpu().numpy().astype(int)
                y_score = probs[valid].detach().cpu().numpy()

            city_y_true.append(y_true)
            city_y_score.append(y_score)

        if not city_y_true:
            continue

        city_y_true = np.concatenate(city_y_true, axis=0)
        city_y_score = np.concatenate(city_y_score, axis=0)

        if len(np.unique(city_y_true)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(city_y_true, city_y_score))
        preds = (city_y_score >= 0.5).astype(int)
        acc = float(accuracy_score(city_y_true, preds))

        per_city_results[city] = {
            "auc": auc,
            "acc": acc,
            "n_patches": int(len(city_y_true)),
        }

        all_y_true.append(city_y_true)
        all_y_score.append(city_y_score)

    if all_y_true:
        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_score = np.concatenate(all_y_score, axis=0)
        if len(np.unique(all_y_true)) < 2:
            auc_global = float("nan")
        else:
            auc_global = float(roc_auc_score(all_y_true, all_y_score))
        preds_global = (all_y_score >= 0.5).astype(int)
        acc_global = float(accuracy_score(all_y_true, preds_global))
        global_results = {
            "auc": auc_global,
            "acc": acc_global,
            "n_patches": int(len(all_y_true)),
        }
    else:
        global_results = {"auc": float("nan"), "acc": float("nan"), "n_patches": 0}

    return global_results, per_city_results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def get_default_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name (e.g. grid_search_20251122-065809_lr1e-4_wc0.05_wd0.1)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name folder under {paths.models}/logs. If omitted, will search all.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation. If None, use batch_size from hparams.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'mps', or 'cpu'. Default: auto-detect.",
    )
    parser.add_argument('--cities', 
                        nargs='+', 
                        type=str, 
                        default=None, 
                        help='List of city names for training and default for evaluation.')

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Locate run dir & load hparams
    # -----------------------------------------------------------------
    run_dir, model_name = find_run_dir(args.run_name, args.model_name)
    print(f"[INFO] Using run_dir={run_dir}, model_name={model_name}")

    hparams = load_hparams(run_dir)

    if args.cities is not None:
        cities = args.cities
    else:
        cities = hparams.get("cities", [])
    
    if isinstance(cities, str):
        # if somehow stored as a single string
        cities = [cities]
    print(f"[INFO] Cities used in this run: {cities}")

    image_size = int(hparams.get("image_size", 224))
    patch_size = int(hparams.get("patch_size", 32))
    buffer_flag = parse_buffer_flag(hparams.get("buffer_around_destruction", True))
    label_map = parse_label_map_from_hparams(hparams)

    eval_batch_size = args.batch_size or int(hparams.get("batch_size", 64))

    if args.device is None:
        device_str = get_default_device_str()
    else:
        device_str = args.device
    device = torch.device(device_str)
    print(f"[INFO] Evaluation device: {device}")

    # backbone path from hparams
    backbone_model_name = hparams.get("backbone_model", "checkpoint-9920")
    backbone_path = os.path.join(paths.models, backbone_model_name)
    print(f"[INFO] Backbone path: {backbone_path}")

    # -----------------------------------------------------------------
    # Compute label distributions (train/valid/test) per city
    # -----------------------------------------------------------------
    print("[INFO] Computing label distributions per city and split...")
    label_stats = {}  # (city, split) -> stats dict
    splits_for_labels = ["train", "valid", "test"]

    for city in cities:
        for split in splits_for_labels:
            stats = compute_label_stats_for_city_split(
                city=city,
                split=split,
                image_size=image_size,
                patch_size=patch_size,
                buffer_flag=buffer_flag,
                label_map=label_map,
            )
            if stats is not None:
                label_stats[(city, split)] = stats

    # Precompute aggregated label stats across cities per split
    label_stats_all = {
        split: aggregate_label_stats_all_cities(label_stats, split)
        for split in splits_for_labels
    }

    # -----------------------------------------------------------------
    # Set up image processor (same as in training)
    # -----------------------------------------------------------------
    processor = transformers.ViTImageProcessor.from_pretrained("facebook/vit-mae-base")

    # -----------------------------------------------------------------
    # Enumerate checkpoints
    # -----------------------------------------------------------------
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpt_files = sorted(
        f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")

    print("[INFO] Found checkpoints:")
    for f in ckpt_files:
        print(f"  - {f}")

    # -----------------------------------------------------------------
    # Evaluate each checkpoint on valid & test + collect all results
    # -----------------------------------------------------------------
    rows = []

    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        print(f"\n[INFO] Evaluating checkpoint: {ckpt_file}")

        # load checkpoint on CPU and pick out state_dict
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        epoch = ckpt.get("epoch", None)
        global_step = ckpt.get("global_step", None)

        # build SiameseModel & load only "model." weights into it
        base_model = SiameseModel(backbone=backbone_path)
        model_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_k = k[len("model."):]  # strip "model."
                model_sd[new_k] = v
        missing, unexpected = base_model.load_state_dict(model_sd, strict=False)
        if missing or unexpected:
            print(f"[WARN] When loading {ckpt_file}: missing_keys={missing}, unexpected_keys={unexpected}")

        base_model.to(device)
        base_model.eval()

        for split in ["valid", "test"]:
            print(f"[INFO]  Split: {split}")
            global_metrics, per_city_metrics = evaluate_split_for_checkpoint(
                model=base_model,
                cities=cities,
                split=split,
                image_size=image_size,
                patch_size=patch_size,
                buffer_flag=buffer_flag,
                batch_size=eval_batch_size,
                label_map=label_map,
                device=device,
                processor=processor,
            )

            # global row ("ALL")
            ls_all = label_stats_all.get(split, {})
            row_global = {
                "run_name": args.run_name,
                "model_name": model_name,
                "checkpoint": ckpt_file,
                "epoch": epoch,
                "global_step": global_step,
                "split": split,
                "city": "ALL",
                "auc": global_metrics.get("auc", float("nan")),
                "accuracy": global_metrics.get("acc", float("nan")),
                "n_patches_metric": global_metrics.get("n_patches", 0),
                # label stats (may be zero if not found)
                "tiles_total": ls_all.get("tiles_total", 0),
                "tiles_with_no_destruction": ls_all.get("tiles_with_no_destruction", 0),
                "tiles_with_destruction": ls_all.get("tiles_with_destruction", 0),
                "tiles_with_mask": ls_all.get("tiles_with_mask", 0),
                "patches_total": ls_all.get("patches_total", 0),
                "patches_no_destruction": ls_all.get("patches_no_destruction", 0),
                "patches_destruction": ls_all.get("patches_destruction", 0),
                "patches_mask": ls_all.get("patches_mask", 0),
            }
            rows.append(row_global)

            # per-city rows
            for city in cities:
                metrics_city = per_city_metrics.get(city, None)
                ls_city = label_stats.get((city, split), None)

                row_city = {
                    "run_name": args.run_name,
                    "model_name": model_name,
                    "checkpoint": ckpt_file,
                    "epoch": epoch,
                    "global_step": global_step,
                    "split": split,
                    "city": city,
                    "auc": float("nan"),
                    "accuracy": float("nan"),
                    "n_patches_metric": 0,
                    "tiles_total": 0,
                    "tiles_with_no_destruction": 0,
                    "tiles_with_destruction": 0,
                    "tiles_with_mask": 0,
                    "patches_total": 0,
                    "patches_no_destruction": 0,
                    "patches_destruction": 0,
                    "patches_mask": 0,
                }

                if metrics_city is not None:
                    row_city["auc"] = metrics_city.get("auc", float("nan"))
                    row_city["accuracy"] = metrics_city.get("acc", float("nan"))
                    row_city["n_patches_metric"] = metrics_city.get("n_patches", 0)

                if ls_city is not None:
                    row_city["tiles_total"] = ls_city.get("tiles_total", 0)
                    row_city["tiles_with_no_destruction"] = ls_city.get("tiles_with_no_destruction", 0)
                    row_city["tiles_with_destruction"] = ls_city.get("tiles_with_destruction", 0)
                    row_city["tiles_with_mask"] = ls_city.get("tiles_with_mask", 0)
                    row_city["patches_total"] = ls_city.get("patches_total", 0)
                    row_city["patches_no_destruction"] = ls_city.get("patches_no_destruction", 0)
                    row_city["patches_destruction"] = ls_city.get("patches_destruction", 0)
                    row_city["patches_mask"] = ls_city.get("patches_mask", 0)

                rows.append(row_city)

    # -----------------------------------------------------------------
    # Build DataFrame & save CSV
    # -----------------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, "run_valid_vs_test_performance.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved results to: {csv_path}")


if __name__ == "__main__":
    main()
