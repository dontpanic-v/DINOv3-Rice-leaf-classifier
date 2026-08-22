import json
import numpy as np
import os
import random
import shutil
import subprocess
import sys
import time
import timm
import torch
import yaml

from src.datasets import RiceDataset, resolve_data_config, build_transform, TIER_CONFIG
from src.engine import train_one_epoch, validate
from src.models import RiceClassifier, freeze_backbone, unfreeze_backbone
from src.preprocessing import split_manifest
from src.utils import save_checkpoint, load_checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# usage: python main.py <config.yaml>
config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f)
print(config)

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# dataset — class_to_index derived from keep_classes, not hardcoded
manifest_path = config["data"]["manifest_path"]
keep_classes = sorted(config["preprocessing"]["keep_classes"])
class_to_index = {label: i for i, label in enumerate(keep_classes)}
num_classes = len(keep_classes)

split_manifest(
    manifest_path,
    config["data"]["train_ratio"],
    config["data"]["val_ratio"],
    config["data"]["test_ratio"],
    config["seed"],
)

# resolve normalization values from backbone without downloading weights
data_config = resolve_data_config(config["model"]["backbone"], config["input_size"])
image_size = data_config["input_size"][1]
mean, std = data_config["mean"], data_config["std"]

# training: tier-based augmentation keyed on class frequency
import pandas as pd
train_manifest = pd.read_csv(manifest_path)
train_manifest = train_manifest[train_manifest["split"] == "train"]
class_counts = train_manifest["normalized_label"].value_counts()

aug = config["augmentation"]
tier_transforms = {}
for class_name, idx in class_to_index.items():
    count = class_counts[class_name]
    if count < aug["severe_threshold"]:
        tier = "heavy"
    elif count < aug["minority_threshold"]:
        tier = "moderate"
    else:
        tier = "light"
    tier_transforms[idx] = build_transform(tier, image_size, mean, std)

train_ds = RiceDataset(
    manifest_path, split="train",
    tier_transforms=tier_transforms, class_to_index=class_to_index)

# val with standard eval transform
val_transform = timm.data.create_transform(**data_config, is_training=False)
val_ds = RiceDataset(
    manifest_path, split="val",
    transform=val_transform, class_to_index=class_to_index)

# weighted sampler — inverse-frequency oversampling for class imbalance
sample_weights = train_ds.manifest["normalized_label"].map(
    lambda l: 1.0 / class_counts[l]).values
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    sampler=sampler,
    num_workers=config["num_workers"],
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
    persistent_workers=True,
)

# model
model = RiceClassifier(
    config["model"]["backbone"], num_classes, config["input_size"])
staging = config["staging"]
if staging["freeze_backbone"]:
    freeze_backbone(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {n_params:,}")

criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

# optimizer — differential lr when backbone is unfrozen
if staging["freeze_backbone"]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
else:
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(map(id, backbone_params))
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    factor = staging.get("backbone_lr_factor", 0.1)
    optimizer = AdamW([
        {"params": backbone_params, "lr": config["lr"] * factor},
        {"params": head_params, "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])

scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

# birth certificate
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)
shutil.copy(config_path, os.path.join(output_dir, "config.yaml"))
git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
with open(os.path.join(output_dir, "git_hash.txt"), "w") as f:
    f.write(git_hash)

start_epoch = 0
best_score = 0
# weights-only load for stage transitions — fresh optimizer/scheduler
if config.get("init_checkpoint") is not None:
    ckpt = torch.load(config["init_checkpoint"], weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["model"])
# full resume restores optimizer/scheduler/epoch
if config["resume"] is not None:
    start_epoch, best_score = load_checkpoint(
        config["resume"], model, optimizer, scheduler)
    start_epoch += 1

writer = SummaryWriter(output_dir)
print("training started ...")
start_time = time.time()

for epoch in range(start_epoch, config["epochs"]):
    train_stats = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch,
        print_freq=config["print_freq"],
        max_norm=config.get("max_norm", 0),
        writer=writer,
    )
    val_stats = validate(
        model, val_loader, criterion, device,
        print_freq=config["print_freq"],
    )
    scheduler.step()

    if val_stats["accuracy"] > best_score:
        best_score = val_stats["accuracy"]
        print(f"new best accuracy: {best_score:.4f} at epoch {epoch}")
        save_checkpoint(
            model, optimizer, epoch, scheduler, best_score,
            os.path.join(output_dir, "best_epoch.pth"),
        )
    save_checkpoint(
        model, optimizer, epoch, scheduler, best_score,
        os.path.join(output_dir, "last_epoch.pth"),
    )

    # one flat dict per epoch — easy to parse post-run
    stats = {
        "epoch": epoch,
        **{f"train_{k}": v for k, v in train_stats.items()},
        **{f"val_{k}": v for k, v in val_stats.items()},
    }
    with open(os.path.join(output_dir, "log.jsonl"), "a") as f:
        f.write(json.dumps(stats) + "\n")
    for k, v in stats.items():
        if k.startswith("val_") and isinstance(v, (int, float)):
            writer.add_scalar(f"val/{k[4:]}", v, epoch)

writer.close()
total_time = time.time() - start_time
print(f"training complete. best accuracy: {best_score:.4f}")
print(f"total time: {total_time:.0f}s")
