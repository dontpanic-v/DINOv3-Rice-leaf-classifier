import numpy as np
import sys
import timm
import torch
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.datasets import RiceDataset, resolve_data_config
from src.models import RiceClassifier
from torch.utils.data import DataLoader

# usage: python evaluate.py <config.yaml> <checkpoint.pth>
config_path = sys.argv[1]
checkpoint_path = sys.argv[2]

with open(config_path) as f:
    config = yaml.safe_load(f)

keep_classes = sorted(config["preprocessing"]["keep_classes"])
class_to_index = {label: i for i, label in enumerate(keep_classes)}
index_to_class = {i: label for label, i in class_to_index.items()}
num_classes = len(keep_classes)

data_config = resolve_data_config(config["model"]["backbone"], config["input_size"])
val_transform = timm.data.create_transform(**data_config, is_training=False)

test_ds = RiceDataset(
    config["data"]["manifest_path"], split="test",
    transform=val_transform, class_to_index=class_to_index)
test_loader = DataLoader(
    test_ds, batch_size=config["batch_size"], shuffle=False,
    num_workers=config["num_workers"], pin_memory=True)

model = RiceClassifier(config["model"]["backbone"], num_classes, config["input_size"])
ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
model.load_state_dict(ckpt["model"])
print(f"loaded {checkpoint_path} (epoch {ckpt.get('epoch')}, score {ckpt.get('best_score')})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = (all_preds == all_labels).mean()
print(f"\ntest accuracy: {accuracy:.4f}")

class_names = [index_to_class[i] for i in range(num_classes)]
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

conf = confusion_matrix(all_labels, all_preds)
fig = plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(
    conf, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted", fontsize=10)
plt.ylabel("Actual", fontsize=10)
plt.title("Confusion Matrix", fontsize=12, fontweight="bold")
plt.xticks(fontsize=7, rotation=45, ha="right")
plt.yticks(fontsize=7, rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved confusion_matrix.png")
