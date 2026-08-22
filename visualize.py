import numpy as np
import sys
import timm
import torch
import yaml

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms

from src.datasets import RiceDataset, resolve_data_config
from src.gradcam import GradCAM
from src.models import RiceClassifier
from torch.utils.data import DataLoader

# usage: python visualize.py <config.yaml> <checkpoint.pth> <mode>
# modes: gradcam, pca, tsne
config_path = sys.argv[1]
checkpoint_path = sys.argv[2]
mode = sys.argv[3]

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


if mode == "gradcam":
    layers_to_show = [20, 25, 31]

    shown_classes = set()
    samples = []
    for i in range(len(test_ds)):
        image, label = test_ds[i]
        if label not in shown_classes:
            shown_classes.add(label)
            record = test_ds.manifest.iloc[i]
            raw_image = Image.open(record["image_path"]).convert("RGB")
            samples.append((image, label, raw_image))
        if len(shown_classes) == num_classes:
            break

    num_cols = 2 + len(layers_to_show)
    fig, axes = plt.subplots(num_classes, num_cols, figsize=(4 * num_cols, 4 * num_classes))

    for row, (image_tensor, label, raw_image) in enumerate(samples):
        axes[row, 0].imshow(raw_image)
        axes[row, 0].set_title(f"GT: {index_to_class[label]}", fontsize=9)
        axes[row, 0].axis("off")

        for col, layer_idx in enumerate(layers_to_show):
            target_layer = model.backbone.blocks[layer_idx]
            with GradCAM(model, target_layer) as grad_cam:
                input_tensor = image_tensor.unsqueeze(0).to(device)
                input_tensor.requires_grad_(True)
                heatmap, predicted_class = grad_cam.generate(input_tensor)

            heatmap_resized = np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    raw_image.size, Image.BICUBIC)) / 255.0

            axes[row, col + 1].imshow(raw_image)
            axes[row, col + 1].imshow(heatmap_resized, cmap="jet", alpha=0.5)
            axes[row, col + 1].set_title(f"Block {layer_idx}", fontsize=9)
            axes[row, col + 1].axis("off")

        axes[row, -1].text(
            0.5, 0.5, f"Pred:\n{index_to_class[predicted_class]}",
            ha="center", va="center", fontsize=10, fontweight="bold")
        axes[row, -1].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_multilayer.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("saved gradcam_multilayer.png")


elif mode == "pca":
    record = test_ds.manifest.iloc[0]
    raw_image = Image.open(record["image_path"]).convert("RGB")

    transform_1024 = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])
    image_1024 = transform_1024(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            features = model.backbone(image_1024)

    patch_features = features[0, model.num_prefix_tokens:].cpu().float().numpy()
    side = int(patch_features.shape[0] ** 0.5)

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_features)
    pca_features = (pca_features - pca_features.min(axis=0)) / (
        pca_features.max(axis=0) - pca_features.min(axis=0))
    pca_image = pca_features.reshape(side, side, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(raw_image)
    ax1.set_title("Input", fontsize=10)
    ax1.axis("off")
    ax2.imshow(pca_image)
    ax2.set_title("DINOv3 PCA Features (1024x1024)", fontsize=10)
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig("pca_1024.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("saved pca_1024.png")


elif mode == "tsne":
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                features = model.backbone(images.to(device))
            cls_tokens = features[:, 0].cpu().float().numpy()
            all_features.append(cls_tokens)
            all_labels.extend(labels.numpy())

    all_features = np.concatenate(all_features)
    all_labels = np.array(all_labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_features)

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    for label_idx in range(num_classes):
        mask = all_labels == label_idx
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=[colors[label_idx]], label=index_to_class[label_idx],
            s=15, alpha=0.7)

    ax.legend(fontsize=8, markerscale=2)
    ax.set_title("t-SNE of DINOv3 CLS Token Embeddings", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("tsne.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("saved tsne.png")


else:
    raise ValueError(f"unknown mode: {mode}")
