"""PyTorch Dataset reading images and labels from a manifest CSV split."""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DataStream(Dataset):
    """Reads one manifest split into (image, label_index) pairs, applying a
    per-sample transform or an augmentation tier lookup, if provided.
    """

    def __init__(self, manifest_path, split="train", transform=None, augmentation=None, class_to_index=None):
        self.transform = transform
        self.augmentation = augmentation
        self.manifest = self._load_manifest(manifest_path, split)
        # a shared class_to_index should be passed in and reused across every
        # split's DataStream (see main.py) -- building it independently per
        # split only agrees by coincidence when every class appears in every
        # split, and silently diverges the moment one doesn't
        self.class_to_index = class_to_index if class_to_index is not None else self._build_class_index()

    def _load_manifest(self, manifest_path, split):
        manifest = pd.read_csv(manifest_path)
        return manifest[manifest["split"] == split].reset_index(drop=True)

    # fallback when no shared mapping is passed in; sorted alphabetically for determinism
    def _build_class_index(self):
        class_labels = sorted(self.manifest["normalized_label"].unique())
        return {label: index for index, label in enumerate(class_labels)}

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        record = self.manifest.iloc[index]
        image = Image.open(record["image_path"]).convert("RGB")
        label = self.class_to_index[record["normalized_label"]]

        if self.augmentation:
            image = self.augmentation.get_transform(label)(image)
        elif self.transform:
            image = self.transform(image)
        return image, label
