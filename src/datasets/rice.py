import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RiceDataset(Dataset):

    def __init__(self, manifest_path, split, transform=None, tier_transforms=None, class_to_index=None):
        self.transform = transform
        self.tier_transforms = tier_transforms
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.class_to_index = class_to_index if class_to_index is not None else self._build_class_index()

    def _build_class_index(self):
        return {label: i for i, label in enumerate(sorted(self.manifest["normalized_label"].unique()))}

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        record = self.manifest.iloc[index]
        image = Image.open(record["image_path"]).convert("RGB")
        label = self.class_to_index[record["normalized_label"]]

        if self.tier_transforms:
            image = self.tier_transforms[label](image)
        elif self.transform:
            image = self.transform(image)

        return image, label
