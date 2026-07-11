"""Class-frequency-aware augmentation: tiered transforms + minority-class oversampling."""
from torch.utils.data import WeightedRandomSampler

from data.transforms import TIER_CONFIG, resolve_data_config, build_transform


def _compute_class_counts(manifest):
    return manifest["normalized_label"].value_counts()


# inverse-frequency weights for WeightedRandomSampler
def _compute_sample_weights(manifest, class_counts):
    return manifest["normalized_label"].map(lambda label: 1.0 / class_counts[label]).values


class DataAugmentation:
    """Assigns each class a light/moderate/heavy transform tier by frequency,
    and builds a WeightedRandomSampler that oversamples minority classes.
    """

    def __init__(self, manifest, class_to_index, backbone_name="vit_large_patch16_dinov3.lvd1689m",
                 severe_threshold=500, minority_threshold=1000, input_size=512):
        self.manifest = manifest
        self.class_to_index = class_to_index
        self.class_counts = _compute_class_counts(manifest)
        self.severe_threshold = severe_threshold
        self.minority_threshold = minority_threshold
        self.data_config = resolve_data_config(backbone_name, input_size)

        image_size = self.data_config["input_size"][1]
        mean, std = self.data_config["mean"], self.data_config["std"]
        tier_transforms = {
            tier: build_transform(tier, image_size, mean, std) for tier in TIER_CONFIG
        }

        # precomputed once here instead of reverse-scanning class_to_index
        # on every __getitem__ call (that scan used to run per-sample, per-epoch)
        self._label_to_transform = {
            index: tier_transforms[self._tier_for_count(self.class_counts[class_name])]
            for class_name, index in class_to_index.items()
        }

    def _tier_for_count(self, count):
        if count < self.severe_threshold:
            return "heavy"
        if count < self.minority_threshold:
            return "moderate"
        return "light"

    # oversamples minority classes proportional to inverse frequency
    def create_sampler(self):
        sample_weights = _compute_sample_weights(self.manifest, self.class_counts)
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # O(1): dict lookup into the mapping built once in __init__.
    def get_transform(self, label):
        return self._label_to_transform[label]
