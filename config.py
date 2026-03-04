"""Hyperparameters, paths, class mapping, and dataset source configs."""

CONFIG = {
    # Paths
    "base_dir": "/content/datasets",
    "manifest_path": "/content/final_manifest.csv",

    # Split
    "train_ratio": 0.8,
    "val_ratio": 0.10,
    "test_ratio": 0.10,

    # Model
    "backbone_name": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "num_classes": 8,
    "input_size": 512,
    "freeze_backbone": True,

    # Training
    "batch_size": 32,
    "num_workers": 12,
    "stage1_epochs": 15,
    "stage2_epochs": 5,
    "stage1_lr": 1e-3,
    "stage2_lr": 1e-4,
    "weight_decay": 0.02,
    "label_smoothing": 0.1,

    # Checkpoints
    "best_model_path": "/content/best_model.pth",
    "drive_backup_path": "/content/drive/MyDrive/DinoLarge/best_model.pth",
}

# Maps raw folder/csv labels to normalized class names.
CLASS_MAPPING = {
    "bacterial_leaf_blight": "Bacterial Leaf Blight",
    "Bacterial Leaf Blight": "Bacterial Leaf Blight",

    "LeafBlast": "Blast",
    "leaf_blast": "Blast",
    "Leaf Blast": "Blast",

    "BrownSpot": "Brown Spot",
    "brown_spot": "Brown Spot",
    "Brown Spot": "Brown Spot",

    "Sheath Blight": "Sheath Blight",
    "sheath_blight": "Sheath Blight",

    "leaf_scald": "Leaf Scald",
    "Leaf scald": "Leaf Scald",

    "narrow_brown_spot": "Narrow Brown Spot",
    "Narrow Brown Leaf Spot": "Narrow Brown Spot",

    "Hispa": "Hispa",
    "Rice Hispa": "Hispa",
    "rice_hispa": "Hispa",

    "Neck_Blast": "Neck Blast",
    "neck_blast": "Neck Blast",

    "Tungro": "Tungro",
    "tungro": "Tungro",

    "Healthy": "Healthy",
    "healthy": "Healthy",
    "Healthy Rice Leaf": "Healthy",
}

# Dataset source configs for manifest generation.
def get_dataset_configs(base_dir):
    return [
        {"root_path": f"{base_dir}/LabelledRice/Labelled", "dataset_name": "labelled_rice", "label_source": "folder"},
        {"root_path": f"{base_dir}/RiceLeafs/train", "dataset_name": "rice_leafs_train", "label_source": "folder"},
        {"root_path": f"{base_dir}/RiceLeafs/validation", "dataset_name": "rice_leafs_val", "label_source": "folder"},
        {"root_path": f"{base_dir}/RiceDiseaseDataset/train", "dataset_name": "rice_disease_dataset_train", "label_source": "folder"},
        {"root_path": f"{base_dir}/RiceDiseaseDataset/validation", "dataset_name": "rice_disease_dataset_val", "label_source": "folder"},
        {"root_path": f"{base_dir}/RiceLeafsDisease/train", "dataset_name": "rice_leafs_disease_train", "label_source": "folder"},
        {"root_path": f"{base_dir}/RiceLeafsDisease/validation", "dataset_name": "rice_leafs_disease_val", "label_source": "folder"},
        {"root_path": f"{base_dir}/Rice_Leaf_Diease/Rice_Leaf_Diease/train", "dataset_name": "rice_leaf_disease_train", "label_source": "folder"},
        {"root_path": f"{base_dir}/Rice_Leaf_Diease/Rice_Leaf_Diease/test", "dataset_name": "rice_leaf_disease_test", "label_source": "folder"},
    ]

# Classes to keep after filtering.
KEEP_CLASSES = [
    "Healthy", "Blast", "Brown Spot", "Hispa",
    "Bacterial Leaf Blight", "Leaf Scald", "Narrow Brown Spot", "Sheath Blight",
]
