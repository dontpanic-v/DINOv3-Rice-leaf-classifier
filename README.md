# Rice Leaf Disease Classification

DINOv3 ViT-Huge+ transfer learning for 8-class rice leaf disease detection. Built for Google Colab.

## Classes

Bacterial Leaf Blight, Blast, Brown Spot, Healthy, Hispa, Leaf Scald, Narrow Brown Spot, Sheath Blight.

## Pipeline

1. **Manifest** — crawl 4 Kaggle datasets, normalize labels across naming variants, compute perceptual hashes
2. **Validation** — deduplicate via phash, remove corrupted files, filter below 1024x1024
3. **Split** — stratified 80/10/10 train/val/test
4. **Augmentation** — three tiers (light/moderate/heavy) by class frequency, `WeightedRandomSampler` for imbalanced classes
5. **Training** — two-stage: frozen backbone then full fine-tune at 1/10th lr. bf16, gradient clipping, cosine annealing
6. **Evaluation** — per-class precision/recall/F1, confusion matrix
7. **Interpretability** — multi-layer Grad-CAM, PCA on patch tokens, t-SNE on CLS embeddings

## Model

| Component | Detail |
|-----------|--------|
| Backbone | `vit_huge_plus_patch16_dinov3.lvd1689m` (timm, self-supervised) |
| Head input | CLS token concatenated with mean-pooled patch tokens |
| Head | BatchNorm1d -> Linear -> GELU -> Dropout(0.2) -> Linear -> GELU -> Dropout(0.2) -> Linear |
| Prefix tokens | 5 (1 CLS + 4 registers, skipped during pooling) |

## Project Structure

```
rice-leaf-disease/
  config.py                     Hyperparameters, paths, class mapping, dataset configs
  main.py                       Entry point — wires all stages together
  requirements.txt
  data/
    manifest.py                 Crawl datasets, build raw CSV
    validation.py               Phash deduplication
    postprocessing.py           Filter corrupted/low-res/unwanted classes
    dataset.py                  PyTorch Dataset
    split.py                    Stratified train/val/test split
    transforms.py               Tier-based transform builder
    augmentation.py             Class weighting + sampler + tier assignment
  models/
    classifier.py               RiceClassifier: DINOv3 backbone + MLP head
  training/
    engine.py                   train_one_epoch, validate
    trainer.py                  Two-stage loop, checkpointing, training curves
  evaluation/
    metrics.py                  Test inference, classification report, confusion matrix
    gradcam.py                  Hook-based Grad-CAM with context manager
    plots.py                   Multi-layer Grad-CAM grid, PCA, t-SNE
  util/
    images.py                   Image discovery, safe loading, perceptual hashing
```

## Datasets

All sourced from Kaggle:

| Dataset | Slug |
|---------|------|
| RiceLeafs + LabelledRice | `shayanriyaz/riceleafs` |
| RiceDiseaseDataset | `minhhuy2810/rice-diseases-image-dataset` |
| RiceLeafsDisease | `dedeikhsandwisaputra/rice-leafs-disease-dataset` |
| Rice_Leaf_Disease | `loki4514/rice-leaf-diseases-detection` |

## Usage

Download Kaggle datasets first (Kaggle API required), then:

```bash
python main.py
```

Or run stages individually:

```python
from main import build_manifest, validate_and_clean, verify_paths, train, evaluate, visualize

build_manifest()
validate_and_clean()
verify_paths()
model, trainer, train_dataset, test_dataset, test_loader, augmentation = train()
evaluate(model, test_loader, train_dataset.class_to_index)
visualize(model, test_dataset, test_loader, augmentation)
```

## Key Design Decisions

**CLS + patch mean pooling** — DINOv3 distributes spatial information across patch tokens. Concatenating CLS with averaged patches gives both global semantics and local texture signal (following the paper's Section 6.2.1).

**Three augmentation tiers** — classes range from ~200 to ~2000+ images. Augmentation intensity is proportional to how much a class needs it, rather than uniform heavy augmentation.

**bf16 over fp16** — DINOv3 was pre-trained in bf16. Using fp16 introduces numerical mismatches and requires a GradScaler.

**Two-stage training** — Stage 1 trains only the head on frozen features, establishing a good classifier. Stage 2 fine-tunes the backbone at 1/10th lr to avoid catastrophic forgetting.

## Requirements

```
torch, torchvision, timm, imagehash, pandas, numpy, Pillow, scikit-learn, seaborn, matplotlib
```

## License

MIT
