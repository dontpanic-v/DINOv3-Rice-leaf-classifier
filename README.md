# Rice Leaf Disease Classification

DINOv3 ViT-Huge+ transfer learning for 8-class rice leaf disease detection. Built for Google Colab.

## Classes

Bacterial Leaf Blight, Blast, Brown Spot, Healthy, Hispa, Leaf Scald, Narrow Brown Spot, Sheath Blight.

## Pipeline

1. **Preprocess** — crawl 4 Kaggle datasets, normalize labels, deduplicate via phash, filter corrupted/low-res, stratified split
2. **Train** — two-stage: frozen backbone then full fine-tune at 1/10th lr. bf16, gradient clipping, cosine annealing
3. **Evaluate** — per-class precision/recall/F1, confusion matrix
4. **Visualize** — multi-layer Grad-CAM, PCA on patch tokens, t-SNE on CLS embeddings

## Model

| Component | Detail |
|-----------|--------|
| Backbone | `vit_huge_plus_patch16_dinov3.lvd1689m` (timm, self-supervised) |
| Head input | CLS token concatenated with mean-pooled patch tokens |
| Head | BatchNorm1d -> Linear -> GELU -> Dropout(0.2) -> Linear -> GELU -> Dropout(0.2) -> Linear |
| Prefix tokens | 5 (1 CLS + 4 registers, skipped during pooling) |

## Project Structure

```
configs/
  default.yaml                 Stage 1 config (frozen backbone)
  default_stage2.yaml           Stage 2 config (fine-tune)
src/
  models/
    classifier.py               RiceClassifier: DINOv3 backbone + MLP head
  datasets/
    rice.py                     PyTorch Dataset over manifest CSV
    transforms.py               Tier-based transform builder
  engine.py                     train_one_epoch, validate
  utils.py                      Checkpointing, SmoothedValue, MetricLogger
  gradcam.py                    Hook-based Grad-CAM with context manager
  images.py                     Image discovery, safe loading, perceptual hashing
  preprocessing.py              Manifest building, deduplication, filtering, splitting
main.py                         Training entry point
preprocess.py                   Data pipeline entry point
evaluate.py                     Test-set evaluation
visualize.py                    Grad-CAM, PCA, t-SNE visualization
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
# one-time data preparation
python preprocess.py configs/default.yaml

# stage 1: train head on frozen backbone
python main.py configs/default.yaml

# stage 2: fine-tune full model
python main.py configs/default_stage2.yaml

# test-set evaluation
python evaluate.py configs/default_stage2.yaml outputs/stage2/best_epoch.pth

# visualization (modes: gradcam, pca, tsne)
python visualize.py configs/default_stage2.yaml outputs/stage2/best_epoch.pth gradcam
```

## Key Design Decisions

**YAML config per stage** — each stage is its own config file and run. Stage 2 loads stage 1's best checkpoint via `init_checkpoint`. New experiments are a config file, not a code edit.

**CLS + patch mean pooling** — DINOv3 distributes spatial information across patch tokens. Concatenating CLS with averaged patches gives both global semantics and local texture signal.

**Three augmentation tiers** — classes range from ~200 to ~2000+ images. Augmentation intensity scales with how much a class needs it (heavy/moderate/light by frequency thresholds).

**bf16 over fp16** — DINOv3 was pre-trained in bf16. Using fp16 introduces numerical mismatches and requires a GradScaler.

**Two-stage training** — Stage 1 trains only the head on frozen features. Stage 2 fine-tunes the backbone at 1/10th lr to avoid catastrophic forgetting.

## Requirements

```
torch, torchvision, timm, imagehash, pandas, numpy, Pillow, scikit-learn, seaborn, matplotlib, pyyaml, tensorboard
```

## License

MIT
