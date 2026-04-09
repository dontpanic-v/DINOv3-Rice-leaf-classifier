"""DINOv3-backbone rice leaf disease classifier: CLS + mean-pooled patch tokens -> MLP head."""
import torch
import torch.nn as nn
import timm


class RiceClassifier(nn.Module):
    """DINOv3 ViT backbone with a BatchNorm + MLP head over CLS + mean-pooled patch tokens."""

    def __init__(self, backbone_name="vit_huge_plus_patch16_dinov3.lvd1689m", num_classes=8, freeze_backbone=True, input_size=512):
        super().__init__()
        self.backbone = self._create_backbone(backbone_name, input_size)
        self.embed_dim = self.backbone.num_features

        # token layout: [CLS, reg0..reg3, patch0, patch1, ...]
        self.num_prefix_tokens = self.backbone.num_prefix_tokens  # 5 (1 CLS + 4 registers)

        # BatchNorm handles feature dimension outliers (paper Appendix A.2)
        self.bn = nn.BatchNorm1d(self.embed_dim * 2)
        self.head = self._create_head(num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    # global_pool='' returns raw tokens (batch, seq_len, embed_dim) instead of pooling them
    def _create_backbone(self, backbone_name, input_size):
        model = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="")
        model.set_input_size(img_size=input_size)
        return model

    def _create_head(self, num_classes):
        dim = self.embed_dim * 2
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 4, num_classes))

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    # Paper Section 6.2.1: CLS + average patch tokens for classification.
    def forward(self, x):
        features = self.backbone(x)
        cls_token = features[:, 0]
        patch_tokens = features[:, self.num_prefix_tokens:]
        patch_mean = patch_tokens.mean(dim=1)
        combined = torch.cat([cls_token, patch_mean], dim=-1)
        combined = self.bn(combined)
        logits = self.head(combined)
        return logits
