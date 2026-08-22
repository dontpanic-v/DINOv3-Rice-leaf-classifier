import torch
import torch.nn as nn
import timm


class RiceClassifier(nn.Module):

    def __init__(self, backbone_name, num_classes, input_size=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="")
        self.backbone.set_input_size(img_size=input_size)
        self.embed_dim = self.backbone.num_features
        # token layout: [CLS, reg0..reg3, patch0, patch1, ...]
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        # BatchNorm handles feature dimension outliers before the head
        self.bn = nn.BatchNorm1d(self.embed_dim * 2)
        self.head = self._build_head(num_classes)

    def _build_head(self, num_classes):
        dim = self.embed_dim * 2
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 4, num_classes))

    # CLS + average patch tokens — DINOv3 distributes spatial information across
    # patch tokens, so concatenating gives both global semantics and local texture
    def forward(self, x):
        features = self.backbone(x)
        cls_token = features[:, 0]
        patch_tokens = features[:, self.num_prefix_tokens:]
        patch_mean = patch_tokens.mean(dim=1)
        combined = torch.cat([cls_token, patch_mean], dim=-1)
        combined = self.bn(combined)
        return self.head(combined)


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True
    return model
