import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from cloth_tool.dataset import ATTRIBUTES, REGRESSION_ATTRS


class ClothClassifier(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # classification heads — one per categorical attribute
        self.heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, len(classes)),
            )
            for attr, classes in ATTRIBUTES.items()
        })

        # regression heads — one per continuous attribute, output in [0, 1]
        self.reg_heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1),
                nn.Sigmoid(),
            )
            for attr in REGRESSION_ATTRS
        })

    def forward(self, x):
        features = self.backbone(x)
        cls_out = {attr: head(features) for attr, head in self.heads.items()}
        reg_out = {attr: head(features).squeeze(1) for attr, head in self.reg_heads.items()}
        return cls_out, reg_out

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
