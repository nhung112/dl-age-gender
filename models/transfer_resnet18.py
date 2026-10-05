import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18MultiTask(nn.Module):
    def __init__(self, num_age_classes=6, num_gender_classes=2, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(features, num_age_classes)
        )

        self.gender_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(features, num_gender_classes)
        )

    def forward(self, images):
        features = self.backbone(images)
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)

        return {
            "age": age_logits,
            "gender": gender_logits
        }