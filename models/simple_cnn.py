import torch
from torch import nn

from age_config import NUM_AGE_CLASSES


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.age_head = nn.Linear(128, NUM_AGE_CLASSES)
        self.gender_head = nn.Linear(128, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu"
                )
                nn.init.zeros_(module.bias)

    def forward(self, images):
        features = self.features(images)
        features = self.global_pool(features)
        features = torch.flatten(features, start_dim=1)
        features = self.shared_fc(features)
        return {
            "age": self.age_head(features),
            "gender": self.gender_head(features)
        }

    def get_gradcam_layer(self):
        """Return the last feature layer used by Grad-CAM."""
        return self.features[-1][-1]  # MaxPool của khối cuối