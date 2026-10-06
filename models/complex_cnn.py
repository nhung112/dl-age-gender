from torch import nn
from age_config import NUM_AGE_CLASSES

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def make_stage(in_channels, out_channels, num_blocks, stride):
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for _ in range(num_blocks - 1):
        layers.append(ResidualBlock(out_channels, out_channels, 1))
    return nn.Sequential(*layers)


def make_stem():
    # 7x7 stride 2 + maxpool stride 2: giảm kích thước 4 lần (224 -> 56)
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def make_head(in_features, hidden, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_features))


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="relu"
            )
            nn.init.zeros_(m.bias)
    for m in model.modules():
        if isinstance(m, ResidualBlock):
            nn.init.zeros_(m.bn2.weight)


class ComplexCNN(nn.Module):
    """
    CNN residual train từ đầu, backbone dùng chung cho cả 2 tác vụ.
    Stem -> stage1 -> stage2 -> stage3 -> stage4 -> GAP -> {age_head, gender_head}
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.stem = make_stem()
        self.stage1 = make_stage(32, 64, num_blocks=2, stride=1)
        self.stage2 = make_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = make_stage(256, 512, num_blocks=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.age_head = make_head(512, 256, NUM_AGE_CLASSES, dropout)
        self.gender_head = make_head(512, 128, 2, dropout)
        initialize_weights(self)

    def forward(self, images):
        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        features = self.pool(x).flatten(1)
        return {
            "age": self.age_head(features),        # (B, 7) 
            "gender": self.gender_head(features),  # (B, 2) 
        }

    def get_gradcam_layer(self, task="age"):
        # Backbone chung nên cả 2 tác vụ dùng cùng một layer.
        # Trả về output của block cuối (sau cộng shortcut + ReLU).
        if task in ("age", "gender"):
            return self.stage4[-1]