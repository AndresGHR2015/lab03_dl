import torch.nn as nn


class MultiTaskMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.gender_head = nn.Linear(128, 2)
        self.age_head = nn.Linear(128, 1)

    def forward(self, x):
        shared_output = self.shared(x)
        gender_output = self.gender_head(shared_output)
        age_output = self.age_head(shared_output)
        return gender_output, age_output


class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.gender_head = nn.Linear(256, 2)
        self.age_head = nn.Linear(256, 1)

    def forward(self, x):
        feature_output = self.features(x)
        shared_output = self.shared(feature_output)
        gender_output = self.gender_head(shared_output)
        age_output = self.age_head(shared_output)
        return gender_output, age_output


class MultiTaskResNet(nn.Module):
    def __init__(self, freeze_backbone=True, pretrained=True):
        super().__init__()
        from torchvision import models
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.base = models.resnet18(weights=weights)
        in_features = self.base.fc.in_features

        if freeze_backbone:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

        self.base.fc = nn.Identity()
        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, x):
        base_output = self.base(x)
        gender_output = self.gender_head(base_output)
        age_output = self.age_head(base_output)
        return gender_output, age_output
