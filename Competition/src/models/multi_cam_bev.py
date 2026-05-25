import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)

_CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")
_CACHED = {
    "efficientnet_b4": os.path.join(_CACHE_DIR, "efficientnet_b4_rwightman-23ab8bcd.pth"),
    "convnext_tiny":   os.path.join(_CACHE_DIR, "convnext_tiny-983f1562.pth"),
}


def _build_backbone(name: str):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        extractor = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3,
        )
        return extractor, 1024
    else:
        raise ValueError("Not working")


class CameraEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet50", out_dim: int = 512):
        super().__init__()
        self.features, backbone_ch = _build_backbone(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(backbone_ch, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class BEVDecoder(nn.Module):
    def __init__(self, in_dim: int = 512, out_size: tuple = (188, 126)):
        super().__init__()
        self.out_size = out_size
        self.init_h, self.init_w = 6, 4
        self.fc = nn.Linear(in_dim, 256 * self.init_h * self.init_w)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.fc(x).view(B, 256, self.init_h, self.init_w)
        x = self.up(x)
        return x[:, :, :self.out_size[0], :self.out_size[1]]


class MultiCamBEVModel(nn.Module):
    def __init__(
        self,
        name: str = "model",
        backbone: str = "resnet50",
        cam_feat_dim: int = 512,
        num_cameras: int = 4,
        out_size: tuple = (188, 126),
    ):
        super().__init__()
        self.name = name
        self.encoder = CameraEncoder(backbone=backbone, out_dim=cam_feat_dim)
        fused_dim = cam_feat_dim * num_cameras
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        self.decoder = BEVDecoder(in_dim=512, out_size=out_size)

    def forward(self, images):
        feats = [self.encoder(img) for img in images]
        x = torch.cat(feats, dim=1)
        x = self.fusion(x)
        return self.decoder(x)
