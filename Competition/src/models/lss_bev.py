import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)


def _build_backbone(name: str):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        extractor = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3,
        )
        return extractor, 1024
    else:
        raise ValueError("Other backbones are not working")


class DepthNet(nn.Module):
    def __init__(self, in_ch: int, num_depth_bins: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_depth_bins, 1),
        )

    def forward(self, x):
        return self.net(x)


class CameraEncoderLSS(nn.Module):
    def __init__(self, backbone: str, feat_ch: int, num_depth_bins: int):
        super().__init__()
        self.backbone, backbone_ch = _build_backbone(backbone)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(backbone_ch, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.depth_net = DepthNet(backbone_ch, num_depth_bins)

    def forward(self, x):
        feats = self.backbone(x)
        depth_logits = self.depth_net(feats)
        cam_feats = self.feat_proj(feats)
        return cam_feats, depth_logits


class LiftSplatShoot(nn.Module):
    def __init__(
        self,
        num_depth_bins: int,
        d_min: float,
        d_max: float,
        bev_h: int,
        bev_w: int,
        bev_res: float,
        bev_x_offset: float,
        bev_y_offset: float,
    ):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_res = bev_res
        self.bev_x_offset = bev_x_offset
        self.bev_y_offset = bev_y_offset

        depths = torch.linspace(d_min, d_max, num_depth_bins)
        self.register_buffer("depths", depths)

    def forward(self, cam_feats, depth_logits, intrinsics, car2cams):
        B, C, Hf, Wf = cam_feats.shape
        D = self.num_depth_bins

        depth_weights = depth_logits.softmax(dim=1)

        voxel_feats = (
            cam_feats.unsqueeze(2) * depth_weights.unsqueeze(1)
        )

        us = torch.linspace(0, 1, Wf, device=cam_feats.device)
        vs = torch.linspace(0, 1, Hf, device=cam_feats.device)
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
        grid_u = grid_u.reshape(-1)
        grid_v = grid_v.reshape(-1)
        N_pix = Hf * Wf

        K = intrinsics[:, :3, :3].float()
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]

        img_h_orig = 256.0
        img_w_orig = 512.0
        scale_x = img_w_orig / Wf
        scale_y = img_h_orig / Hf

        px = grid_u.unsqueeze(0) * Wf * scale_x
        py = grid_v.unsqueeze(0) * Hf * scale_y

        depths = self.depths.to(cam_feats.device)

        px = px.unsqueeze(2).expand(B, N_pix, D)
        py = py.unsqueeze(2).expand(B, N_pix, D)
        d  = depths.unsqueeze(0).unsqueeze(0).expand(B, N_pix, D)

        fx = fx.view(B, 1, 1)
        fy = fy.view(B, 1, 1)
        cx = cx.view(B, 1, 1)
        cy = cy.view(B, 1, 1)

        x_cam = (px - cx) / fx * d
        y_cam = (py - cy) / fy * d
        z_cam = d

        ones = torch.ones_like(x_cam)
        pts_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)

        cam2car = torch.inverse(car2cams.float())
        pts_car = torch.einsum("bij,bndk->bndi", cam2car[:, :3, :], pts_cam)

        x_car = pts_car[..., 0]
        y_car = pts_car[..., 1]

        bev_i = (x_car / self.bev_res - self.bev_x_offset).long()
        bev_j = (y_car / self.bev_res - self.bev_y_offset).long()

        valid = (
            (bev_i >= 0) & (bev_i < self.bev_h) &
            (bev_j >= 0) & (bev_j < self.bev_w)
        )

        voxel_feats_flat = voxel_feats.permute(0, 3, 4, 2, 1).reshape(B, N_pix, D, C)

        bev_map = torch.zeros(B, C, self.bev_h, self.bev_w, device=cam_feats.device)
        bev_count = torch.zeros(B, 1, self.bev_h, self.bev_w, device=cam_feats.device)

        for b in range(B):
            v = valid[b]
            bi = bev_i[b][v]
            bj = bev_j[b][v]
            feats_v = voxel_feats_flat[b][v]
            idx = bi * self.bev_w + bj
            bev_map[b].reshape(C, -1).scatter_add_(1, idx.unsqueeze(0).expand(C, -1), feats_v.T)
            bev_count[b].reshape(1, -1).scatter_add_(1, idx.unsqueeze(0), torch.ones(1, idx.shape[0], device=cam_feats.device))

        bev_map = bev_map / (bev_count + 1e-6)
        return bev_map


class BEVDecoderLSS(nn.Module):
    def __init__(self, in_ch: int, out_size: tuple):
        super().__init__()
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return self.net(x)


class LSSBEVModel(nn.Module):
    def __init__(
        self,
        name: str = "lss_bev",
        backbone: str = "resnet50",
        feat_ch: int = 64,
        num_depth_bins: int = 32,
        d_min: float = 1.0,
        d_max: float = 50.0,
        num_cameras: int = 4,
        out_size: tuple = (188, 126),
        bev_h: int = 188,
        bev_w: int = 126,
        bev_res: float = 0.8,
        bev_x_offset: float = 0.0,
        bev_y_offset: float = -63.0,
    ):
        super().__init__()
        self.name = name
        self.num_cameras = num_cameras

        self.encoder = CameraEncoderLSS(
            backbone=backbone,
            feat_ch=feat_ch,
            num_depth_bins=num_depth_bins,
        )

        self.lss = LiftSplatShoot(
            num_depth_bins=num_depth_bins,
            d_min=d_min,
            d_max=d_max,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_res=bev_res,
            bev_x_offset=bev_x_offset,
            bev_y_offset=bev_y_offset,
        )

        fused_ch = feat_ch * num_cameras
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, feat_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )

        self.decoder = BEVDecoderLSS(in_ch=feat_ch, out_size=out_size)

    def forward(self, images, intrinsics, car2cams):
        bev_maps = []
        for i in range(self.num_cameras):
            img = images[i]
            intr = intrinsics[i]
            c2c = car2cams[i]

            cam_feats, depth_logits = self.encoder(img)
            bev = self.lss(cam_feats, depth_logits, intr, c2c)
            bev_maps.append(bev)

        x = torch.cat(bev_maps, dim=1)
        x = self.fusion(x)
        return self.decoder(x)
