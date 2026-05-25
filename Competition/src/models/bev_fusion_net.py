import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

def _build_grid_params(x_range, y_range, z_range):
    cell_size = torch.tensor(
        [r[2] for r in (x_range, y_range, z_range)], dtype=torch.float32
    )
    grid_origin = torch.tensor(
        [r[0] + r[2] / 2.0 for r in (x_range, y_range, z_range)], dtype=torch.float32
    )
    grid_shape = torch.tensor(
        [int(round((r[1] - r[0]) / r[2])) for r in (x_range, y_range, z_range)],
        dtype=torch.long,
    )
    return cell_size, grid_origin, grid_shape


class ImageBackbone(nn.Module):
    _CH_S16 = 1024
    _CH_S32 = 2048

    def __init__(self, num_out_channels: int = 96, use_pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        base = resnet50(weights=weights)
        self.feature_net = create_feature_extractor(
            base,
            return_nodes={"layer3": "scale16", "layer4": "scale32"},
        )
        self.proj_s16 = nn.Conv2d(self._CH_S16, num_out_channels, kernel_size=1)
        self.proj_s32 = nn.Conv2d(self._CH_S32, num_out_channels, kernel_size=1)
        self.merge = nn.Sequential(
            nn.Conv2d(num_out_channels, num_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        raw = self.feature_net(imgs)
        coarse = self.proj_s32(raw["scale32"])
        fine   = self.proj_s16(raw["scale16"])
        coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return self.merge(fine + coarse)

class FrustumTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bev_channels: int,
        img_hw: tuple,
        feat_hw: tuple,
        x_range: list,
        y_range: list,
        z_range: list,
        depth_range: list,
        spatial_downsample: int = 1,
    ):
        super().__init__()
        self.img_hw  = img_hw
        self.feat_hw = feat_hw

        cell, origin, shape = _build_grid_params(x_range, y_range, z_range)
        self.register_buffer("cell",   cell)
        self.register_buffer("origin", origin)
        self.register_buffer("shape",  shape)

        self.num_bev_ch = bev_channels
        self.depth_range = depth_range

        frustum = self._make_frustum()
        self.register_buffer("frustum", frustum)
        self.num_depth_bins = frustum.shape[0]

        # depth + feature prediction head
        mid = in_channels
        self.pred_head = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, self.num_depth_bins + self.num_bev_ch, 1),
        )

        if spatial_downsample > 1:
            self.post_pool = nn.Sequential(
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(bev_channels, bev_channels, 3,
                          stride=spatial_downsample, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.post_pool = nn.Identity()

    def _make_frustum(self) -> torch.Tensor:
        iH, iW = self.img_hw
        fH, fW = self.feat_hw
        d_start, d_stop, d_step = self.depth_range
        depth_vals = torch.arange(d_start, d_stop, d_step, dtype=torch.float)
        D = depth_vals.shape[0]
        depth_grid = depth_vals.view(-1, 1, 1).expand(D, fH, fW)
        u_grid = torch.linspace(0, iW - 1, fW).view(1, 1, fW).expand(D, fH, fW)
        v_grid = torch.linspace(0, iH - 1, fH).view(1, fH, 1).expand(D, fH, fW)
        return torch.stack((u_grid, v_grid, depth_grid), dim=-1)  # (D, fH, fW, 3)

    def _unproject_to_ego(self, rot_c2e, trans_c2e, K_inv):
        B, N = trans_c2e.shape[:2]
        pts = self.frustum.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1, -1).clone()
        # back-project: (u*d, v*d, d)
        pts = torch.cat((pts[..., :2] * pts[..., 2:3], pts[..., 2:3]), dim=-1)
        # rotate into ego frame
        R = rot_c2e @ K_inv
        pts = (R.view(B, N, 1, 1, 1, 3, 3) @ pts.unsqueeze(-1)).squeeze(-1)
        pts = pts + trans_c2e.view(B, N, 1, 1, 1, 3)
        return pts  # (B, N, D, fH, fW, 3)

    def _extract_depth_feats(self, flat_feats: torch.Tensor):
        """Run pred_head and split into depth weights and BEV features."""
        B, N, C, fH, fW = flat_feats.shape
        x = flat_feats.view(B * N, C, fH, fW)
        x = self.pred_head(x)
        depth_w = x[:, :self.num_depth_bins].softmax(dim=1)
        bev_f   = x[:, self.num_depth_bins: self.num_depth_bins + self.num_bev_ch]
        # outer product: (BN, C, D, fH, fW)
        voxels = depth_w.unsqueeze(1) * bev_f.unsqueeze(2)
        return (
            voxels.view(B, N, self.num_bev_ch, self.num_depth_bins, fH, fW)
                  .permute(0, 1, 3, 4, 5, 2)
                  .contiguous()
        )  # (B, N, D, fH, fW, C)

    def _voxel_pool(self, ego_pts, voxels):
        """Scatter voxel features into a flat BEV grid."""
        B, N, D, fH, fW, C = voxels.shape
        total = B * N * D * fH * fW

        flat_vox = voxels.reshape(total, C)

        # voxel grid indices
        half_cell = self.cell / 2.0
        grid_idx = ((ego_pts - (self.origin - half_cell)) / self.cell).long()
        grid_idx = grid_idx.view(total, 3)

        nx = int(self.shape[0])
        ny = int(self.shape[1])
        nz = int(self.shape[2])

        # batch index column
        b_col = (
            torch.arange(B, device=flat_vox.device, dtype=torch.long)
            .view(B, 1).expand(B, total // B).reshape(-1, 1)
        )
        grid_idx = torch.cat((grid_idx, b_col), dim=1)  # (total, 4)

        # mask out-of-bounds
        in_bounds = (
            (grid_idx[:, 0] >= 0) & (grid_idx[:, 0] < nx)
            & (grid_idx[:, 1] >= 0) & (grid_idx[:, 1] < ny)
            & (grid_idx[:, 2] >= 0) & (grid_idx[:, 2] < nz)
        )
        flat_vox  = flat_vox[in_bounds]
        grid_idx  = grid_idx[in_bounds]

        linear = (
            grid_idx[:, 3] * (nz * nx * ny)
            + grid_idx[:, 2] * (nx * ny)
            + grid_idx[:, 0] * ny
            + grid_idx[:, 1]
        )

        bev_buf = flat_vox.new_zeros(B * nz * nx * ny, C)
        bev_buf = bev_buf.index_add(0, linear, flat_vox)
        bev_buf = bev_buf.view(B, nz, nx, ny, C).permute(0, 4, 1, 2, 3).contiguous()
        # collapse z dimension
        return torch.cat(bev_buf.unbind(dim=2), dim=1)  # (B, C*nz, nx, ny)

    def forward(self, img_feats, cam2ego_mats, cam_intrinsics):
        rot   = cam2ego_mats[..., :3, :3]
        trans = cam2ego_mats[..., :3, 3]
        K     = cam_intrinsics[..., :3, :3]
        K_inv = torch.inverse(K)

        ego_pts = self._unproject_to_ego(rot, trans, K_inv)
        voxels  = self._extract_depth_feats(img_feats)
        bev     = self._voxel_pool(ego_pts, voxels)
        return self.post_pool(bev)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv_a = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm_a = nn.BatchNorm2d(out_ch)
        self.conv_b = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm_b = nn.BatchNorm2d(out_ch)
        self.act    = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm_a(self.conv_a(x)))
        h = self.norm_b(self.conv_b(h))
        return self.act(h + self.shortcut(x))


def _build_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
    layers = [ResBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(ResBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class GridDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_cfg: tuple = ((2, 96, 2), (2, 192, 2)),
        out_channels: int = 128,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        ch = in_channels
        stage_out_chs = []
        for n_blk, n_ch, st in stage_cfg:
            self.stages.append(_build_stage(ch, n_ch, n_blk, st))
            stage_out_chs.append(n_ch)
            ch = n_ch

        skip_ch = stage_out_chs[-1] + stage_out_chs[0]
        self.skip_merge = nn.Sequential(
            nn.Conv2d(skip_ch, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage_outs = []
        for stage in self.stages:
            x = stage(x)
            stage_outs.append(x)
        deepest = F.interpolate(
            stage_outs[-1], size=stage_outs[0].shape[-2:],
            mode="bilinear", align_corners=False,
        )
        return self.skip_merge(torch.cat([deepest, stage_outs[0]], dim=1))

class OccHead(nn.Module):
    def __init__(self, in_channels: int, target_hw: tuple):
        super().__init__()
        self.target_hw = target_hw
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
        return self.classifier(x)


class BEVFusionNet(nn.Module):
    def __init__(
        self,
        name: str = "bev_fusion_net",
        img_h: int = 256,
        img_w: int = 512,
        backbone_out_ch: int = 192,
        bev_ch: int = 64,
        x_range: tuple = (0.0,   150.0,  150.0 / 188),
        y_range: tuple = (-50.0,  50.0,  100.0 / 126),
        z_range: tuple = (-5.0,   5.0,   10.0),
        depth_range: tuple = (1.0, 150.0, 2.5),
        spatial_downsample: int = 2,
        decoder_stage_cfg: tuple = ((2, 96, 2), (2, 192, 2)),
        decoder_out_ch: int = 128,
        out_size: tuple = (188, 126),
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.name = name
        self.out_size = tuple(out_size)

        feat_h = img_h // 16
        feat_w = img_w // 16

        self.backbone = ImageBackbone(
            num_out_channels=backbone_out_ch,
            use_pretrained=use_pretrained,
        )
        self.view_transform = FrustumTransform(
            in_channels=backbone_out_ch,
            bev_channels=bev_ch,
            img_hw=(img_h, img_w),
            feat_hw=(feat_h, feat_w),
            x_range=list(x_range),
            y_range=list(y_range),
            z_range=list(z_range),
            depth_range=list(depth_range),
            spatial_downsample=spatial_downsample,
        )
        self.bev_proj = nn.Sequential(
            nn.Conv2d(bev_ch, decoder_out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_out_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = GridDecoder(
            in_channels=decoder_out_ch,
            stage_cfg=decoder_stage_cfg,
            out_channels=decoder_out_ch,
        )
        self.occ_head = OccHead(in_channels=decoder_out_ch, target_hw=self.out_size)

    def forward(
        self,
        imgs: torch.Tensor,
        cam2ego_mats: torch.Tensor,
        cam_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        B, N = imgs.shape[:2]

        # encode all cameras in one batch pass
        flat_imgs = imgs.reshape(B * N, *imgs.shape[2:])
        flat_feats = self.backbone(flat_imgs)
        _, Cf, fH, fW = flat_feats.shape
        cam_feats = flat_feats.view(B, N, Cf, fH, fW)

        # lift → splat → BEV
        bev = self.view_transform(cam_feats, cam2ego_mats, cam_intrinsics)
        bev = self.bev_proj(bev)
        bev = self.decoder(bev)
        return self.occ_head(bev)
