from PIL import Image
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

CAMERA_NAMES = [
    "/camera/inner/frontal/middle",
    "/camera/inner/frontal/far",
    "/side/left/forward",
    "/side/right/forward",
]

INTRINSICS_NAMES = [
    "/camera/inner/frontal/middle/intrinsic_params",
    "/camera/inner/frontal/far/intrinsic_params",
    "/side/left/forward/intrinsic_params",
    "/side/right/forward/intrinsic_params",
]

CAR2CAM_NAMES = [
    "/camera/inner/frontal/middle/car_to_cam",
    "/camera/inner/frontal/far/car_to_cam",
    "/side/left/forward/car_to_cam",
    "/side/right/forward/car_to_cam",
]

GRIDS_NAMES = [
    "gt_occupancy_grid",
]


class BaseDataset(Dataset):
    def __init__(self, data_dir: Path, mode: str = "train"):
        self.mode = mode
        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((256, 512)),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.info = pd.read_csv(data_dir / "info.csv", index_col=0)
        self.images_paths = []
        self.intrinsics_paths = []
        self.car2cam_paths = []
        if self.mode != "test":
            self.static_grids_paths = []

        base_dir = data_dir.parent

        for _, row in self.info.iterrows():
            self.images_paths.append([base_dir / row[name] for name in CAMERA_NAMES])
            self.intrinsics_paths.append([base_dir / row[name] for name in INTRINSICS_NAMES])
            self.car2cam_paths.append([base_dir / row[name] for name in CAR2CAM_NAMES])
            if self.mode != "test":
                self.static_grids_paths.append([base_dir / row[name] for name in GRIDS_NAMES])


    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        images = [self.transform(Image.open(img_path)) for img_path in self.images_paths[idx]]
        intrinsics = [torch.from_numpy(np.load(intr_path)).float() for intr_path in self.intrinsics_paths[idx]]
        car2cams = [torch.from_numpy(np.load(car2cam_path)).float() for car2cam_path in self.car2cam_paths[idx]]

        if self.mode != "test":
            static_grids = [np.load(grid_path) for grid_path in self.static_grids_paths[idx]]
            return images, intrinsics, car2cams, static_grids

        return images, intrinsics, car2cams
