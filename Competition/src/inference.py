import shutil
import torch
import numpy as np
import tqdm
import logging
from pathlib import Path
import pandas as pd
from src.models.lss_bev import LSSBEVModel
from src.models.bev_fusion_net import BEVFusionNet

log = logging.getLogger(__name__)

def run_inference(model, test_loader, cfg, device):
    test_dir = Path(cfg.paths.test_dir)
    test_info = pd.read_csv(test_dir / 'info.csv', index_col=0)

    submission_dir = Path(cfg.paths.checkpoint).parent
    grids_dir = submission_dir / 'predicted_static_grids'
    grids_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(test_dir / 'info.csv', submission_dir / 'info.csv')

    model.eval()
    log.info(f"Starting inference")

    with torch.inference_mode():
        for i, (images, intrinsics, car2cams) in enumerate(tqdm.tqdm(test_loader, desc='Inference')):
            images = [img.to(device) for img in images]

            if isinstance(model, LSSBEVModel):
                intrinsics = [m.to(device) for m in intrinsics]
                car2cams = [m.to(device) for m in car2cams]
                logits = model(images, intrinsics, car2cams)
            elif isinstance(model, BEVFusionNet):
                imgs_stacked = torch.stack(images, dim=1)
                intr_stacked = torch.stack([m.to(device) for m in intrinsics], dim=1)
                extr_stacked = torch.stack([m.to(device) for m in car2cams],  dim=1)
                logits = model(imgs_stacked, extr_stacked, intr_stacked)
            else:
                logits = model(images)

            preds = (torch.sigmoid(logits) > 0.5).float()

            original_path = Path(test_info.iloc[i]['predicted_occupancy_grid'])
            preds_path = grids_dir / original_path.name

            np.save(preds_path, preds.view(1, cfg.model.out_size[0], cfg.model.out_size[1]).cpu().numpy().astype(np.int32))

    log.info(f"Inference completed. Results saved to {submission_dir}")
