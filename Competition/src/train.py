import torch
import torch.nn as nn
import tqdm
import numpy as np
import logging
from pathlib import Path
from src.metrics import compute_iou
from src.models.lss_bev import LSSBEVModel
from src.models.bev_fusion_net import BEVFusionNet


log = logging.getLogger(__name__)


def _forward(model, images, intrinsics, car2cams, device):
    images = [img.to(device) for img in images]
    if isinstance(model, LSSBEVModel):
        intrinsics = [m.to(device) for m in intrinsics]
        car2cams = [m.to(device) for m in car2cams]
        return model(images, intrinsics, car2cams)
    if isinstance(model, BEVFusionNet):
        # dataset returns lists of per-camera tensors; stack into (B, N, ...) tensors
        imgs_stacked  = torch.stack(images, dim=1)                    # (B, N, 3, H, W)
        intr_stacked  = torch.stack([m.to(device) for m in intrinsics], dim=1)  # (B, N, 4, 4)
        extr_stacked  = torch.stack([m.to(device) for m in car2cams],  dim=1)   # (B, N, 4, 4)
        return model(imgs_stacked, extr_stacked, intr_stacked)
    return model(images)


def train_model(
    model,
    train_loader,
    val_loader,
    cfg,
    optimizer,
    scheduler,
    criterion,
    device
):
    use_amp = getattr(cfg.train, "amp16", False) and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_iou = 0.0
    checkpoint_path = Path(cfg.paths.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_path.parent / "model_last.pt"

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs} [Train]")
        for images, intrinsics, car2cams, gt in pbar:
            gt = gt[0].to(device).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = _forward(model, images, intrinsics, car2cams, device)
                loss = criterion(logits, gt)
            if loss == 0.0:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        iou_scores = []
        with torch.inference_mode():
            for images, intrinsics, car2cams, gt in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs} [Val]"):
                gt = gt[0].to(device).float()

                logits = _forward(model, images, intrinsics, car2cams, device)
                preds_bin = (torch.sigmoid(logits) > 0.5).float()
                iou_scores.append(compute_iou(preds_bin, gt, cfg.train.ignore_val))

        mean_iou = float(np.mean(iou_scores))
        log.info(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val IoU: {mean_iou:.4f}")

        # always save the latest checkpoint
        torch.save(model.state_dict(), last_checkpoint_path)
        log.info(f"Last checkpoint saved to {last_checkpoint_path}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), checkpoint_path)
            log.info(f"New best model saved to {checkpoint_path} with IoU: {best_iou:.4f}")

    return model
