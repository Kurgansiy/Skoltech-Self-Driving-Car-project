import torch
import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from src.train import train_model
from src.inference import run_inference
from src.dataset import BaseDataset

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

log = logging.getLogger(__name__)


def make_test_loader(cfg):
    test_dataset = BaseDataset(
        data_dir=Path(cfg.paths.test_dir),
        mode='test'
    )
    return DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.train.num_workers
    )


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="config"
)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    log.info(f"Instantiating model: {cfg.model._target_}")
    model = instantiate(cfg.model).to(device)

    if cfg.mode == 'train':
        train_dataset = BaseDataset(
            data_dir=Path(cfg.paths.train_dir),
            mode='train'
        )
        val_dataset = BaseDataset(
            data_dir=Path(cfg.paths.val_dir),
            mode='val'
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True
        )
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            amsgrad=True,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.train.max_lr,
            epochs=cfg.train.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=cfg.train.pct_start,
        )
        criterion = instantiate(cfg.loss)
        log.info(f"Using loss: {cfg.loss._target_}")

        log.info(f"Starting training for {cfg.model.name}")
        train_model(model, train_loader, val_loader, cfg, optimizer, scheduler, criterion, device)

    elif cfg.mode == 'infer':
        infer_ckpt = cfg.paths.get("infer_checkpoint", None)
        checkpoint_path = Path(infer_ckpt) if infer_ckpt else Path(cfg.paths.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        # support both raw state-dict and full training-state dicts
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        log.info(f"Loaded checkpoint from {checkpoint_path}")

        test_loader = make_test_loader(cfg)
        run_inference(model, test_loader, cfg, device)

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
