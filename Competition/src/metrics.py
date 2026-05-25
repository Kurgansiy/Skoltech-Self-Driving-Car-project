def compute_iou(preds_bin, gt, ignore_val=255):
    valid = gt != ignore_val

    pred = preds_bin[valid].bool()
    target = gt[valid].bool()

    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()

    return intersection / (union + 1e-8)
