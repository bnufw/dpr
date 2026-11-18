from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def isolated_rng(seed, device):
    """Auxiliary feature passes must not advance the training RNG streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

def compute_rotations(images, image_size, task_dict, targets, task_id):
    # Generate the three auxiliary rotations for the first task.
    images_rot = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(1, 4)], 1)
    images_rot = images_rot.view(-1, 3, image_size,  image_size)
    target_rot = torch.stack([(targets * 3 + k)+ len(task_dict[task_id]) -1 for k in range(1, 4)], 1).view(-1)
    return images_rot, target_rot

@torch.no_grad()
def get_task_features(backbone, old_backbone, trn_loader, device):
    backbone.eval()
    if old_backbone is not None:
        old_backbone.eval()
    features_list = []
    old_features_list = []
    labels_list = []
    for images, labels in trn_loader:
        images = images.to(device, non_blocking=True)
        features_list.append(backbone(images))
        labels_list.append(labels.to(device, dtype=torch.long, non_blocking=True))
        if old_backbone is not None:
            old_features_list.append(old_backbone(images))
    old_features = torch.cat(old_features_list) if old_features_list else None
    return torch.cat(features_list), torch.cat(labels_list), old_features
