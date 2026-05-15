from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def _replace_last_linear(module: nn.Module, num_classes: int) -> bool:
    """
    递归替换模型里最后一个 Linear。
    返回是否替换成功。
    """
    # 常见路径优先
    if hasattr(module, "proj") and isinstance(module.proj, nn.Linear):
        in_features = module.proj.in_features
        module.proj = nn.Linear(in_features, num_classes)
        return True

    if hasattr(module, "projection") and isinstance(module.projection, nn.Linear):
        in_features = module.projection.in_features
        module.projection = nn.Linear(in_features, num_classes)
        return True

    children = list(module.named_children())
    for name, child in reversed(children):
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            setattr(module, name, nn.Linear(in_features, num_classes))
            return True
        if _replace_last_linear(child, num_classes):
            return True
    return False


def load_slow_r50_charades(num_classes: int, device: torch.device) -> nn.Module:
    """
    加载 pytorchvideo 的 slow_r50，并替换最后分类头。
    优先从本地 torch hub cache 加载。
    """
    hub_dir = Path(torch.hub.get_dir())
    candidates = [
        hub_dir / "facebookresearch_pytorchvideo_main",
        hub_dir / "facebookresearch_pytorchvideo_master",
    ]

    local_repo = None
    for p in candidates:
        if p.exists():
            local_repo = p
            break

    if local_repo is not None:
        print(f"[INFO] load slow_r50 from local hub cache: {local_repo}")
        model = torch.hub.load(str(local_repo), "slow_r50", pretrained=True, source="local")
    else:
        print("[WARN] local hub cache not found, fallback to online torch.hub.load")
        model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)

    replaced = _replace_last_linear(model, num_classes=num_classes)
    if not replaced:
        raise RuntimeError("Failed to replace final classification head in slow_r50.")

    model = model.to(device)
    return model


def freeze_backbone_except_head(model: nn.Module):
    """
    先只训练最后分类头，适合作为第一轮 baseline。
    """
    for p in model.parameters():
        p.requires_grad = False

    # 解冻最后一个线性层
    found = False
    for m in model.modules():
        if isinstance(m, nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
            found = True
    if not found:
        raise RuntimeError("No Linear layer found to unfreeze.")