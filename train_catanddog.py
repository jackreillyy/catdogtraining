import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18, ResNet18_Weights


# ---------- Device ----------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- Transforms (PIL -> Tensor) ----------
def build_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                  # PIL -> float tensor [0,1]
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ---------- Datasets ----------
def make_datasets(root: str,
                  img_size: int = 224,
                  val_split: float = 0.1,
                  seed: int = 42
                  ) -> Tuple[torch.utils.data.Dataset,
                             torch.utils.data.Dataset,
                             torch.utils.data.Dataset]:
    """
    Returns (train_ds, val_ds, test_ds) where labels are guaranteed {0=cat, 1=dog}.
    Uses OxfordIIITPet with target_types="binary-category".
    """
    train_tf, eval_tf = build_transforms(img_size)

    # Base datasets (PIL loader by default). download=True is fine if internet/cache present.
    base_trainval = OxfordIIITPet(root=root, split="trainval",
                              target_types="binary-category", download=True, transform=train_tf)
    base_test     = OxfordIIITPet(root=root, split="test",
                              target_types="binary-category", download=True, transform=eval_tf)


    # Split train/val
    val_len = int(len(base_trainval) * val_split)
    train_len = len(base_trainval) - val_len

    full_train, full_val = random_split(
        base_trainval, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # Ensure val uses eval transforms (swap the transform on the underlying dataset)
    # random_split returns a Subset; we override its dataset's transform for eval
    full_val.dataset.transform = eval_tf

    # Tiny wrapper to expose classes for logging
    class_names = ["cat", "dog"]
    for subset in (full_train, full_val, base_test):
        setattr(subset, "classes", class_names)

    return full_train, full_val, base_test


# ---------- Model ----------
def make_model(num_classes: int = 2, pretrained: str = "none", finetune: bool = True) -> nn.Module:
    if pretrained.lower() == "imagenet":
        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as e:
            print(f"[warn] Could not load ImageNet weights ({e}). Using random init.")
            model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    # If finetuning, leave requires_grad=True (default); if not, freeze backbone
    if not finetune:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return model


# ---------- Eval ----------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True).long()
        logits = model(imgs)
        loss = loss_fn(logits, targets)
        loss_sum += loss.item() * imgs.size(0)
        pred = logits.argmax(1)
        correct += (pred == targets).sum().item()
        total += imgs.size(0)
    return (loss_sum / max(total, 1)), (correct / max(total, 1))


# ---------- Train ----------
def train(args):
    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = make_datasets(args.data, args.img_size, args.val_split, args.seed)

    # DataLoaders
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0)
    )

    # Peek at labels (should be [0,1])
    for name, ld in [("train", train_loader), ("val", val_loader)]:
        _, t = next(iter(ld))
        print(f"[label-check:{name}] unique targets -> {sorted(set(int(i) for i in t))}")

    model = make_model(num_classes=2, pretrained=args.pretrained, finetune=not args.freeze).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best_resnet18.pt"

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (imgs, targets) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                logits = model(imgs)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
            else:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(imgs)
                    loss = loss_fn(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | step {step}/{len(train_loader)} | loss {running/50:.4f}")
                running = 0.0

        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"[val] epoch {epoch}: loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc}, best_path)
            print(f"  ✅ saved best to {best_path} (acc={val_acc:.4f})")

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")


# ---------- Predict ----------
@torch.no_grad()
def predict(image_path: str, ckpt_path: str, img_size: int = 224):
    device = get_device()
    model = make_model(num_classes=2, pretrained="none", finetune=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    from PIL import Image
    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    logits = model(img)
    prob = torch.softmax(logits, dim=1).squeeze().cpu()
    idx = int(prob.argmax().item())
    label = "dog" if idx == 1 else "cat"
    print(f"Prediction: {label} (cat={prob[0]:.3f}, dog={prob[1]:.3f})")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data", help="Root folder for Oxford-IIIT Pet")
    p.add_argument("--out", type=str, default="./runs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", type=str, default="none", choices=["none","imagenet"])
    p.add_argument("--freeze", action="store_true", help="freeze backbone (linear probe)")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--predict", type=str, default=None, help="path to image to classify")
    p.add_argument("--ckpt", type=str, default="./runs/best_resnet18.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict(args.predict, args.ckpt, img_size=args.img_size)
    else:
        train(args)
