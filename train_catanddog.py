# train_catanddog.py
import argparse, os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

try:
    from torchvision.models import resnet18, ResNet18_Weights
    _HAS_WEIGHTS = True
except Exception:
    from torchvision.models import resnet18
    ResNet18_Weights = None
    _HAS_WEIGHTS = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- helpers to make transforms robust to PIL or tensor inputs ---
from PIL import Image
import torchvision.transforms.functional as TF

def _ensure_tensor(img: torch.Tensor | Image.Image) -> torch.Tensor:
    """Return CHW float tensor in [0,1]. If already a tensor, convert dtype/scale if needed."""
    if isinstance(img, torch.Tensor):
        # If uint8 0..255, scale to float 0..1; else leave as-is
        if img.dtype == torch.uint8:
            return img.float() / 255.0
        return img.float()
    elif isinstance(img, Image.Image):
        return TF.to_tensor(img)  # float 0..1
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


class EnsureTensor(transforms.Transform):
    def __call__(self, img):
        return _ensure_tensor(img)


def make_datasets(root: str, img_size: int = 224, val_split: float = 0.1, seed: int = 42
                  ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create train/val/test datasets and GUARANTEE labels are {0=cat,1=dog}.
    All transforms are applied in a wrapper that first ensures tensor type.
    """
    # Compose that works on PIL **or** tensors
    train_tf = transforms.Compose([
        EnsureTensor(),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    eval_tf = transforms.Compose([
        EnsureTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # Base datasets MUST have transform=None so only the wrapper applies transforms
    base_trainval = OxfordIIITPet(root=root, split="trainval",
                                  target_types="category", download=True, transform=None)
    base_test = OxfordIIITPet(root=root, split="test",
                              target_types="category", download=True, transform=None)

    # Build breed->species map (0=cat, 1=dog)
    cat_breeds = {
        "Abyssinian","Bengal","Birman","Bombay","British_Shorthair",
        "Egyptian_Mau","Maine_Coon","Persian","Ragdoll",
        "Russian_Blue","Siamese","Sphynx"
    }
    name_to_idx = {name:i for i,name in enumerate(base_trainval.classes)}
    cat_idx = {name_to_idx[n] for n in cat_breeds if n in name_to_idx}
    map37 = torch.ones(len(base_trainval.classes), dtype=torch.long)
    for i in cat_idx:
        map37[i] = 0  # cat -> 0; default dog -> 1

    # Wrapper that applies transforms and maps labels
    from torch.utils.data import Dataset
    class SpeciesMapDataset(Dataset):
        def __init__(self, base, map37, transform=None):
            self.base = base
            self.map37 = map37
            self.transform = transform

        def __len__(self): return len(self.base)

        def __getitem__(self, idx):
            img, breed_idx = self.base[idx]  # img may be PIL or tensor depending on upstream
            if self.transform is not None:
                img = self.transform(img)
            label = int(self.map37[int(breed_idx)].item())  # -> {0,1}
            return img, label

        @property
        def classes(self): return ["cat","dog"]

    # Wrap trainval with train transforms, then split; swap val to eval transforms
    full_train = SpeciesMapDataset(base_trainval, map37, transform=train_tf)
    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(
        full_train, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )
    val_ds.dataset = SpeciesMapDataset(base_trainval, map37, transform=eval_tf)
    test_ds = SpeciesMapDataset(base_test, map37, transform=eval_tf)
    return train_ds, val_ds, test_ds


def make_model(num_classes: int = 2, finetune: bool = True) -> nn.Module:
    if _HAS_WEIGHTS and ResNet18_Weights is not None:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(pretrained=True)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    if finetune:
        for p in model.parameters():
            p.requires_grad = True
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device).long()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return loss_sum / total, correct / total


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = make_datasets(args.data, args.img_size, args.val_split, args.seed)

    pin_mem = (device.type == "cuda")
    num_workers = max(2, (os.cpu_count() or 4) // 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_mem)

    # sanity: labels must be {0,1}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        _, t = next(iter(loader))
        print(f"[label-check:{name}] unique targets -> {torch.unique(t).tolist()}")

    model = make_model(num_classes=2, finetune=not args.freeze).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_path = 0.0, Path(args.out) / "best_resnet18.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (imgs, targets) in enumerate(train_loader, 1):
            imgs, targets = imgs.to(device), targets.to(device).long()
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | step {step}/{len(train_loader)} | loss {running/50:.4f}")
                running = 0.0

        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"[val] epoch {epoch}: loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "val_acc": best_val_acc,
                        "epoch": epoch}, best_path)
            print(f"  ✅ saved new best to {best_path} (acc={best_val_acc:.4f})")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")


def predict(image_path: str, ckpt_path: str, img_size: int = 224):
    device = get_device()
    model = make_model(num_classes=2, finetune=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    tf = transforms.Compose([
        EnsureTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    from PIL import Image
    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        prob = torch.softmax(logits, dim=1).squeeze().cpu()
        idx = int(torch.argmax(prob).item())
        label = "dog" if idx == 1 else "cat"
        print(f"Prediction: {label}  (cat={prob[0]:.3f}, dog={prob[1]:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out", type=str, default="./runs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze", action="store_true", help="freeze backbone (linear probe)")
    p.add_argument("--predict", type=str, default=None)
    p.add_argument("--ckpt", type=str, default="./runs/best_resnet18.pt")
    args = p.parse_args()

    if args.predict:
        predict(args.predict, args.ckpt, img_size=args.img_size)
    else:
        train(args)

