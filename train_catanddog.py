import argparse, os, time, math, random, itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

# Handle torchvision version differences for weights API
try:
    from torchvision.models import resnet18, ResNet18_Weights
    _HAS_WEIGHTS = True
except Exception:
    from torchvision.models import resnet18
    ResNet18_Weights = None
    _HAS_WEIGHTS = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_datasets(root, img_size=224, val_split=0.1, seed=42):
    # Train/val split from the official "trainval" split; separate "test" is used for final evaluation
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),  # ImageNet stats
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    trainval = OxfordIIITPet(root=root, split="trainval",
                             target_types="category",
                             download=True, transform=train_tf)
    test = OxfordIIITPet(root=root, split="test",
                         target_types="category",
                         download=True, transform=eval_tf)

    # --- NEW: map 37-breed labels -> species (0=cat, 1=dog) ---
    # Cat breed names exactly as used by torchvision's OxfordIIITPet.classes
    cat_breeds = {
        "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
        "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll",
        "Russian_Blue", "Siamese", "Sphynx"
    }
    name_to_idx = {name: i for i, name in enumerate(trainval.classes)}
    cat_idx = {name_to_idx[name] for name in cat_breeds if name in name_to_idx}

    def to_species(label: int) -> int:
        # label is breed index 0..36
        return 0 if label in cat_idx else 1  # 0=cat, 1=dog

    trainval.target_transform = to_species
    test.target_transform = to_species
    # --- end NEW ---

    # Simple random split for validation (stratified split would be nicer, but this is fine to start)
    val_len = int(len(trainval) * val_split)
    train_len = len(trainval) - val_len
    train_ds, val_ds = random_split(
        trainval, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )
    # Use eval transforms for val loader
    val_ds.dataset.transform = eval_tf
    return train_ds, val_ds, test

def make_model(num_classes=2, finetune=True):
    if _HAS_WEIGHTS and ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(pretrained=True)
    # Replace the final head
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    if finetune:
        # Fine-tune all layers (set False to freeze backbone)
        for p in model.parameters():
            p.requires_grad = True
    return model

@torch.no_grad()
def evaluate(model, loader, device):
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
    num_workers = max(2, os.cpu_count() // 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_mem)

    model = make_model(num_classes=2, finetune=not args.freeze)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_path = 0.0, Path(args.out) / "best_resnet18.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
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

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

def predict(image_path, ckpt_path, img_size=224):
    device = get_device()
    model = make_model(num_classes=2, finetune=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
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
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze", action="store_true", help="freeze backbone (linear probe)")
    p.add_argument("--predict", type=str, default=None, help="image path to run a single prediction")
    p.add_argument("--ckpt", type=str, default="./runs/best_resnet18.pt", help="checkpoint for --predict")
    args = p.parse_args()

    if args.predict:
        predict(args.predict, args.ckpt, img_size=args.img_size)
    else:
        train(args)
