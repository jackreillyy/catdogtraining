#!/usr/bin/env python3
import argparse, os, sys, time
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="./runs")
    args = parser.parse_args()

    print("=== Environment ===")
    print("Python:", sys.version)
    try:
        import torchvision
        print("torch:", torch.__version__)
        print("torchvision:", torchvision.__version__)
    except Exception as e:
        print("torch/torchvision import error:", repr(e))
        raise

    # Import your training module
    try:
        import train_catanddog as tcd
    except Exception as e:
        print("Failed to import train_catanddog.py:", repr(e))
        print("Make sure this script sits next to train_catanddog.py or PYTHONPATH includes your repo.")
        raise

    device = tcd.get_device()
    print(f"Device selected: {device} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    # Build datasets/loaders
    print("\n=== Building datasets/loaders ===")
    train_ds, val_ds, test_ds = tcd.make_datasets(
        root=args.data, img_size=args.img_size, val_split=0.1, seed=args.seed
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=max(2, (os.cpu_count() or 4)//2), pin_memory=(device.type=="cuda")
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(2, (os.cpu_count() or 4)//2), pin_memory=(device.type=="cuda")
    )

    # Sanity-check labels
    print("\n=== Label sanity check ===")
    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))
    u_train = torch.unique(y_train)
    u_val = torch.unique(y_val)
    print("train unique targets:", u_train.tolist())
    print("val   unique targets:", u_val.tolist())
    bad_train = torch.any((y_train < 0) | (y_train > 1))
    bad_val   = torch.any((y_val   < 0) | (y_val   > 1))
    if bad_train or bad_val:
        print("ERROR: Found targets outside {0,1}.")
        print("Offending train uniques:", u_train.tolist(), "val uniques:", u_val.tolist())
        raise SystemExit(1)
    else:
        print("OK: Targets are in {0,1}.")

    # Build model and do a single train step
    print("\n=== One-step training smoke test ===")
    model = tcd.make_model(num_classes=2, finetune=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x_train = x_train.to(device)
    y_train = y_train.to(device).long()
    logits = model(x_train)
    print("logits shape:", tuple(logits.shape))
    assert logits.shape[1] == 2, "Model must output 2 logits for binary classification."

    # Hard assert before loss (extra safety)
    if torch.any((y_train < 0) | (y_train > 1)):
        print("ERROR: Bad targets in batch:", torch.unique(y_train).tolist())
        raise SystemExit(1)

    loss = criterion(logits, y_train)
    print("loss:", float(loss.item()))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print("Backprop/optimizer step OK.")

    # Save a tiny smoke-test checkpoint
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "_smoketest.pt")
    torch.save({"model": model.state_dict(), "step": 1, "loss": float(loss.item())}, ckpt_path)
    print(f"Saved smoke-test checkpoint to: {ckpt_path}")

    print("\n✅ Smoke test completed successfully.")

if __name__ == "__main__":
    # Optional: clearer CUDA error location if something goes wrong
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    main()
