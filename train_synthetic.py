import os, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.endocast2brain.data.synthetic import SyntheticEndocastBrainSet
from src.endocast2brain.models.unet3d import TinyUNet3D

def dice_coeff(pred, target, eps=1e-6):
    # pred, target: [B,1,D,H,W] in {0,1} or probabilities
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3,4))
    denom = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4)) + eps
    return (2 * inter / denom).mean().item()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--size", type=int, default=64)   # 64^3 fits 8GB VRAM (batch size 1–2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", type=str, default="runs/exp1")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    ds = SyntheticEndocastBrainSet(n=args.samples, size=args.size)
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = TinyUNet3D(in_channels=1, out_channels=1, base=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        # Validation
        model.eval()
        vdice, vloss = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss += loss_fn(logits, y).item()
                vdice += dice_coeff(torch.sigmoid(logits), y)
        vloss /= len(val_loader); vdice /= len(val_loader)

        print(f"Train loss {train_loss:.4f} | Val loss {vloss:.4f} | Val Dice {vdice:.3f}")

        # Save best checkpoint
        ckpt_path = os.path.join(args.save_dir, "best.pt")
        if vdice > best_dice:
            best_dice = vdice
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"  Saved best to {ckpt_path}")

if __name__ == "__main__":
    main()
