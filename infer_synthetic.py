import os, argparse, torch, numpy as np
from src.endocast2brain.data.synthetic import SyntheticEndocastBrainSet
from src.endocast2brain.models.unet3d import TinyUNet3D
from src.endocast2brain.utils.vis import save_three_plane_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="runs/infer")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = TinyUNet3D(in_channels=1, out_channels=1, base=8).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    ds = SyntheticEndocastBrainSet(n=args.count, size=args.size, seed=1337)
    for i in range(args.count):
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)  # [1,1,D,H,W]
        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0,0].cpu().numpy()
        endo = x[0,0].cpu().numpy()
        gt   = y[0,0].cpu().numpy()
        save_three_plane_png(endo, pred, gt, os.path.join(args.out_dir, f"sample_{i:02d}.png"))

    print(f"Saved {args.count} prediction triptychs to {args.out_dir}")

if __name__ == "__main__":
    main()
