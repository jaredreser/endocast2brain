import numpy as np, random, torch
from torch.utils.data import Dataset

def _rand_ellipsoid_mask(size, radii_range=(0.25, 0.45), center_jitter=0.05):
    D = H = W = size
    zc = 0.5 + (random.random() - 0.5) * 2 * center_jitter
    yc = 0.5 + (random.random() - 0.5) * 2 * center_jitter
    xc = 0.5 + (random.random() - 0.5) * 2 * center_jitter
    rz = random.uniform(*radii_range)
    ry = random.uniform(*radii_range)
    rx = random.uniform(*radii_range)

    zz, yy, xx = np.meshgrid(np.linspace(0,1,D), np.linspace(0,1,H), np.linspace(0,1,W), indexing="ij")
    ellip = ((zz-zc)**2/rz**2 + (yy-yc)**2/ry**2 + (xx-xc)**2/rx**2) <= 1.0
    return ellip.astype(np.float32)

def _gyri_like_warp(vol, amplitude=0.03, freq=10):
    """Add a sinusoidal displacement to simulate cortical folds."""
    D,H,W = vol.shape
    zz, yy, xx = np.meshgrid(np.linspace(0,1,D), np.linspace(0,1,H), np.linspace(0,1,W), indexing="ij")
    disp = amplitude * (np.sin(2*np.pi*freq*xx) * np.sin(2*np.pi*freq*yy))
    warped = np.clip(vol + disp, 0.0, 1.0)
    return warped

def _endocast_from_brain(brain_mask, margin=2):
    """Dilate the brain mask slightly to simulate the skull’s inner surface."""
    import scipy.ndimage as ndi
    skull = ndi.binary_dilation(brain_mask > 0.5, iterations=margin)
    return skull.astype(np.float32)

class SyntheticEndocastBrainSet(Dataset):
    """
    Returns (endocast, brain) as 1xDHW tensors in [0,1].
    Endocast is a dilated version of the brain; brain gets minor warps.
    """
    def __init__(self, n=128, size=64, seed=123):
        self.n = n
        self.size = size
        self.rng = random.Random(seed)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        random.seed(self.rng.random() * 1e9)
        base = _rand_ellipsoid_mask(self.size, radii_range=(0.28, 0.42))
        brain = _gyri_like_warp(base, amplitude=0.05, freq=self.rng.randint(6,12))
        brain = (brain > 0.5).astype(np.float32)
        endo  = _endocast_from_brain(brain, margin=self.rng.randint(1,3)).astype(np.float32)

        # Add light Gaussian noise to endocast
        noise = np.random.normal(0, 0.03, size=endo.shape).astype(np.float32)
        endo = np.clip(endo + noise, 0.0, 1.0)

        x = torch.from_numpy(endo[None])   # [1,D,H,W]
        y = torch.from_numpy(brain[None])  # [1,D,H,W]
        return x, y
