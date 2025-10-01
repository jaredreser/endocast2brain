import numpy as np
import matplotlib.pyplot as plt

def save_three_plane_png(endo, pred, gt, path):
    """
    endo, pred, gt: 3D arrays [D,H,W] in [0,1].
    Saves coronal (y), sagittal (x), axial (z) mid-slices in a 3x3 grid:
    row1=endocast, row2=prediction, row3=ground truth.
    """
    D,H,W = endo.shape
    zc, yc, xc = D//2, H//2, W//2

    def row(img):
        return [img[zc,:,:], img[:,yc,:], img[:,:,xc]]

    rows = [row(endo), row(pred), row(gt)]
    titles = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]

    plt.figure(figsize=(9,9))
    for r in range(3):
        for c in range(3):
            ax = plt.subplot(3,3,r*3+c+1)
            ax.imshow(rows[r][c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
