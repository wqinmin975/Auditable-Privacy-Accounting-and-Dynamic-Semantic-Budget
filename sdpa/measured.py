"""Measured calibration from CelebAMask-HQ images + part masks.

This module replaces the synthetic anchor with statistics measured on real
faces. Two quantities are *measured* from the image/mask pairs:

    g_h     mean within-part pixel variance (proxy for the released code
            variance of block h; image-space, see caveat below)
    range_h mean 0.5%--99.5% clipped intensity range of the part, used in the
            block L2 sensitivity  Delta f_h = sqrt(d_h) * range_h

The utility weight ``w_h`` and residual fraction ``kappa_h`` cannot be measured
without the trained semantic encoder and a latent noise sweep; they keep the
default surrogate values and are flagged in ``meta``. A latent-level measured
calibration (real decomposed codes) is the remaining step before the numbers
can be quoted as measurements in a paper.

Block -> CelebAMask-HQ mask class mapping (19 classes, 0 = background):

    0  skin      [4]            1  hair       [1]
    2  hat       [2]            3  ears       [3,17,18]
    4  l_brow    [5]            5  r_brow     [6]
    6  l_eye     [7]            7  r_eye      [8]
    8  eye_g     [9]            9  nose       [10]
    10 u_lip     [11]           11 l_lip      [12]
    12 mouth     [13]           13 cloth      [14]
    14 neck      [15,16]
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .calibrate import (
    Calibration,
    N_SHARED,
    D_SHARED,
    D_LOCAL,
    _default_weights,
    _default_kappa,
)

# block index -> mask class ids
BLOCK_CLASSES: list[list[int]] = [
    [4],            # 0  skin
    [1],            # 1  hair
    [2],            # 2  hat
    [3, 17, 18],    # 3  ears
    [5],            # 4  l_brow
    [6],            # 5  r_brow
    [7],            # 6  l_eye
    [8],            # 7  r_eye
    [9],            # 8  eye_g
    [10],           # 9  nose
    [11],           # 10 u_lip
    [12],           # 11 l_lip
    [13],           # 12 mouth
    [14],           # 13 cloth
    [15, 16],       # 14 neck
]

BLOCK_NAMES = [
    "skin", "hair", "hat", "ears", "l_brow", "r_brow", "l_eye", "r_eye",
    "eye_g", "nose", "u_lip", "l_lip", "mouth", "cloth", "neck",
]

# Identity-bearing blocks in this measured layout (face shape + features).
MEASURED_IDENTITY = (0, 1, 4, 5, 6, 7)


def _decode_rows(parquet_path: str | Path, n: int):
    """Yield (image float32 HxWx3 in [0,1], mask int64 HxW) for the first n rows."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(str(parquet_path))
    rows = f.read_row_group(0).slice(0, n).to_pylist()
    for r in rows:
        img = np.asarray(
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        msk = Image.open(io.BytesIO(r["label"]["bytes"])).resize(
            (img.shape[1], img.shape[0]), Image.NEAREST
        )
        yield img, np.asarray(msk, dtype=np.int64)


def measure_part_stats(parquet_path: str | Path, n: int = 500):
    """Per-block measured variance and clipped intensity range.

    Returns
    -------
    g : (H,) float   mean within-part pixel variance, averaged over channels
    rng : (H,) float mean 0.5%--99.5% clipped range
    area : (H,) float mean part pixel fraction
    """
    H = len(BLOCK_CLASSES)
    g_acc = [[] for _ in range(H)]
    rng_acc = [[] for _ in range(H)]
    area_acc = [[] for _ in range(H)]

    for img, msk in _decode_rows(parquet_path, n):
        total = msk.size
        for h, classes in enumerate(BLOCK_CLASSES):
            m = np.isin(msk, classes)
            area_acc[h].append(m.sum() / total)
            if m.sum() > 20:
                px = img[m]
                g_acc[h].append(float(px.var()))
                q = np.quantile(px, [0.005, 0.995])
                rng_acc[h].append(float(q[1] - q[0]))

    g = np.array([np.mean(v) if v else np.nan for v in g_acc])
    rng = np.array([np.mean(v) if v else np.nan for v in rng_acc])
    area = np.array([np.mean(v) for v in area_acc])
    # fallback for classes that never appear in the subset
    g = np.where(np.isnan(g), np.nanmean(g), g)
    rng = np.where(np.isnan(rng), np.nanmean(rng), rng)
    return g, rng, area


def measured_calibration(
    parquet_path: str | Path,
    n: int = 500,
    scenario: str = "task_decoupled",
    identity_idx=MEASURED_IDENTITY,
    identity_weight: float = 4e-3,
    seed: int = 0,
) -> Calibration:
    """Build a Calibration whose g_h and Delta f_h come from real images.

    Parameters
    ----------
    scenario : {"reconstruction", "task_decoupled"}
        reconstruction declares every block private; task_decoupled declares
        only the identity blocks private and down-weights their task weight.
    """
    rng = np.random.default_rng(seed)
    g_part, rng_part, area = measure_part_stats(parquet_path, n)

    d = np.array([D_SHARED] * N_SHARED + [D_LOCAL] * (len(BLOCK_CLASSES) - N_SHARED),
                 dtype=float)
    # Trace proxy g_h = Tr Sigma_h ~ d_h * (per-coordinate pixel variance).
    g = d * g_part
    df = np.sqrt(d) * rng_part

    w = _default_weights(len(BLOCK_CLASSES))
    kappa = _default_kappa(len(BLOCK_CLASSES))
    w = w * (1.0 + 0.08 * rng.standard_normal(len(w))).clip(0.8, 1.2)
    kappa = kappa * (1.0 + 0.05 * rng.standard_normal(len(kappa))).clip(0.85, 1.10)
    kappa = np.clip(kappa, 0.2, 0.98)

    mask = np.ones(len(BLOCK_CLASSES), dtype=bool)
    if scenario == "task_decoupled":
        mask = np.zeros(len(BLOCK_CLASSES), dtype=bool)
        idx = [i for i in identity_idx if i < len(BLOCK_CLASSES)]
        mask[idx] = True
        w[idx] = identity_weight
    elif scenario != "reconstruction":
        raise ValueError("scenario must be reconstruction or task_decoupled")

    wk = w * kappa
    return Calibration(
        g=g, df=df, w=w, kappa=kappa, d=d, private_mask=mask, n_samples=n,
        meta={
            "source": f"celebamaskhq-measured-{scenario}",
            "n_images": n,
            "block_names": BLOCK_NAMES,
            "part_area_fraction": area.round(4).tolist(),
            "measured": ["g", "df"],
            "surrogate": ["w", "kappa"],
            "note": "g_h/Delta f_h measured on real CelebAMask-HQ parts; w_h/kappa_h "
                    "are default surrogates pending the trained semantic encoder.",
        },
    )
