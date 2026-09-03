"""Shared pycortex-style bivariate colour mapping for CIFTI outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_CMAP_NAME = "PU_RdBu_covar_alpha.png"


def find_pycortex_colormap(explicit_path: str | None = None) -> Path:
    """Find the 2-D colormap installed in the active Python environment."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"2-D colormap not found: {path}")
        return path

    candidates = [
        Path(sys.prefix) / "share" / "pycortex" / "colormaps" / DEFAULT_CMAP_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not locate {DEFAULT_CMAP_NAME}; pass --colormap-png explicitly."
    )


def load_rgba_texture(path: Path) -> np.ndarray:
    """Load a pycortex RGB(A) texture as float values in [0, 1]."""
    from matplotlib.image import imread

    rgba = np.asarray(imread(path), dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB(A) colormap image, got {rgba.shape}")
    if rgba.shape[2] == 3:
        rgba = np.concatenate(
            [rgba, np.ones((*rgba.shape[:2], 1), dtype=np.float32)], axis=2
        )
    return np.clip(rgba, 0.0, 1.0)


def quantize_bivariate(
    dim1: np.ndarray,
    dim2: np.ndarray,
    texture: np.ndarray,
    *,
    bins: int = 32,
    vmin: float = 0.0,
    vmax: float = 0.4,
) -> tuple[np.ndarray, dict[int, tuple[str, tuple[float, float, float, float]]]]:
    """Quantize two axes and build the corresponding CIFTI label table.

    The orientation reproduces pycortex ``Dataview2D._to_raw``: ``dim1`` is
    horizontal and ``dim2`` is the inverted vertical texture coordinate.
    """
    dim1 = np.asarray(dim1, dtype=np.float32)
    dim2 = np.asarray(dim2, dtype=np.float32)
    if dim1.shape != dim2.shape:
        raise ValueError(f"Axis shapes differ: {dim1.shape} versus {dim2.shape}")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not vmax > vmin:
        raise ValueError("vmax must be greater than vmin")

    valid = np.isfinite(dim1) & np.isfinite(dim2)
    x = np.clip((np.nan_to_num(dim1, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    y = np.clip((np.nan_to_num(dim2, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    xbin = np.minimum((x * bins).astype(np.int32), bins - 1)
    ybin = np.minimum((y * bins).astype(np.int32), bins - 1)

    keys = (1 + ybin * bins + xbin).astype(np.int32)
    keys[~valid] = 0

    height, width = texture.shape[:2]
    labels: dict[int, tuple[str, tuple[float, float, float, float]]] = {
        0: ("INVALID", (0.0, 0.0, 0.0, 0.0))
    }
    for iy in range(bins):
        for ix in range(bins):
            tx = int(np.round(((ix + 0.5) / bins) * (width - 1)))
            ty = int(np.round((1.0 - (iy + 0.5) / bins) * (height - 1)))
            rgba = tuple(float(value) for value in texture[ty, tx, :4])
            lo1, hi1 = vmin + (vmax - vmin) * np.array([ix, ix + 1]) / bins
            lo2, hi2 = vmin + (vmax - vmin) * np.array([iy, iy + 1]) / bins
            key = 1 + iy * bins + ix
            labels[key] = (
                f"dim1[{lo1:.4f},{hi1:.4f}) dim2[{lo2:.4f},{hi2:.4f})",
                rgba,
            )
    return keys, labels


def save_bivariate_dlabel(
    keys: np.ndarray,
    labels: dict,
    template_cifti: Path,
    output_path: Path,
    map_name: str,
) -> None:
    """Save quantized bivariate colours in a Workbench-compatible dlabel."""
    template = nib.load(str(template_cifti))
    brain_axis = template.header.get_axis(1)
    if keys.size != brain_axis.size:
        raise ValueError(
            f"Map has {keys.size} grayordinates but template has {brain_axis.size}"
        )
    label_axis = nib.cifti2.LabelAxis([map_name], [labels])
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, brain_axis))
    nib.save(nib.Cifti2Image(keys.reshape(1, -1), header=header), str(output_path))


def save_pair_dscalar(
    dim1: np.ndarray,
    dim2: np.ndarray,
    names: tuple[str, str],
    template_cifti: Path,
    output_path: Path,
) -> None:
    """Save the unquantized pair for queries and separate overlays."""
    template = nib.load(str(template_cifti))
    brain_axis = template.header.get_axis(1)
    data = np.vstack([dim1, dim2]).astype(np.float32)
    if data.shape[1] != brain_axis.size:
        raise ValueError(
            f"Maps have {data.shape[1]} grayordinates but template has {brain_axis.size}"
        )
    scalar_axis = nib.cifti2.ScalarAxis(list(names))
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    nib.save(nib.Cifti2Image(data, header=header), str(output_path))


def save_legend(
    texture: np.ndarray,
    output_path: Path,
    roi_a: str,
    roi_b: str,
    vmin: float,
    vmax: float,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str = "Bivariate connective-field colour key",
) -> None:
    """Save the continuous 2-D key needed to interpret the dlabel colours."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(4.2, 3.8), dpi=160)
    axis.imshow(texture, origin="upper", extent=[vmin, vmax, vmin, vmax])
    axis.set_xlabel(xlabel or f"{roi_b} raw split $R^2$ (blue axis)")
    axis.set_ylabel(ylabel or f"{roi_a} raw split $R^2$ (red axis)")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
