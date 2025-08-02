"""Apply Gaussian filter to identify peaks + Width at Half Maximum (WHM) in Raman spectra.
Create a png image with Gaussian smoothed spectra, Gaussian overlayed on raw spectra and SG-filter data.
Identify the peaks in the Gaussian smoothed spectra and calculate, return another json file
with peak positions, intensities, and WHM values."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

RAW_PATH   = Path("data/graphene_1.txt")
SG_PATH    = Path("results/graphene/graphene_1_denoised.csv")
OUT_DIR    = Path("results/graphene")
PNG_FILE   = OUT_DIR / "gaussian_spectra.png"
PEAKS_JSON = OUT_DIR / "graphene_1_peaks.json"

RANGE_EXPR = "1000 <= Wavenumber <= 3000"
NUM_PEAKS  = 4 

def load_raw() -> pd.DataFrame:
    return (
        pd.read_csv(RAW_PATH, sep=r"\s+", header=None,
                    names=["Wavenumber", "Intensity"], decimal=",")
          .query(RANGE_EXPR)
          .sort_values("Wavenumber")
          .reset_index(drop=True)
    )

def load_sg() -> pd.DataFrame:
    return pd.read_csv(SG_PATH).query(RANGE_EXPR).reset_index(drop=True)

def gaussian_smooth(y: np.ndarray, sigma: int = 5) -> np.ndarray:
    return gaussian_filter1d(y, sigma)

def detect_peaks(x: np.ndarray, y: np.ndarray, keep: int) -> tuple[list[int], list[dict]]:
    idx, _ = find_peaks(y, prominence=0.05 * y.max())
    if len(idx) > keep:          
        idx = idx[np.argsort(-y[idx])][:keep]
    widths, _, left, right = peak_widths(y, idx, rel_height=0.5)
    step = np.mean(np.diff(x))
    peaks = [
        {
            "position": float(x[i]),
            "intensity": float(y[i]),
            "whm": float(w * step),
            "left_half": float(x[int(l)]),
            "right_half": float(x[int(r)]),
        }
        for i, w, l, r in zip(idx, widths, left, right)
    ]
    return idx.tolist(), peaks

def save_json(peaks: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PEAKS_JSON, "w") as fh:
        json.dump(peaks, fh, indent=2)

def make_plots(x, y_raw, x_sg, y_sg, y_gauss, peak_idx):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    axes[0].plot(x, y_gauss, color="green")
    axes[0].scatter(x[peak_idx], y_gauss[peak_idx],
                    marker="x", color="red", zorder=5)
    axes[0].set_title("Gaussian-Smoothed Raman Spectrum")
    axes[0].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0].set_ylabel("Intensity (a.u.)")

    # 2 – Overlay
    axes[1].plot(x, y_raw, label="Raw", alpha=0.3)
    axes[1].plot(x_sg, y_sg, label="Savitzky–Golay", color="orange")
    axes[1].plot(x, y_gauss, label="Gaussian", color="green")
    axes[1].scatter(x[peak_idx], y_gauss[peak_idx],
                    marker="x", color="red", zorder=5, label="Peaks")
    axes[1].set_title("Overlay: Raw vs. SG vs. Gaussian")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].set_ylabel("Intensity (a.u.)")
    axes[1].legend()

    zoom_mask_raw = (x >= 1200) & (x <= 1700)
    zoom_mask_sg  = (x_sg >= 1200) & (x_sg <= 1700)
    axes[2].plot(x[zoom_mask_raw], y_raw[zoom_mask_raw], alpha=0.3)
    axes[2].plot(x_sg[zoom_mask_sg], y_sg[zoom_mask_sg], color="orange")
    axes[2].plot(x[zoom_mask_raw], y_gauss[zoom_mask_raw], color="green")
    axes[2].scatter(x[peak_idx], y_gauss[peak_idx],
                    marker="x", color="red", zorder=5)
    axes[2].set_title("Zoom 1200–1700 cm⁻¹")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].set_ylabel("Intensity (a.u.)")

    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight")

def main():
    raw_df = load_raw()
    sg_df  = load_sg()

    x_raw  = raw_df["Wavenumber"].to_numpy()
    y_raw  = raw_df["Intensity"].to_numpy()
    x_sg   = sg_df["Wavenumber"].to_numpy()
    y_sg   = sg_df["Denoised"].to_numpy()
    y_gaus = gaussian_smooth(y_raw, sigma=5)

    idx, peaks = detect_peaks(x_raw, y_gaus, NUM_PEAKS)
    save_json(peaks)
    make_plots(x_raw, y_raw, x_sg, y_sg, y_gaus, idx)

if __name__ == "__main__":
    main()
