from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

RAW_PATH      = Path("data/graphene_1.txt")
SG_PATH       = Path("results/graphene/graphene_1_denoised.csv")
OUT_DIR       = Path("results/graphene")
PNG_FILE      = OUT_DIR / "gaussian_spectra.png"
PEAKS_JSON    = OUT_DIR / "graphene_1_peaks.json"
RANGE_EXPR    = "1000 <= Wavenumber <= 3000"

def load_raw() -> pd.DataFrame:
    return (
        pd.read_csv(
            RAW_PATH,
            sep=r"\s+",
            header=None,
            names=["Wavenumber", "Intensity"],
            decimal=",",
        )
        .query(RANGE_EXPR)
        .sort_values("Wavenumber")
        .reset_index(drop=True)
    )

def load_sg() -> pd.DataFrame:
    return pd.read_csv(SG_PATH).query(RANGE_EXPR).reset_index(drop=True)

def gaussian_smooth(y: np.ndarray, sigma: int = 5) -> np.ndarray:
    return gaussian_filter1d(y, sigma)

def detect_peaks(x: np.ndarray, y: np.ndarray) -> list[dict]:
    idx, props = find_peaks(y, prominence=0.05 * y.max())
    widths, _, left, right = peak_widths(y, idx, rel_height=0.5)
    step = np.mean(np.diff(x))
    peaks = []
    for i, w, l, r in zip(idx, widths, left, right):
        peaks.append(
            {
                "position": float(x[i]),
                "intensity": float(y[i]),
                "whm": float(w * step),
                "left_half": float(x[int(l)]),
                "right_half": float(x[int(r)]),
            }
        )
    return peaks

def save_json(peaks: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PEAKS_JSON, "w") as fh:
        json.dump(peaks, fh, indent=2)

def make_plots(x, y_raw, y_sg, y_gauss, peaks):
    p_x   = [p["position"]   for p in peaks]
    p_int = [p["intensity"]  for p in peaks]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    axes[0].plot(x, y_gauss, color="green")
    axes[0].scatter(p_x, p_int, marker="x", color="red", zorder=3, label="Peaks")
    axes[0].set_title("Gaussian-Smoothed Raman Spectrum")
    axes[0].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0].set_ylabel("Intensity (a.u.)")
    axes[0].legend()

    axes[1].plot(x, y_raw, label="Raw", alpha=0.3, linewidth="5")
    axes[1].plot(x, y_sg,  label="Savitzky-Golay", color="orange", linewidth="3")
    axes[1].plot(x, y_gauss, label="Gaussian", color="green")
    axes[1].scatter(p_x, p_int, marker="x", color="red", zorder=3)
    axes[1].set_title("Overlay: Raw vs. SG vs. Gaussian")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].set_ylabel("Intensity (a.u.)")
    axes[1].legend()

    zoom_mask = (x >= 1500) & (x <= 1650)
    axes[2].plot(x[zoom_mask], y_raw[zoom_mask], alpha=0.3, linewidth="5")
    axes[2].plot(x[zoom_mask], y_sg[zoom_mask], color="orange", linewidth="3")
    axes[2].plot(x[zoom_mask], y_gauss[zoom_mask], color="green")

    zoom_px   = [px for px in p_x if 1500 <= px <= 1650]
    zoom_pint = [peaks[p_x.index(px)]["intensity"] for px in zoom_px]
    axes[2].scatter(zoom_px, zoom_pint, marker="x", color="red", zorder=4)
    axes[2].set_title("Zoom 1200–1700 cm⁻¹")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].set_ylabel("Intensity (a.u.)")

    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight")

def main():
    raw_df = load_raw()
    sg_df  = load_sg()

    x = raw_df["Wavenumber"].to_numpy()
    y_raw = raw_df["Intensity"].to_numpy()
    y_sg  = sg_df["Denoised"].to_numpy()
    y_gauss = gaussian_smooth(y_raw, sigma=5)

    peaks = detect_peaks(x, y_gauss)
    save_json(peaks)
    make_plots(x, y_raw, y_sg, y_gauss, peaks)

if __name__ == "__main__":
    main()
