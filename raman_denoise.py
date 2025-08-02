"""Apply Savitzky-Golay filter to denoise Raman spectra. 
3rd order polynomial with a window size of 11.

Create a png image with denoised spectra, denoised spectra over raw spectra,
and a zoomed in view of the denoised spectra at one wavenumber range.

Save the denoised spectra to a csv."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

DATA_PATH  = Path("data/graphene_1.txt")
OUTPUT_DIR = Path("results/graphene")
PNG_FILE   = OUTPUT_DIR / "denoised_spectra.png"
CSV_FILE   = OUTPUT_DIR / "graphene_1_denoised.csv"
ZOOM_RANGE = (2250, 2500)

def main() -> None:
    df = (
        pd.read_csv(
            DATA_PATH,
            sep=r"\s+",
            header=None,
            names=["Wavenumber", "Intensity"],
            decimal=",",
        )
        .query("1000 <= Wavenumber <= 3000")
        .sort_values("Wavenumber")
    )

    df["Denoised"] = savgol_filter(df["Intensity"].to_numpy(), 21, 3, mode="nearest")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df[["Wavenumber", "Denoised"]].to_csv(CSV_FILE, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    axes[0].plot(df["Wavenumber"], df["Denoised"], color="green")
    axes[0].set_title("Denoised Raman Spectrum")
    axes[0].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0].set_ylabel("Intensity (a.u.)")

    axes[1].plot(df["Wavenumber"], df["Intensity"], label="Raw", alpha=0.4, linewidth=5)
    axes[1].plot(df["Wavenumber"], df["Denoised"], label="Denoised", color="green", linewidth=1.5)
    axes[1].set_title("Raw vs. Denoised")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].set_ylabel("Intensity (a.u.)")
    axes[1].legend()

    zoom = df.query(f"{ZOOM_RANGE[0]} <= Wavenumber <= {ZOOM_RANGE[1]}")
    axes[2].plot(zoom["Wavenumber"], zoom["Intensity"], alpha=0.4, label="Raw", linewidth=5)
    axes[2].plot(zoom["Wavenumber"], zoom["Denoised"], color="green", label="Denoised", linewidth=1.5)
    axes[2].set_title(f"Zoom {ZOOM_RANGE[0]}–{ZOOM_RANGE[1]} cm⁻¹")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].set_ylabel("Intensity (a.u.)")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
