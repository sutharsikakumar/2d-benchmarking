"""Plot the raw raman spectra."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/graphene_1.txt")
OUTPUT_DIR = Path("results/graphene")
OUTPUT_FILE = OUTPUT_DIR / "raw_spectrum.png"

def main() -> None:
    spectrum = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=["Wavenumber", "Intensity"],
        decimal=",",
    ).query("1000 <= Wavenumber <= 3000")

    fig, ax = plt.subplots()
    ax.plot(spectrum["Wavenumber"], spectrum["Intensity"])
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Raw Raman Spectrum (Graphene)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
