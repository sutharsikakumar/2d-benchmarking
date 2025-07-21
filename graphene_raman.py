"""baseline‑correct, smooth, seed peaks, Voigt‑fit each peak region separately, and report centre + FWHM."""
DATA_FILE  = "sample_data/raman/Gr_YC_2.txt"
OUTPUT_PNG = "images/graphene_fit_GRYC2.png" 

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from lmfit.models import VoigtModel


def load_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    encodings = ["utf‑8", "latin1", "cp1252", "iso‑8859‑1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                             names=["nu", "I"], encoding=enc, engine="python")
            return df["nu"].values, df["I"].values
        except UnicodeDecodeError as e:
            last_err = e
    raise UnicodeDecodeError(
        f"Couldn’t decode {path}. Tried {', '.join(encodings)}.\nLast error: {last_err}"
    )


def baseline_poly5(nu: np.ndarray, I: np.ndarray) -> np.ndarray:
    return I - np.polyval(np.polyfit(nu, I, deg=5), nu)


def smooth(I: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    window |= 1        # ensure odd
    return savgol_filter(I, window_length=window, polyorder=poly)


def seed_peaks(I: np.ndarray, prominence_frac: float = 0.02) -> np.ndarray:
    peaks, _ = find_peaks(I, prominence=prominence_frac * I.max())
    return peaks


def make_voigt() -> VoigtModel:
    return VoigtModel(prefix="p_")


def slice_region(nu, I, center_idx, half_range_cm=120):
    cν = nu[center_idx]
    mask = (nu > cν - half_range_cm) & (nu < cν + half_range_cm)
    return nu[mask], I[mask]


def fit_region(nu_roi, I_roi):
    model = make_voigt()
    pars  = model.guess(I_roi, x=nu_roi)
    return model.fit(I_roi, pars, x=nu_roi)


def voigt_fwhm(sigma, gamma):
    return 0.5346 * (2 * gamma) + np.sqrt(0.2166 * (2 * gamma) ** 2 +
                                          (2.355 * sigma) ** 2)


def main():

    nu, I_raw = load_txt(DATA_FILE)
    I_bl = baseline_poly5(nu, I_raw)
    I_sm = smooth(I_bl)

    peak_idx = seed_peaks(I_sm)
    print("Seeded peaks at:", [f"{nu[i]:.1f} cm⁻¹" for i in peak_idx])

    results = []
    for idx in peak_idx:
        nu_roi, I_roi = slice_region(nu, I_sm, idx)
        out = fit_region(nu_roi, I_roi)

        if out.success:
            cen   = out.params["p_center"].value
            sigma = out.params["p_sigma"].value
            gamma = out.params["p_gamma"].value
            fwhm  = voigt_fwhm(sigma, gamma)

            results.append(dict(Seed=f"{nu[idx]:.1f}",
                                Center=cen,
                                FWHM=fwhm))

        plt.figure(figsize=(4, 3))
        plt.plot(nu_roi, I_roi,        label="data", lw=1)
        plt.plot(nu_roi, out.best_fit, label="Voigt fit", lw=1.2)
        plt.gca().invert_xaxis(); plt.legend(); plt.tight_layout()
        Path("images").mkdir(exist_ok=True)
        plt.savefig(f"images/fit_roi_{int(nu[idx])}.png", dpi=300)
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(nu, I_raw, label="raw", alpha=0.4)
    plt.plot(nu, I_bl,  label="baseline‑removed", alpha=0.6)
    plt.plot(nu, I_sm,  label="smoothed", lw=1.2)
    plt.scatter(nu[peak_idx], I_sm[peak_idx], marker="x", c="red", label="seeds")
    plt.xlabel("Raman shift (cm⁻¹)"); plt.ylabel("Intensity (a.u.)")
    plt.gca().invert_xaxis(); plt.legend(); plt.tight_layout()
    Path(OUTPUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300); plt.close()


    if results:
        df = pd.DataFrame(results)
        print("\n=== Peak fit summary ===")
        print(df.to_string(index=False, float_format="%.3f"))
    else:
        print("No peaks successfully fitted.")


if __name__ == "__main__":
    main()
