#!/usr/bin/env python3
"""
Flexible Graphene Raman Analysis -- looks for graphene-like peaks 
with more flexible ranges and manual override option
"""
DATA_FILE   = "sample_data/raman/raman_sample.txt"
OUT_PNG     = "images/graphene_fit_test.png"
OUT_JSON    = "features/raman_sample_features.json"

# Set to True to save ALL peaks, not just graphene ones
SAVE_ALL_PEAKS = True

# Flexible graphene peak ranges (can be adjusted)
D_RANGE = (1200, 1450)   # D peak
G_RANGE = (1500, 1650)   # G peak  
TwoD_RANGE = (2500, 2900) # 2D peak
# ---------------------------------------------------------------------

from pathlib import Path, PurePath
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from lmfit.models import VoigtModel


def load_txt(path):
    enc = ["utf‑8","latin1","cp1252","iso‑8859‑1"]
    for e in enc:
        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                             names=["nu","I"], encoding=e, engine="python")
            return df["nu"].values, df["I"].values
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path}")

def baseline_poly5(nu, I):
    return I - np.polyval(np.polyfit(nu, I, 5), nu)

def smooth(I, window=11, poly=3):
    window |= 1
    return savgol_filter(I, window_length=window, polyorder=poly)

def seed_peaks(I, prom=0.02):
    p,_ = find_peaks(I, prominence=prom*I.max())
    return p

def make_voigt(): 
    return VoigtModel(prefix="p_")

def slice_region(nu,I,idx,w=120):
    c = nu[idx]
    m = (nu>c-w)&(nu<c+w)
    return nu[m],I[m]

def fit_region(nu_roi,I_roi):
    m = make_voigt()
    pars = m.guess(I_roi,x=nu_roi)
    return m.fit(I_roi,pars,x=nu_roi)

def voigt_fwhm(sigma,gamma):
    return 0.5346*(2*gamma)+np.sqrt(0.2166*(2*gamma)**2+(2.355*sigma)**2)

def classify_peak(center_cm):
    """Classify peaks with flexible ranges"""
    if D_RANGE[0] < center_cm < D_RANGE[1]: 
        return "D"
    if G_RANGE[0] < center_cm < G_RANGE[1]: 
        return "G"
    if TwoD_RANGE[0] < center_cm < TwoD_RANGE[1]: 
        return "2D"
    return None

def main():
    nu,I_raw = load_txt(DATA_FILE)
    I_bl  = baseline_poly5(nu,I_raw)
    I_sm  = smooth(I_bl)
    idxs  = seed_peaks(I_sm)

    print("Seeded peaks:", [f"{nu[i]:.1f} cm⁻¹" for i in idxs])
    print(f"Looking for graphene peaks in ranges:")
    print(f"  D: {D_RANGE[0]}-{D_RANGE[1]} cm⁻¹")
    print(f"  G: {G_RANGE[0]}-{G_RANGE[1]} cm⁻¹") 
    print(f"  2D: {TwoD_RANGE[0]}-{TwoD_RANGE[1]} cm⁻¹")
    print(f"Save all peaks: {SAVE_ALL_PEAKS}")
    
    feats = {}
    Path("images").mkdir(exist_ok=True)

    graphene_peaks_found = 0
    all_peaks_data = []
    
    for i, idx in enumerate(idxs):
        nu_roi,I_roi = slice_region(nu,I_sm,idx)
        out = fit_region(nu_roi,I_roi)
        
        if not out.success: 
            print(f"Peak {i+1} at {nu[idx]:.1f} cm⁻¹: fit failed")
            continue

        cen  = out.params["p_center"].value
        sig  = out.params["p_sigma"].value
        gam  = out.params["p_gamma"].value
        amp  = out.params["p_amplitude"].value
        fwhm = voigt_fwhm(sig,gam)

        label = classify_peak(cen)
        
        print(f"Peak {i+1}: {cen:.1f} cm⁻¹, FWHM={fwhm:.1f}, classified as: {label or 'other'}")
        
        # Store peak data
        peak_data = {
            "center_cm": round(cen, 2),
            "fwhm_cm": round(fwhm, 2), 
            "amplitude": round(amp, 2),
            "classification": label or "other"
        }
        all_peaks_data.append(peak_data)
        
        # Add graphene-specific features
        if label:
            feats[f"nu_{label}"] = round(cen,2)
            feats[f"fwhm_{label}"] = round(fwhm,2)
            feats[f"amp_{label}"] = round(amp,2)
            graphene_peaks_found += 1
        
        # Add generic peak features if requested
        if SAVE_ALL_PEAKS:
            feats[f"peak_{i+1}_center"] = round(cen, 2)
            feats[f"peak_{i+1}_fwhm"] = round(fwhm, 2)
            feats[f"peak_{i+1}_amplitude"] = round(amp, 2)
            if label:
                feats[f"peak_{i+1}_type"] = label

        # ROI plot
        plt.figure(figsize=(4,3))
        plt.plot(nu_roi,I_roi,lw=1,label="data")
        plt.plot(nu_roi,out.best_fit,lw=1.2,label="Voigt fit")
        title = f"Peak {i+1}: {cen:.1f} cm⁻¹"
        if label:
            title += f" ({label})"
        plt.title(title)
        plt.xlabel("Raman shift (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"images/peak_{i+1}_{int(cen)}.png", dpi=300)
        plt.close()

    # Store summary info
    feats["all_peaks"] = all_peaks_data
    feats["num_total_peaks"] = len(all_peaks_data)
    feats["num_graphene_peaks"] = graphene_peaks_found
    
    # Compute graphene ratios if peaks present
    if "amp_D" in feats and "amp_G" in feats:
        feats["I_D_over_I_G"] = round(feats["amp_D"]/feats["amp_G"],3)
        print(f"I(D)/I(G) = {feats['I_D_over_I_G']}")
    if "amp_2D" in feats and "amp_G" in feats:
        feats["I_2D_over_I_G"] = round(feats["amp_2D"]/feats["amp_G"],3)
        print(f"I(2D)/I(G) = {feats['I_2D_over_I_G']}")

    feats["source_file"] = PurePath(DATA_FILE).name

    # Save JSON
    Path(OUT_JSON).parent.mkdir(parents=True,exist_ok=True)
    with open(OUT_JSON,"w") as f: 
        json.dump(feats,f,indent=2)
    print(f"\nSaved features → {OUT_JSON}")

    # Overview plot  
    plt.figure(figsize=(10,6))
    plt.plot(nu,I_raw,label="raw",alpha=.4)
    plt.plot(nu,I_bl,label="baseline‑removed",alpha=.6)
    plt.plot(nu,I_sm,label="smoothed",lw=1.2)
    
    # Color-code peaks by type
    colors = {"D": "red", "G": "green", "2D": "blue", "other": "orange"}
    for i, (idx, peak_data) in enumerate(zip(idxs, all_peaks_data)):
        color = colors[peak_data["classification"]]
        plt.scatter(nu[idx], I_sm[idx], c=color, marker="x", s=50)
        label = peak_data["classification"]
        if label != "other":
            label = label.upper()
        plt.annotate(f"{label}\n{nu[idx]:.0f}", 
                    xy=(nu[idx], I_sm[idx]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.xlabel("Raman shift (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Raman Spectrum Analysis")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(OUT_PNG).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(OUT_PNG,dpi=300)
    plt.close()
    print(f"Saved plot     → {OUT_PNG}")

    print(f"\n=== Summary ===")
    print(f"Total peaks found: {len(all_peaks_data)}")
    print(f"Graphene peaks: {graphene_peaks_found}")
    print(f"Features saved: {len(feats)}")
    
    if graphene_peaks_found == 0:
        print("\n⚠️  No graphene peaks detected in expected ranges!")
        print("Your data may be from a different material, or you may need to:")
        print("- Check if you have the right data file")
        print("- Adjust the peak ranges at the top of the script")
        print("- Set SAVE_ALL_PEAKS = True to analyze whatever material this is")

if __name__ == "__main__":
    main()