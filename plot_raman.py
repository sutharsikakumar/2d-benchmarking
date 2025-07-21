from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

FILE_PATH = "sample_data/raman/Gr_YC_2.txt"
OUT_DIR   = "images"

def load_raman(path: Path) -> pd.DataFrame:
    """Return DataFrame [Wavenumber, Intensity], auto‑detecting encoding."""
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=r"\s+",
                comment="#",
                header=None,
                names=["Wavenumber", "Intensity"],
                engine="python",
                encoding=enc,
            )
        except UnicodeDecodeError as e:
            last_err = e
    raise UnicodeDecodeError(
        f"Couldn’t decode {path}. Tried {', '.join(encodings)}.\nLast error: {last_err}"
    )


def plot_and_save(df: pd.DataFrame, outfile: Path, title: str | None = None) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(df["Wavenumber"], df["Intensity"], lw=1)
    plt.gca().invert_xaxis()
    plt.xlabel("Raman shift (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved plot → {outfile}")


def main() -> None:
    path = Path(FILE_PATH)
    if not path.is_file():
        sys.exit(f"File not found: {path}. Edit FILE_PATH at the top of the script.")
    df = load_raman(path)
    outfile = Path(OUT_DIR) / f"{path.stem}.png"
    plot_and_save(df, outfile, title=path.stem)


if __name__ == "__main__":
    main()
