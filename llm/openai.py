#!/usr/bin/env python3
"""
Analyze Raman peaks with GPT-4o.

Reads a JSON list of detected peaks, queries the OpenAI API for likely band /
element assignments, and writes an aggregated analysis file.

Requirements
------------
pip install --upgrade "openai>=1.0" python-dotenv
Put your key in a `.env` (OPENAI_API_KEY=sk-...) or export it in the shell.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap
import time
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError

# ──────────────────────────── configuration ──────────────────────────── #

load_dotenv()                                   # read .env if present

ROOT = pathlib.Path(__file__).resolve().parents[1]

PEAK_FILE = ROOT / "results" / "graphene" / "graphene_1_peaks.json"
OUT_FILE  = ROOT / "results" / "peaks_analysis_gpt4o.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is required")

OPENAI_MODEL = "gpt-4o"
TEMPERATURE  = 0.4
MAX_RETRIES  = 3                                # retry on transient API errors

SYSTEM_PROMPT = (
    "You are an expert Raman spectroscopy analyst.  Using domain knowledge and "
    "peer-reviewed literature, suggest three possible band/element assignments "
    "for the peak (ranked 1–3) with concise rationales, then choose one as the "
    "final answer. Respond ONLY with valid JSON."
)

# ────────────────────────────── helpers ──────────────────────────────── #

def build_prompt(peak: dict) -> str:
    """Format the user prompt for a single Raman peak."""
    return textwrap.dedent(
        f"""
        Raman peak assignment request
        -----------------------------
        Position : {peak['position']:.2f} cm⁻¹
        Intensity: {peak['intensity']:.1f} a.u.
        FWHM     : {peak['whm']:.2f} cm⁻¹

        Respond with JSON of the form:
        {{
          "candidates": [
            {{ "name": "...", "rank": 1, "rationale": "..." }},
            {{ "name": "...", "rank": 2, "rationale": "..." }},
            {{ "name": "...", "rank": 3, "rationale": "..." }}
          ],
          "final": "..."
        }}
        """
    ).strip()


def ask_openai(prompt: str, client: OpenAI) -> Dict:
    """Send a prompt, return parsed JSON (with retry + validation)."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            return json.loads(content)

        except (APIError, RateLimitError) as err:
            last_error = err
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)   # exponential back-off
                continue
        except json.JSONDecodeError as err:
            raise ValueError(f"OpenAI returned invalid JSON:\n{content}") from err

    # if we drop out of the loop
    raise RuntimeError(f"OpenAI request failed after {MAX_RETRIES} attempts") from last_error


def analyze(peaks: List[dict]) -> List[dict]:
    """Run the analysis for every peak."""
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=30)
    analyses: List[dict] = []

    for peak in peaks:
        result = ask_openai(build_prompt(peak), client)
        # add original peak meta for convenience
        result.update(
            position=peak["position"],
            intensity=peak["intensity"],
            whm=peak["whm"],
        )
        analyses.append(result)

    return analyses

# ────────────────────────────── main ─────────────────────────────────── #

def main() -> None:
    if not PEAK_FILE.exists():
        raise FileNotFoundError(f"Peak file not found: {PEAK_FILE}")

    peaks: List[dict] = json.loads(PEAK_FILE.read_text())
    results = {"peaks_analysis": analyze(peaks)}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"✓ Analysis written to {OUT_FILE.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
