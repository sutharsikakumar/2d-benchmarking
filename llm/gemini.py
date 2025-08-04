import os
import json
import pathlib
import textwrap
from typing import List, Dict
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
PEAK_FILE = ROOT / "results/graphene/graphene_1_peaks.json"
OUT_FILE = ROOT / "results/peaks_analysis.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
GEMINI_MODEL = "gemini-1.5-pro-latest"
TEMPERATURE = 0.4

SYSTEM_PROMPT = (
    "You are an expert Raman spectroscopy analyst. Using your domain knowledge "
    "and comparing to verified literature, suggest three possible band/element "
    "assignments for a given Raman peak (rank 1-3) with concise rationales, and "
    "select one as the final answer. Respond ONLY with valid JSON."
)

def build_prompt(peak: dict) -> str:
    return textwrap.dedent(
        f"""
        Raman peak assignment
        Position: {peak['position']:.2f} cm-1
        Intensity: {peak['intensity']:.1f}
        FWHM: {peak['whm']:.2f} cm-1

        Respond with JSON:
        {{
          "candidates": [
            {{"name": "...", "rank": 1, "rationale": "..."}},
            {{"name": "...", "rank": 2, "rationale": "..."}},
            {{"name": "...", "rank": 3, "rationale": "..."}}
          ],
          "final": "..."
        }}
        """
    )

def ask_gemini(prompt: str) -> Dict:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": TEMPERATURE},
    )
    
    text = resp.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Gemini returned invalid JSON:\n{resp.text}") from err

def analyze(peaks: List[dict]) -> List[dict]:
    analyses = []
    for peak in peaks:
        analysis = ask_gemini(build_prompt(peak))
        analysis["position"] = peak["position"]
        analyses.append(analysis)
    return analyses

def main() -> None:
    if not PEAK_FILE.exists():
        raise FileNotFoundError(f"Peak file not found: {PEAK_FILE}")
    
    peaks = json.loads(PEAK_FILE.read_text())
    results = {"peaks_analysis": analyze(peaks)}
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()