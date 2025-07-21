#!/usr/bin/env python3
"""
Send one Raman‑feature JSON file to Gemini‑1.5‑Pro and print its analysis.
"""

import json, os, google.generativeai as genai, sys, textwrap

FEATURE_FILE = sys.argv[1] if len(sys.argv)==2 else "features/raman_sample_features.json"

# 1.  Load the JSON vector
with open(FEATURE_FILE) as f:
    feats = json.load(f)

# 2.  Build a small, structured prompt
prompt = textwrap.dedent(f"""
    You are an expert in graphene Raman spectroscopy.

    TASK:
      • Estimate the layer number (mono / bi / multi).
      • Evaluate defect density qualitatively (low / medium / high).
      • Summarise your reasoning in < 120 words.

    Raman peak summary (JSON):
    {json.dumps(feats, indent=2)}
""")

# 3.  Call Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")
resp  = model.generate_content(prompt)

print("\n--- Gemini response ---\n")
print(resp.text)
