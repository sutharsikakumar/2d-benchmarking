"""
pairwise benchmarking with elo score calculations
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import textwrap
import itertools
import re
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError


K_FACTOR = 32
MODEL_NAME = "gpt-4o" 
TEMPERATURE = 0.0 
REQUEST_TIMEOUT_S = 60

PROMPT_TMPL = textwrap.dedent("""
You are a domain expert in Raman spectroscopy evaluating two independent
peak-assignment analyses of the *same* graphene spectrum.

##### Candidate A
**summary.json**
```json
{a_summary}
```
**peaks_analysis.json**
```json
{a_analysis}
```
##### Candidate B
**summary.json**
```json
{b_summary}
```
**peaks_analysis.json**
```json
{b_analysis}
```
##### Task
Compare the scientific quality, correctness, clarity, and completeness of the
two candidates. Decide which one is BETTER OVERALL.

Respond strictly with valid JSON:
```json
{{
  "winner": "A" | "B",
  "reasoning": "<concise 2-3 sentence justification>"
}}
```
""").strip()


class EloPlayer:
    def __init__(self, name: str):
        self.name = name
        self.rating = 1000
        self.wins: List[str] = []
        self.losses: List[str] = []
        self.reasons: List[str] = []

    def expected(self, opp: 'EloPlayer') -> float:
        return 1 / (1 + 10 ** ((opp.rating - self.rating) / 400))

    def update(self, opp: 'EloPlayer', score: int) -> None:
        exp = self.expected(opp)
        delta = K_FACTOR * (score - exp)
        self.rating += delta


def call_gpt(prompt: str) -> Dict[str, str]:
    client = OpenAI()
    try:
        print("    Making API call to OpenAI...")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT_S,
            messages=[
                {"role": "system", "content": "You are a careful scientific evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        content = resp.choices[0].message.content
        print(f"    Raw response: {content[:200]}...")
        
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end != -1:
                content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end != -1:
                content = content[json_start:json_end].strip()
        
        print(f"    Parsed JSON content: {content}")
        result = json.loads(content)
        print(f"    Parsed result: {result}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"    JSON decode error: {e}")
        print(f"    Raw content: {content}")
        raise RuntimeError(f"Failed to parse JSON response: {e}")
    except (APIError, RateLimitError) as e:
        print(f"    OpenAI API error: {e}")
        raise RuntimeError(f"OpenAI error: {e}")
    except Exception as e:
        print(f"    Unexpected error: {e}")
        raise RuntimeError(f"Unexpected error: {e}")

def extract_model_name_from_summary(filename: str) -> str:
    """Extract model name from summary file like 'summary_claude_haiku35.json'"""
    match = re.match(r"summary_(.+)\.json", filename)
    if match:
        return match.group(1)
    return None

def extract_model_name_from_peaks(filename: str) -> str:
    """Extract model name from peaks analysis file with various patterns"""
    match = re.match(r"peaks_analysis_(.+)_updated_prompt\.json", filename)
    if match:
        return match.group(1)
    
    match = re.match(r"peaks_analysis_(.+)\.json", filename)
    if match:
        return match.group(1)
    
    return None

def find_peaks_file_for_model(model_name: str, graphene_dir: pathlib.Path) -> pathlib.Path:
    """Find the peaks analysis file for a given model, trying both naming patterns"""
    updated_path = graphene_dir / f"peaks_analysis_{model_name}_updated_prompt.json"
    if updated_path.exists():
        return updated_path
    regular_path = graphene_dir / f"peaks_analysis_{model_name}.json"
    if regular_path.exists():
        return regular_path
    
    return None

def run_benchmark(root: pathlib.Path) -> Dict[str, EloPlayer]:
    graphene_dir = root / "graphene"
    if not graphene_dir.is_dir():
        raise ValueError(f"Expected directory {graphene_dir}, not found")
    summary_files = list(graphene_dir.glob("summary_*.json"))
    peaks_files = list(graphene_dir.glob("peaks_analysis_*.json"))

    print(f"Found {len(summary_files)} summary files and {len(peaks_files)} peaks analysis files")

    summary_models = set()
    for f in summary_files:
        model_name = extract_model_name_from_summary(f.name)
        if model_name:
            summary_models.add(model_name)
            print(f"  Summary: {f.name} -> {model_name}")

    peaks_models = set()
    for f in peaks_files:
        model_name = extract_model_name_from_peaks(f.name)
        if model_name:
            peaks_models.add(model_name)
            print(f"  Peaks: {f.name} -> {model_name}")

    model_names = summary_models.intersection(peaks_models)
    
    print(f"\nDetected {len(model_names)} complete models: {sorted(model_names)}")
    if summary_models != peaks_models:
        print("Warning: Mismatched files detected:")
        if summary_models - peaks_models:
            print(f"  Models with summary but no peaks_analysis: {sorted(summary_models - peaks_models)}")
        if peaks_models - summary_models:
            print(f"  Models with peaks_analysis but no summary: {sorted(peaks_models - summary_models)}")

    if len(model_names) < 2:
        raise ValueError(f"Need at least 2 complete models, found {len(model_names)}")

    players = {name: EloPlayer(name) for name in model_names}

    data = {}
    for name in model_names:
        summary_path = graphene_dir / f"summary_{name}.json"
        peaks_path = find_peaks_file_for_model(name, graphene_dir)
        
        if not (summary_path.exists() and peaks_path and peaks_path.exists()):
            print(f"Warning: Missing files for model {name}")
            print(f"  Summary exists: {summary_path.exists()}")
            print(f"  Peaks path: {peaks_path}")
            print(f"  Peaks exists: {peaks_path.exists() if peaks_path else False}")
            continue
            
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            with open(peaks_path) as f:
                analysis = json.load(f)
            data[name] = {
                "summary": json.dumps(summary, indent=2),
                "analysis": json.dumps(analysis, indent=2)
            }
            print(f"Loaded data for {name}")
        except json.JSONDecodeError as e:
            print(f"Error loading JSON for {name}: {e}")
            continue
        except Exception as e:
            print(f"Error loading files for {name}: {e}")
            continue

    players = {name: players[name] for name in data.keys()}
    
    if len(players) < 2:
        raise ValueError(f"Need at least 2 models with valid data, found {len(players)}")

    print(f"\nRunning comparisons for {len(players)} models: {sorted(players.keys())}")

    comparison_count = 0
    total_comparisons = len(list(itertools.combinations(players.keys(), 2)))
    
    for a, b in itertools.combinations(players.keys(), 2):
        comparison_count += 1
        print(f"Comparison {comparison_count}/{total_comparisons}: {a} vs {b}")
        
        a_data, b_data = data[a], data[b]
        prompt = PROMPT_TMPL.format(
            a_summary=a_data["summary"],
            a_analysis=a_data["analysis"],
            b_summary=b_data["summary"],
            b_analysis=b_data["analysis"]
        )
        
        try:
            result = call_gpt(prompt)
        except Exception as e:
            print(f"  ERROR calling GPT for {a} vs {b}: {e}")
            continue

        if not isinstance(result, dict):
            print(f"  ERROR: Result is not a dict: {type(result)}")
            continue
            
        if "winner" not in result:
            print(f"  ERROR: No 'winner' key in result: {result}")
            continue
            
        if "reasoning" not in result:
            print(f"  ERROR: No 'reasoning' key in result: {result}")
            continue

        winner = result["winner"].strip().upper()
        reasoning = result["reasoning"]
        
        if winner == "A":
            old_a_rating = players[a].rating
            old_b_rating = players[b].rating
            
            players[a].update(players[b], 1)
            players[b].update(players[a], 0)
            players[a].wins.append(b)
            players[b].losses.append(a)
            
            print(f"  Winner: {a}")
            print(f"    {a}: {old_a_rating:.1f} -> {players[a].rating:.1f}")
            print(f"    {b}: {old_b_rating:.1f} -> {players[b].rating:.1f}")
            
        elif winner == "B":
            old_a_rating = players[a].rating
            old_b_rating = players[b].rating
            
            players[a].update(players[b], 0)
            players[b].update(players[a], 1)
            players[b].wins.append(a)
            players[a].losses.append(b)
            
            print(f"  Winner: {b}")
            print(f"    {a}: {old_a_rating:.1f} -> {players[a].rating:.1f}")
            print(f"    {b}: {old_b_rating:.1f} -> {players[b].rating:.1f}")
            
        else:
            print(f"  ERROR: Invalid winner value: '{winner}' (expected 'A' or 'B')")
            print(f"  Full result: {result}")
            continue

        # Store reasoning
        players[a].reasons.append(f"vs {b}: {reasoning}")
        players[b].reasons.append(f"vs {a}: {reasoning}")
        
        print(f"  Reasoning: {reasoning}")

    return players
def save_reports(players: Dict[str, EloPlayer], root: pathlib.Path) -> None:
    reports_dir = pathlib.Path("summary_analysis")
    reports_dir.mkdir(exist_ok=True)
    for name, p in players.items():
        with open(reports_dir / f"{name}.md", "w") as h:
            h.write(f"# {name}\n\nFinal ELO: {p.rating:.1f}\n\n")
            h.write(f"## Record\nWins: {len(p.wins)} | Losses: {len(p.losses)}\n\n")
            h.write("## Reasoning Log\n")
            for r in p.reasons:
                h.write(f"- {r}\n")
            h.write("\n---\n")
            h.write(f"[summary.json]({root / 'graphene' / f'summary_{name}.json'})\n")
            
            peaks_path = find_peaks_file_for_model(name, root / 'graphene')
            if peaks_path:
                h.write(f"[peaks_analysis.json]({peaks_path})\n")

def dump_ranking(players: Dict[str, EloPlayer]) -> None:
    ranking = sorted(players.values(), key=lambda p: p.rating, reverse=True)
    print("\n=== FINAL RANKING ===")
    for i, p in enumerate(ranking, 1):
        mark = "🏆" if i == 1 else "⭐" if i <= 3 else ""
        print(f"{i:2d}. {p.name:<20} {p.rating:7.1f} ({len(p.wins)}W-{len(p.losses)}L) {mark}")

    with open("summary_analysis/elo_ranking.json", "w") as f:
        json.dump(
            [
                {
                    "rank": i,
                    "model": p.name,
                    "elo": round(p.rating, 1),
                    "wins": len(p.wins),
                    "losses": len(p.losses)
                }
                for i, p in enumerate(ranking, 1)
            ],
            f,
            indent=2
        )

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results",
        type=pathlib.Path,
        help="Directory containing graphene sub-folder with model JSON files"
    )
    args = parser.parse_args()
    
    try:
        players = run_benchmark(args.root)
        save_reports(players, args.root)
        dump_ranking(players)
        print(f"\nBenchmarking completed! Results saved in summary_analysis/")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)