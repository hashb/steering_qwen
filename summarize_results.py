#!/usr/bin/env python3
"""Create a side-by-side Markdown report from an experiment's JSONL."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def safe_text(text):
    # Keep model-generated Markdown/HTML from changing the report structure.
    return "\n".join("    " + line for line in (text.splitlines() or [""]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or results.jsonl file.")
    args = parser.parse_args()
    source = args.run / "results.jsonl" if args.run.is_dir() else args.run
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not records:
        parser.error("No completed records found.")
    groups = defaultdict(list)
    for row in records:
        groups[row["condition"]].append(row)
    lines = ["# Neuron steering comparison", "",
             "These metrics measure output changes, not whether the model became ‘evil’. "
             "Review the paired completions and use a separate labeled evaluation for behavioral claims.", "",
             "| Condition | Runs | Different from baseline | Mean next-token KL | Hit token limit |",
             "|---|---:|---:|---:|---:|"]
    for name, rows in groups.items():
        lines.append(f"| {name} | {len(rows)} | {sum(not r['same_tokens_as_baseline'] for r in rows)} "
                     f"| {statistics.mean(r['next_token_kl_baseline_to_condition'] for r in rows):.6f} "
                     f"| {sum(r['hit_token_limit'] for r in rows)} |")
    pairs = defaultdict(list)
    for row in records:
        pairs[(row["prompt_id"], row["seed"])].append(row)
    for (prompt_id, seed), rows in pairs.items():
        lines.extend(["", f"## Prompt {prompt_id}, seed {seed}", "", safe_text(rows[0]["prompt"])])
        for row in rows:
            lines.extend(["", f"### {row['condition']}", "", safe_text(row["completion"]), ""])
    output = source.parent / "report.md"
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:7 + len(groups)]))
    print(f"\nFull report: {output}")


if __name__ == "__main__":
    main()
