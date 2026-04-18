#!/usr/bin/env python3
"""Dump the full SC26 paper LaTeX source into a single Markdown file for LLM review."""

import json
import re
from collections import defaultdict
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent / "3rdparty/hpctoolkit_pcsampling/sc26"
OUTPUT = Path(__file__).resolve().parent.parent / "sc26_full_paper.md"

# Reading order matching main.tex \input sequence
SECTION_ORDER = [
    ("main.tex", "Main (preamble)"),
    ("sections/abstract.tex", "Abstract"),
    ("sections/introduction.tex", "I. Introduction"),
    ("sections/background.tex", "II. Background"),
    ("sections/backslicing.tex", "III. Backward Slicing through GPU Machine Code"),
    ("sections/llm.tex", "IV. LLM-Based Optimization"),
    ("sections/evaluation.tex", "V. Evaluation"),
    ("sections/casestudies.tex", "VI. Case Studies"),
    ("sections/related.tex", "VII. Related Work"),
    ("sections/conclusion.tex", "VIII. Conclusion"),
]

TABLE_FILES = [
    ("tables/capability.tex", "Table: Capability Comparison"),
    ("tables/hardware.tex", "Table: Hardware Platforms"),
    ("tables/workloads.tex", "Table: Workloads"),
    ("tables/backslice.tex", "Table: Back-Slicing Results (Table IV)"),
    ("tables/ablation.tex", "Table: Diagnostic Context Comparison (Table V)"),
]

LISTING_FILES = [
    ("listings/kernel.tex", "Listing: LTIMES_NOVIEW Kernel"),
    ("listings/kripke_chain.tex", "Figure: Kripke LEO Analysis (Source + Chain)"),
    ("listings/llama_diff.tex", "Listing: llama.cpp Diff"),
    ("listings/qs_original.tex", "Listing: QuickSilver Original"),
    ("listings/qs_optimized.tex", "Listing: QuickSilver Optimized"),
    ("listings/hipkittens_rmsnorm.tex", "Listing: HipKittens RMSNorm"),
    ("listings/llm_prompt.tex", "Listing: LLM Prompt"),
]

def read_file(rel_path: str) -> str:
    p = PAPER_DIR / rel_path
    if p.exists():
        return p.read_text()
    return f"[FILE NOT FOUND: {rel_path}]"

def main():
    parts = []
    parts.append("# SC26 Paper: Full LaTeX Source\n")
    parts.append(f"Generated for LLM review. Paper directory: `{PAPER_DIR}`\n")
    parts.append("---\n")

    # Sections in reading order
    for rel_path, title in SECTION_ORDER:
        parts.append(f"\n## {title}\n")
        parts.append(f"**File:** `{rel_path}`\n")
        parts.append("```latex")
        parts.append(read_file(rel_path))
        parts.append("```\n")

    # Tables
    parts.append("\n---\n# Tables\n")
    for rel_path, title in TABLE_FILES:
        parts.append(f"\n## {title}\n")
        parts.append(f"**File:** `{rel_path}`\n")
        parts.append("```latex")
        parts.append(read_file(rel_path))
        parts.append("```\n")

    # Listings
    parts.append("\n---\n# Listings\n")
    for rel_path, title in LISTING_FILES:
        parts.append(f"\n## {title}\n")
        parts.append(f"**File:** `{rel_path}`\n")
        parts.append("```latex")
        parts.append(read_file(rel_path))
        parts.append("```\n")

    # AD Appendix
    parts.append("\n---\n# AD Appendix (Artifact Description)\n")
    parts.append(f"\n## AD Appendix\n")
    parts.append(f"**File:** `ad_appendix.tex`\n")
    parts.append("```latex")
    parts.append(read_file("ad_appendix.tex"))
    parts.append("```\n")

    # Bibliography
    parts.append("\n---\n# Bibliography\n")
    parts.append(f"\n## refs.bib\n")
    parts.append(f"**File:** `refs.bib`\n")
    parts.append("```bibtex")
    parts.append(read_file("refs.bib"))
    parts.append("```\n")

    # Statistical Significance (paired t-tests)
    parts.append("\n---\n# Statistical Significance (Paired t-tests)\n")

    # Include the t-test script source code
    ttest_script = Path(__file__).resolve().parent / "statistical_significance.py"
    if ttest_script.exists():
        parts.append("## Script: `scripts/statistical_significance.py`\n")
        parts.append("```python")
        parts.append(ttest_script.read_text())
        parts.append("```\n")

    # Run t-tests and append results
    parts.append("## Results\n")
    try:
        from scipy import stats
        BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
        jsonl_files = sorted(BENCH_DIR.glob("*-results-*/*.jsonl"))
        if not jsonl_files:
            parts.append("No JSONL benchmark files found.\n")
        for jsonl_path in jsonl_files:
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]
            groups = defaultdict(lambda: {"orig": [], "opt": []})
            for r in records:
                groups[r["kernel"]]["orig"].append(r["orig_ms"])
                groups[r["kernel"]]["opt"].append(r["opt_ms"])

            parts.append(f"\n## {jsonl_path.name} ({jsonl_path.parent.name})\n")
            lines = []
            lines.append(f"| {'Kernel':<30s} | {'N':>3s} | {'Speedup':>8s} | {'t-stat':>8s} | {'p-value':>10s} | {'Sig':>3s} |")
            lines.append(f"|{'-'*31}|{'-'*4}|{'-'*9}|{'-'*9}|{'-'*11}|{'-'*4}|")
            sig_01 = 0
            total = 0
            for k in sorted(groups):
                orig = groups[k]["orig"]
                opt = groups[k]["opt"]
                n = min(len(orig), len(opt))
                if n < 2:
                    continue
                total += 1
                orig, opt = orig[:n], opt[:n]
                mean_spd = (sum(orig) / n) / (sum(opt) / n)  # ratio of means
                t, p = stats.ttest_rel(orig, opt)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                if p < 0.01:
                    sig_01 += 1
                lines.append(f"| {k:<30s} | {n:>3d} | {mean_spd:>7.2f}× | {t:>8.2f} | {p:>10.4g} | {sig:>3s} |")
            lines.append("")
            lines.append(f"**Significant at p<0.01: {sig_01}/{total}**")
            parts.append("\n".join(lines))
            parts.append("")
    except ImportError:
        parts.append("scipy not available; skipping t-tests.\n")

    OUTPUT.write_text("\n".join(parts))
    print(f"Written to {OUTPUT} ({OUTPUT.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
