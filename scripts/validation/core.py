#!/usr/bin/env python3
"""Ground Truth Validation: Leo root causes vs actual optimization locations.

Compares Leo's reported root cause source lines with the actual code changes
that achieved speedup. Metric: source-line distance between Leo's reported
root cause and the optimization diff.

Steps:
  1. (Optional) Re-run Leo on unoptimized HPCToolkit profiles (CPU-only)
  2. Parse Leo output → extract top-K root cause source:line entries
  3. Compute diff between original and optimized source → modified lines
  4. Compare: Leo lines vs diff lines → distance metric
  5. (Optional, GPU) Measure actual speedup

Usage:
  uv run python -m scripts.validation.main
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import (
    VENDOR_SUFFIX,
    REALWORLD_APPS, STALL_LINE_RE, SOURCE_LOC_RE, HUNK_RE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RootCause:
    """A single root cause entry from Leo output."""

    stall_file: str
    stall_line: int
    stall_opcode: str
    root_file: str
    root_line: int
    root_opcode: str
    cycles: int
    pct: str
    speedup: str


@dataclass
class DiffRegion:
    """Modified line range from a unified diff."""

    start: int  # first modified line in original file
    end: int  # last modified line in original file (inclusive)


@dataclass
class ValidationResult:
    """Result for one kernel/vendor combination."""

    kernel: str
    vendor: str
    root_causes: list[RootCause] = field(default_factory=list)
    diff_lines: list[int] = field(default_factory=list)  # all modified lines
    diff_file: str = ""  # source file that was diffed
    top1_distance: Optional[int] = None
    top1_match: str = ""  # EXACT, NEAR, SAME-FUNC, SAME-FILE, NO-OPT, UNMAPPED, NO-DATA
    topk_min_distance: Optional[int] = None
    topk_best_match: str = ""
    has_optimization: bool = False
    has_leo_data: bool = False
    error: str = ""
    # Raw text for export and LLM eval
    leo_stall_text: str = ""
    leo_dep_chains: str = ""
    original_source: str = ""
    diff_text: str = ""


# ---------------------------------------------------------------------------
# Step 2: Parse Leo output
# ---------------------------------------------------------------------------


def extract_stall_section(text: str) -> str:
    """Extract the STALL ANALYSIS section from Leo output as raw text."""
    lines_out = []
    in_section = False
    for line in text.splitlines():
        if "STALL ANALYSIS" in line:
            in_section = True
        if in_section:
            lines_out.append(line)
            if lines_out and line.strip().startswith("====") and len(lines_out) > 2:
                break
    return "\n".join(lines_out) if lines_out else ""


def extract_dep_chains_section(text: str) -> str:
    """Extract the DEPENDENCY CHAINS section from Leo output as raw text."""
    lines_out = []
    in_section = False
    for line in text.splitlines():
        if "DEPENDENCY CHAINS" in line:
            in_section = True
        if in_section:
            lines_out.append(line)
            if lines_out and line.strip().startswith("====") and len(lines_out) > 2:
                break
    return "\n".join(lines_out) if lines_out else ""


def parse_leo_output(text: str, top_k: int = 5) -> list[RootCause]:
    """Parse Leo's STALL ANALYSIS table and return top-K root causes."""
    results = []
    in_stall_section = False

    for line in text.splitlines():
        stripped = line.strip()

        # Detect start of stall analysis section
        if "STALL ANALYSIS" in stripped:
            in_stall_section = True
            continue

        # Detect end (separator or next section)
        if in_stall_section and stripped.startswith("===="):
            in_stall_section = False
            continue

        if not in_stall_section:
            continue

        # Skip header/separator lines
        if stripped.startswith("---") or stripped.startswith("Stall Location"):
            continue

        m = STALL_LINE_RE.match(stripped)
        if not m:
            continue

        stall_loc, stall_op, root_loc, root_op, cycles_str, pct, speedup = m.groups()

        # Parse source locations
        stall_m = SOURCE_LOC_RE.match(stall_loc)
        root_m = SOURCE_LOC_RE.match(root_loc)

        stall_file = stall_m.group(1) if stall_m else ""
        stall_line = int(stall_m.group(2)) if stall_m else 0
        root_file = root_m.group(1) if root_m else ""
        root_line = int(root_m.group(2)) if root_m else 0

        cycles = int(cycles_str.replace(",", ""))

        results.append(RootCause(
            stall_file=stall_file,
            stall_line=stall_line,
            stall_opcode=stall_op.strip(),
            root_file=root_file,
            root_line=root_line,
            root_opcode=root_op,
            cycles=cycles,
            pct=pct,
            speedup=speedup,
        ))

    # Sort by cycles descending, take top-K
    results.sort(key=lambda r: r.cycles, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Step 3: Compute optimization diff
# ---------------------------------------------------------------------------


def kernel_to_source_path(kernel_name: str) -> tuple[str, str]:
    """Map kernel directory name to source subdirectory and base name.

    Returns (subdir, basename):
      Apps_LTIMES → ("apps", "LTIMES")
      Polybench_2MM → ("polybench", "POLYBENCH_2MM")
    """
    if kernel_name.startswith("Apps_"):
        base = kernel_name[5:]
        return "apps", base
    elif kernel_name.startswith("Polybench_"):
        base = kernel_name[10:]
        return "polybench", f"POLYBENCH_{base}"
    return "", kernel_name


def get_diff_lines(original: Path, optimized: Path) -> tuple[list[int], str]:
    """Run diff -u and extract modified line numbers from the original file.

    Returns (sorted_line_numbers, raw_diff_text).
    """
    if not original.exists() or not optimized.exists():
        return [], ""

    try:
        result = subprocess.run(
            ["diff", "-u", str(original), str(optimized)],
            capture_output=True, text=True,
        )
    except Exception:
        return [], ""

    raw_diff = result.stdout
    lines_modified: set[int] = set()
    current_old_line = 0
    current_old_count = 0
    old_offset = 0

    for diff_line in raw_diff.splitlines():
        hunk_m = HUNK_RE.match(diff_line)
        if hunk_m:
            current_old_line = int(hunk_m.group(1))
            current_old_count = int(hunk_m.group(2) or "1")
            old_offset = 0
            # Add all lines in the hunk range as "modified region"
            for i in range(current_old_count):
                lines_modified.add(current_old_line + i)
            continue

        if not current_old_line:
            continue

        # Lines starting with '-' are removed from original (modified)
        if diff_line.startswith("-") and not diff_line.startswith("---"):
            lines_modified.add(current_old_line + old_offset)
            old_offset += 1
        # Lines starting with '+' are additions (no original line consumed)
        elif diff_line.startswith("+") and not diff_line.startswith("+++"):
            # The insertion point in the original is adjacent to the current line
            lines_modified.add(current_old_line + old_offset)
        # Context lines
        elif diff_line.startswith(" "):
            old_offset += 1

    return sorted(lines_modified), raw_diff


def get_diff_for_kernel(
    benchmarks_dir: Path, kernel: str, vendor: str
) -> tuple[list[int], str, str]:
    """Get diff lines for a kernel/vendor combination.

    Returns (modified_lines, source_filename, raw_diff_text).
    """
    subdir, basename = kernel_to_source_path(kernel)
    suffix = VENDOR_SUFFIX.get(vendor, "")
    if not subdir or not suffix:
        return [], "", ""

    filename = f"{basename}{suffix}"
    original = benchmarks_dir / "original" / "src" / subdir / filename
    optimized = benchmarks_dir / "optimized" / "src" / subdir / filename

    diff_lines, raw_diff = get_diff_lines(original, optimized)
    return diff_lines, filename, raw_diff


def get_git_diff_lines(
    benchmarks_root: Path, git_diff_config: dict
) -> tuple[list[int], str, str]:
    """Get modified lines from a git diff between commits.

    Used for apps where the optimization is in git history (e.g., LULESH).
    Returns (modified_lines, source_filename, raw_diff_text).
    """
    repo_path = benchmarks_root / git_diff_config["repo"]
    base_commit = git_diff_config["base_commit"]
    file_path = git_diff_config["file"]

    if not repo_path.exists():
        return [], "", ""

    try:
        result = subprocess.run(
            ["git", "diff", f"{base_commit}..HEAD", "--", file_path],
            capture_output=True, text=True, cwd=str(repo_path),
        )
    except Exception:
        return [], "", ""

    raw_diff = result.stdout
    lines_modified: set[int] = set()
    current_old_line = 0
    current_old_count = 0
    old_offset = 0

    for diff_line in raw_diff.splitlines():
        hunk_m = HUNK_RE.match(diff_line)
        if hunk_m:
            current_old_line = int(hunk_m.group(1))
            current_old_count = int(hunk_m.group(2) or "1")
            old_offset = 0
            for i in range(current_old_count):
                lines_modified.add(current_old_line + i)
            continue

        if not current_old_line:
            continue

        if diff_line.startswith("-") and not diff_line.startswith("---"):
            lines_modified.add(current_old_line + old_offset)
            old_offset += 1
        elif diff_line.startswith("+") and not diff_line.startswith("+++"):
            lines_modified.add(current_old_line + old_offset)
        elif diff_line.startswith(" "):
            old_offset += 1

    filename = os.path.basename(file_path)
    return sorted(lines_modified), filename, raw_diff


# ---------------------------------------------------------------------------
# Step 4: Compare — distance metric
# ---------------------------------------------------------------------------


def classify_distance(distance: Optional[int]) -> str:
    """Classify a distance into match category."""
    if distance is None:
        return ""
    if distance == 0:
        return "EXACT"
    if distance <= 5:
        return "NEAR"
    if distance <= 20:
        return "SAME-FUNC"
    return "SAME-FILE"


def compute_distance(
    root_causes: list[RootCause],
    diff_lines: list[int],
    diff_filename: str,
) -> tuple[Optional[int], Optional[int]]:
    """Compute min distance from root cause lines to nearest diff line.

    Returns (top1_distance, topk_min_distance).
    """
    if not diff_lines:
        return None, None

    def min_dist_for_root(rc: RootCause) -> Optional[int]:
        # Check both stall location and root cause location
        candidates = []
        if rc.root_file and rc.root_line > 0:
            # Match by filename basename (Leo may report just basename)
            rc_base = os.path.basename(rc.root_file)
            diff_base = os.path.basename(diff_filename)
            if rc_base == diff_base:
                candidates.append(rc.root_line)
        if rc.stall_file and rc.stall_line > 0:
            stall_base = os.path.basename(rc.stall_file)
            diff_base = os.path.basename(diff_filename)
            if stall_base == diff_base:
                candidates.append(rc.stall_line)

        if not candidates:
            return None

        best = None
        for cand_line in candidates:
            for dl in diff_lines:
                d = abs(cand_line - dl)
                if best is None or d < best:
                    best = d
        return best

    # Top-1 distance
    top1_dist = None
    if root_causes:
        top1_dist = min_dist_for_root(root_causes[0])

    # Top-K minimum distance
    topk_min = None
    for rc in root_causes:
        d = min_dist_for_root(rc)
        if d is not None and (topk_min is None or d < topk_min):
            topk_min = d

    return top1_dist, topk_min


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------


def validate_kernel(
    results_dir: Path,
    benchmarks_dir: Path,
    kernel: str,
    vendor: str,
    top_k: int,
) -> ValidationResult:
    """Run validation for one kernel/vendor pair."""
    result = ValidationResult(kernel=kernel, vendor=vendor)

    # Step 1/2: Get Leo output (prefer latest over original)
    leo_file = results_dir / kernel / vendor / "leo_output_latest.txt"
    if not leo_file.exists():
        leo_file = results_dir / kernel / vendor / "leo_output.txt"
    leo_text = leo_file.read_text() if leo_file.exists() else None

    if not leo_text:
        result.error = "No Leo output found"
        return result

    result.has_leo_data = True
    result.leo_stall_text = extract_stall_section(leo_text)
    result.leo_dep_chains = extract_dep_chains_section(leo_text)

    # Step 2: Parse
    root_causes = parse_leo_output(leo_text, top_k)
    result.root_causes = root_causes

    if not root_causes:
        result.error = "No root causes parsed from Leo output"
        result.top1_match = "NO-DATA"
        result.topk_best_match = "NO-DATA"
        return result

    # Check if all root causes are hex-only (unmapped)
    has_source = any(rc.root_file or rc.stall_file for rc in root_causes)

    # Step 3: Get diff
    diff_lines, diff_filename, raw_diff = get_diff_for_kernel(benchmarks_dir, kernel, vendor)
    result.diff_lines = diff_lines
    result.diff_file = diff_filename
    result.diff_text = raw_diff
    result.has_optimization = len(diff_lines) > 0

    # Read original source file for LLM context
    if diff_filename:
        subdir, basename = kernel_to_source_path(kernel)
        orig_path = benchmarks_dir / "original" / "src" / subdir / diff_filename
        if orig_path.exists():
            result.original_source = orig_path.read_text()

    if not result.has_optimization:
        result.top1_match = "NO-OPT"
        result.topk_best_match = "NO-OPT"
        return result

    if not has_source:
        result.top1_match = "UNMAPPED"
        result.topk_best_match = "UNMAPPED"
        return result

    # Step 4: Compare
    top1_dist, topk_min = compute_distance(root_causes, diff_lines, diff_filename)
    result.top1_distance = top1_dist
    result.topk_min_distance = topk_min

    if top1_dist is not None:
        result.top1_match = classify_distance(top1_dist)
    else:
        result.top1_match = "UNMAPPED"

    if topk_min is not None:
        result.topk_best_match = classify_distance(topk_min)
    else:
        result.topk_best_match = "UNMAPPED"

    return result


TIMESTAMP_DIR_RE = re.compile(r"-\d{8}-\d{6}$")


def find_latest_result_dir(results_root: Path, pattern: str) -> Optional[Path]:
    """Find the latest result directory matching a glob pattern.

    Only considers directories whose names end with a YYYYMMDD-HHMMSS
    timestamp. Sorted by name so the last match is the latest.
    """
    matches = sorted([
        p for p in results_root.glob(pattern)
        if p.is_dir() and TIMESTAMP_DIR_RE.search(p.name)
    ])
    return matches[-1] if matches else None


def validate_realworld_app(
    results_root: Path,
    benchmarks_root: Path,
    app_name: str,
    vendor: str,
    app_config: dict,
    top_k: int,
) -> Optional[ValidationResult]:
    """Run validation for one real-world app / vendor pair."""
    pattern = app_config["result_patterns"].get(vendor)
    if not pattern:
        return None  # no config for this vendor

    source_pair = app_config.get("source_files", {}).get(vendor)
    git_diff_config = app_config.get("git_diff")
    if not source_pair and not git_diff_config:
        return None

    result = ValidationResult(kernel=app_name, vendor=vendor)

    # Find latest result directory
    result_dir = find_latest_result_dir(results_root, pattern)
    if not result_dir:
        result.error = f"No result dir matching {pattern}"
        return result

    # Read Leo output (prefer latest over original)
    leo_file = result_dir / "leo_output_latest.txt"
    if not leo_file.exists():
        leo_file = result_dir / "leo_output.txt"
    if not leo_file.exists():
        result.error = f"No leo_output.txt in {result_dir.name}"
        return result
    leo_text = leo_file.read_text()
    result.has_leo_data = True
    result.leo_stall_text = extract_stall_section(leo_text)
    result.leo_dep_chains = extract_dep_chains_section(leo_text)

    # Parse
    root_causes = parse_leo_output(leo_text, top_k)
    result.root_causes = root_causes
    if not root_causes:
        result.error = "No root causes parsed from Leo output"
        result.top1_match = "NO-DATA"
        result.topk_best_match = "NO-DATA"
        return result

    has_source = any(rc.root_file or rc.stall_file for rc in root_causes)

    # Diff original vs optimized
    if git_diff_config:
        diff_lines, diff_filename, raw_diff = get_git_diff_lines(benchmarks_root, git_diff_config)
        # Read original source from git
        repo_path = benchmarks_root / git_diff_config["repo"]
        orig_file = repo_path / git_diff_config["file"]
        if orig_file.exists():
            result.original_source = orig_file.read_text()
    else:
        orig_path = benchmarks_root / source_pair[0]
        opt_path = benchmarks_root / source_pair[1]
        diff_lines, raw_diff = get_diff_lines(orig_path, opt_path)
        diff_filename = orig_path.name
        if orig_path.exists():
            result.original_source = orig_path.read_text()
    result.diff_lines = diff_lines
    result.diff_file = diff_filename
    result.diff_text = raw_diff
    result.has_optimization = len(diff_lines) > 0

    if not result.has_optimization:
        result.top1_match = "NO-OPT"
        result.topk_best_match = "NO-OPT"
        return result

    if not has_source:
        result.top1_match = "UNMAPPED"
        result.topk_best_match = "UNMAPPED"
        return result

    # Compare
    top1_dist, topk_min = compute_distance(root_causes, diff_lines, diff_filename)
    result.top1_distance = top1_dist
    result.topk_min_distance = topk_min

    if top1_dist is not None:
        result.top1_match = classify_distance(top1_dist)
    else:
        result.top1_match = "UNMAPPED"

    if topk_min is not None:
        result.topk_best_match = classify_distance(topk_min)
    else:
        result.topk_best_match = "UNMAPPED"

    return result
