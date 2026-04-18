#!/usr/bin/env python3
"""Extract validation data from HPCToolkit databases and Leo outputs.

Reads all databases once, saves results to a JSON cache file.
plot_validation_paper.py then reads the cache (instant).

Usage:
    uv run python scripts/extract_validation_data.py
    uv run python scripts/extract_validation_data.py --output results/validation/cache.json
"""

import json
import glob
import os
import re
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEO_ROOT = os.path.dirname(SCRIPT_DIR)

KERNEL_DIRS = [
    "Apps_MASS3DEA", "Apps_LTIMES_NOVIEW", "Apps_LTIMES",
    "Polybench_3MM", "Polybench_2MM", "Polybench_GEMM",
    "Apps_PRESSURE", "Apps_ENERGY", "Apps_FIR",
    "Apps_ZONAL_ACCUMULATION_3D", "Apps_VOL3D", "Apps_DEL_DOT_VEC_2D",
    "Apps_DIFFUSION3DPA", "Apps_CONVECTION3DPA", "Apps_MASS3DPA",
]
VENDORS = ["nvidia", "amd", "intel"]


def extract_culprit_category(leo_path: str) -> str:
    """Extract stall category from Leo output.

    Strategy:
    1. Parse stall table top-1 row: get both stall_opcode and root_opcode
    2. If root_opcode is a memory op (LDG, global_load, etc.) → gmem
    3. If root_opcode is a sync op (s_waitcnt, BAR) → check chain for
       the actual memory op being waited on
    4. If self-stall (root == stall, both compute) → idep
    5. Use stall_opcode as tiebreaker for ambiguous cases (s_waitcnt
       stall → look at what it's waiting for)
    """
    if not os.path.exists(leo_path):
        return None
    with open(leo_path) as f:
        text = f.read()

    # Parse stall table: get top-1 stall_opcode and root_opcode
    stall_opcode = None
    root_opcode = None
    in_table = False
    for line in text.split('\n'):
        if 'Stall Location' in line and 'Root Cause' in line:
            in_table = True
            continue
        if in_table and '---' in line and len(line.strip()) > 10:
            continue
        if in_table:
            if line.strip() == '' or '===' in line:
                break
            # Extract stall opcode (before <--)
            stall_match = re.match(r'\s*\S+\s+(\S+)\s+<--', line)
            if stall_match:
                stall_opcode = stall_match.group(1)
            # Extract root opcode (after <--)
            root_match = re.search(r'<--\s+\S+\s+(\S+)', line)
            if root_match:
                root_opcode = root_match.group(1)
            break  # Only need top-1

    if not root_opcode:
        return None

    root_cat = opcode_to_category(root_opcode)
    stall_cat = opcode_to_category(stall_opcode) if stall_opcode else root_cat

    # If root is a memory load → gmem (clear case)
    if root_cat == "gmem":
        return "gmem"

    # If stall is s_waitcnt or sync → the stall is memory/sync, use gmem
    # (s_waitcnt waits for memory ops; the burden should be measured as gmem)
    if stall_opcode and (stall_opcode.startswith('s_waitcnt') or stall_cat == "sync"):
        # Check chain: if chain ends at a memory op, use gmem
        in_chains = False
        for line in text.split('\n'):
            if 'DEPENDENCY CHAINS' in line:
                in_chains = True
                continue
            if in_chains and line.strip() and '←' in line:
                # Check if any node in chain is a memory op
                for part in line.split('←'):
                    m = re.match(r'\s*(\S+)\s+@', part.strip())
                    if m and opcode_to_category(m.group(1)) == "gmem":
                        return "gmem"
                break
            if in_chains and (line.strip() == '' or '===' in line):
                break
        # No memory op in chain → report as sync
        return "sync"

    # If self-stall (compute → compute) → idep
    if root_cat == "idep":
        return "idep"

    # Default
    return root_cat


def opcode_to_category(opcode: str) -> str:
    op = opcode.upper().split('.')[0]
    if op in ('LDG', 'STG', 'LD', 'ST', 'ATOM', 'RED', 'ATOMG', 'ULDC', 'LDC'):
        return "gmem"
    if any(opcode.startswith(p) for p in ('global_load', 'global_store', 'buffer_load',
                                           'buffer_store', 'flat_load', 'flat_store')):
        return "gmem"
    if op in ('LDS', 'STS', 'ATOMS'):
        return "smem"
    if any(opcode.startswith(p) for p in ('ds_read', 'ds_write', 'ds_bpermute', 's_load')):
        return "gmem"  # AMD reports LDS stalls under gcycles:stl:mem
    if op in ('BAR', 'DEPBAR', 'WARPSYNC', 'S_BARRIER'):
        return "sync"
    if opcode.startswith('s_waitcnt'):
        return "sync"
    if op in ('SEND', 'SENDS'):
        return "send"
    return "idep"


CATEGORY_METRICS = {
    "gmem": ["gcycles:stl:gmem", "gcycles:stl:mem"],
    "smem": ["gcycles:stl:lmem", "gcycles:stl:lgkm"],
    "sync": ["gcycles:stl:sync"],
    "idep": ["gcycles:stl:idep"],
    "send": ["gcycles:stl:send"],
}


def read_db_metrics(db_path: str) -> dict:
    """Read GPU metrics summary from an HPCToolkit database."""
    from leo.db.reader import DatabaseReader
    reader = DatabaseReader(db_path)
    return reader.get_gpu_metrics_summary()


def get_burden(metrics: dict, category: str) -> tuple:
    """Compute (burden_pct, category_cycles, total_cycles)."""
    total = metrics.get("gcycles", 0)
    if total == 0:
        return 0.0, 0, 0
    cat_cycles = 0
    for m in CATEGORY_METRICS.get(category, []):
        cat_cycles += metrics.get(m, 0)
    return cat_cycles / total * 100.0, cat_cycles, total


def process_one(task):
    """Process a single kernel/vendor pair — read before and after DBs."""
    key, category, before_db, after_db = task
    try:
        before_metrics = read_db_metrics(before_db)
        before_burden, before_cycles, before_total = get_burden(before_metrics, category)
    except Exception as e:
        return key, None, str(e)

    after_burden, after_cycles, after_total = None, 0, 0
    if after_db:
        try:
            after_metrics = read_db_metrics(after_db)
            after_burden, after_cycles, after_total = get_burden(after_metrics, category)
        except Exception:
            pass

    return key, {
        "category": category,
        "before_burden": before_burden,
        "before_cycles": before_cycles,
        "before_total": before_total,
        "after_burden": after_burden,
        "after_cycles": after_cycles,
        "after_total": after_total,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before-dir', default=os.path.join(LEO_ROOT, 'results/per-kernel'))
    parser.add_argument('--after-dir', default=os.path.join(LEO_ROOT, 'results/validation'))
    parser.add_argument('--output', default=os.path.join(LEO_ROOT, 'results/validation/cache.json'))
    args = parser.parse_args()

    # Build task list
    tasks = []
    for kernel_dir in KERNEL_DIRS:
        for vendor in VENDORS:
            key = f"{kernel_dir}/{vendor}"

            # Culprit category from Leo BEFORE
            leo_paths = [
                os.path.join(args.before_dir, kernel_dir, vendor, "leo_output_latest.txt"),
                os.path.join(args.before_dir, kernel_dir, vendor, "leo_output.txt"),
            ]
            category = None
            for lp in leo_paths:
                category = extract_culprit_category(lp)
                if category:
                    break
            if not category:
                continue

            before_dbs = glob.glob(os.path.join(args.before_dir, kernel_dir, vendor, "hpctoolkit-*-database"))
            after_dbs = glob.glob(os.path.join(args.after_dir, kernel_dir, vendor, "optimized", "hpctoolkit-*-database"))
            if not before_dbs:
                continue

            tasks.append((key, category, before_dbs[0], after_dbs[0] if after_dbs else None))

    print(f"Found {len(tasks)} tasks to process (parallel with {os.cpu_count()} workers)")
    print(flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            fkey, data, err = future.result()
            if err:
                print(f"[{i}/{len(tasks)}] {fkey}: ERROR {err}", flush=True)
            elif data:
                results[fkey] = data
                b = data["before_burden"]
                a = data["after_burden"]
                cat = data["category"]
                if a is not None:
                    print(f"[{i}/{len(tasks)}] {fkey}: {cat} {b:.1f}% → {a:.1f}%", flush=True)
                else:
                    print(f"[{i}/{len(tasks)}] {fkey}: {cat} {b:.1f}% (no AFTER)", flush=True)

    # Save cache
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} entries to {args.output}")


if __name__ == '__main__':
    main()
