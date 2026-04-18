#!/usr/bin/env python3
"""Find dependency chains that span the most distinct source files.

Scans all per-kernel Leo output files and ranks chains by file diversity.
Best candidates for the cross-vendor back-slicing figure are chains where
symptom, intermediate, and root cause are in different source files.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results" / "per-kernel"


def parse_chains(output_path: Path) -> list:
    """Extract dependency chains from a Leo output file."""
    chains = []
    in_chains = False
    text = output_path.read_text(errors="replace")

    for line in text.splitlines():
        if "DEPENDENCY CHAINS" in line:
            in_chains = True
            continue
        if in_chains and line.startswith("---"):
            if chains:  # second separator = end of section
                in_chains = False
            continue
        if not in_chains:
            continue

        line = line.strip()
        if not line:
            continue

        # Parse chain: "inst @ file:line ← inst @ file:line ← ..."
        # Extract cycles from end
        cycles_match = re.search(r'(\d[\d,]*)\s*$', line)
        cycles = int(cycles_match.group(1).replace(',', '')) if cycles_match else 0

        # Extract all "@ file:line" locations
        nodes = re.findall(r'(\S+)\s+@\s+(\S+)', line)
        if not nodes:
            continue

        files = []
        for inst, loc in nodes:
            # Extract just the filename (remove line:col)
            # loc could be "file.cpp:56:8" or "file.hpp:258" or "0x800000105668"
            if loc.startswith("0x"):
                files.append(("<unknown>", inst, loc))
            else:
                parts = loc.split(":")
                fname = parts[0]
                lineno = parts[1] if len(parts) > 1 else "?"
                col = parts[2] if len(parts) > 2 else None
                loc_str = f"{fname}:{lineno}" + (f":{col}" if col else "")
                files.append((fname, inst, loc_str))

        unique_files = set(f[0] for f in files if f[0] != "<unknown>")
        chains.append({
            "nodes": files,
            "unique_files": unique_files,
            "n_files": len(unique_files),
            "n_hops": len(files),
            "cycles": cycles,
        })

    return chains


def parse_stall_table(output_path: Path) -> list:
    """Extract stall analysis rows from a Leo output file."""
    rows = []
    in_table = False
    text = output_path.read_text(errors="replace")

    for line in text.splitlines():
        if "Stall Location" in line and "Root Cause Location" in line:
            in_table = True
            continue
        if in_table and line.startswith("---"):
            if rows:
                in_table = False
            continue
        if not in_table:
            continue

        # Parse: "file:line opcode <-- file:line opcode cycles % speedup"
        match = re.match(
            r'\s*(\S+)\s+(\S+)\s+<--\s+(\S+)\s+(\S+)\s+([\d,]+)\s+([\d.]+)%',
            line
        )
        if match:
            stall_loc, stall_op, cause_loc, cause_op, cycles_str, pct = match.groups()
            stall_file = stall_loc.split(":")[0]
            cause_file = cause_loc.split(":")[0]
            rows.append({
                "stall_loc": stall_loc,
                "stall_op": stall_op,
                "cause_loc": cause_loc,
                "cause_op": cause_op,
                "stall_file": stall_file,
                "cause_file": cause_file,
                "cross_file": stall_file != cause_file,
                "cycles": int(cycles_str.replace(',', '')),
                "pct": float(pct),
            })

    return rows


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    print("=" * 100)
    print("CROSS-FILE DEPENDENCY CHAIN ANALYSIS")
    print("=" * 100)

    # Collect all results
    all_chains = []
    all_stalls = []

    for kernel_dir in sorted(RESULTS_DIR.iterdir()):
        if not kernel_dir.is_dir():
            continue
        kernel = kernel_dir.name

        for vendor_dir in sorted(kernel_dir.iterdir()):
            if not vendor_dir.is_dir():
                continue
            vendor = vendor_dir.name

            output = vendor_dir / "leo_output.txt"
            if not output.exists():
                continue

            # Parse chains
            chains = parse_chains(output)
            for c in chains:
                c["kernel"] = kernel
                c["vendor"] = vendor
                all_chains.append(c)

            # Parse stall table
            stalls = parse_stall_table(output)
            for s in stalls:
                s["kernel"] = kernel
                s["vendor"] = vendor
                all_stalls.append(s)

    # === Report 1: Best multi-file chains ===
    print("\n" + "=" * 100)
    print("TOP 20 CHAINS BY NUMBER OF DISTINCT SOURCE FILES")
    print("=" * 100)

    sorted_chains = sorted(all_chains, key=lambda c: (c["n_files"], c["cycles"]), reverse=True)

    for i, c in enumerate(sorted_chains[:20]):
        files_str = ", ".join(sorted(c["unique_files"]))
        print(f"\n#{i+1}  {c['kernel']} / {c['vendor']}  "
              f"({c['n_files']} files, {c['n_hops']} hops, {c['cycles']:,} cycles)")
        print(f"    Files: {files_str}")
        for fname, inst, loc in c["nodes"]:
            print(f"      {inst:30s} @ {loc}")

    # === Report 2: Kernels with cross-file stalls on ALL 3 vendors ===
    print("\n" + "=" * 100)
    print("KERNELS WITH CROSS-FILE DEPENDENCIES ON MULTIPLE VENDORS")
    print("=" * 100)

    # Group by kernel
    kernel_cross = defaultdict(lambda: defaultdict(list))
    for s in all_stalls:
        if s["cross_file"]:
            kernel_cross[s["kernel"]][s["vendor"]].append(s)

    for kernel in sorted(kernel_cross.keys()):
        vendors = kernel_cross[kernel]
        vendor_names = sorted(vendors.keys())
        # Check for coverage across vendor families
        has_amd = any("amd" in v for v in vendor_names)
        has_nvidia = any("nvidia" in v for v in vendor_names)
        has_intel = any("intel" in v for v in vendor_names)
        n_families = sum([has_amd, has_nvidia, has_intel])

        if n_families >= 2:
            print(f"\n{kernel}  ({n_families} vendor families: "
                  f"{'AMD ' if has_amd else ''}{'NVIDIA ' if has_nvidia else ''}{'Intel' if has_intel else ''})")
            for vendor in vendor_names:
                top = max(vendors[vendor], key=lambda s: s["cycles"])
                print(f"  {vendor:20s}  {top['stall_op']:25s} <-- {top['cause_op']:25s}  "
                      f"{top['stall_file']:30s} <-- {top['cause_file']:30s}  "
                      f"{top['cycles']:>15,} cy ({top['pct']:.1f}%)")

    # === Report 3: Best 3-vendor examples for figure ===
    print("\n" + "=" * 100)
    print("BEST CANDIDATES FOR CROSS-VENDOR FIGURE (3 vendors, prefer cross-file)")
    print("=" * 100)

    # For each kernel, check if all 3 vendor families have chains
    kernel_chains = defaultdict(lambda: defaultdict(list))
    for c in all_chains:
        v = c["vendor"]
        if "amd" in v:
            family = "amd"
        elif "nvidia" in v:
            family = "nvidia"
        elif "intel" in v:
            family = "intel"
        else:
            continue
        kernel_chains[c["kernel"]][family].append(c)

    candidates = []
    for kernel, families in kernel_chains.items():
        if len(families) < 3:
            continue
        # Score: sum of n_files across best chain per vendor
        score = 0
        best = {}
        for fam in ["amd", "nvidia", "intel"]:
            if fam not in families:
                continue
            top = max(families[fam], key=lambda c: (c["n_files"], c["cycles"]))
            best[fam] = top
            score += top["n_files"]
        candidates.append((score, kernel, best))

    candidates.sort(reverse=True)

    for score, kernel, best in candidates[:10]:
        print(f"\n{kernel}  (total file diversity score: {score})")
        for fam in ["amd", "nvidia", "intel"]:
            if fam not in best:
                continue
            c = best[fam]
            chain_str = " <- ".join(f"{inst} @ {loc}" for fname, inst, loc in c["nodes"])
            print(f"  {fam:8s} ({c['n_files']} files, {c['n_hops']} hops): {chain_str}")


if __name__ == "__main__":
    main()
