#!/usr/bin/env python3
"""
Detect kernels where top blame chains share nodes (can be merged into a DAG).

For each kernel × vendor, checks whether chains #2..#N share any node with
chain #1 (the highest-blame chain). When they do, it means a single symptom
traces back to multiple top-ranked root causes — a sign of rich dependency
structure that Leo can expose.

Parses existing Leo log files.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VENDOR_ORDER = ["amd", "nvidia", "intel"]
LOG_VENDOR_MAP = {"amd": "amd", "nvidia": "nvidia", "intel": "intel"}

EXCLUDED_KERNELS = {
    "Stream_ADD", "Stream_COPY", "Stream_DOT", "Stream_MUL", "Stream_TRIAD",
    "Basic_ARRAY_OF_PTRS", "Basic_COPY8", "Basic_DAXPY", "Basic_EMPTY",
    "Basic_INIT3", "Basic_INIT_VIEW1D", "Basic_INIT_VIEW1D_OFFSET",
    "Basic_MULADDSUB", "Basic_NESTED_INIT", "Basic_REDUCE3_INT",
}


@dataclass
class Chain:
    """A single blame chain: list of (opcode, location) from stall → root."""
    nodes: List[Tuple[str, str]]  # [(opcode, location), ...] stall-end first
    blame_cycles: int = 0

    @property
    def depth(self) -> int:
        return len(self.nodes)

    @property
    def node_set(self) -> Set[Tuple[str, str]]:
        return set(self.nodes)

    @property
    def loc_set(self) -> Set[str]:
        """Just the locations (ignoring opcode), for looser matching."""
        return {loc for _, loc in self.nodes}

    def __repr__(self):
        arrow = " ← ".join(f"{op}@{loc}" for op, loc in self.nodes)
        return f"{arrow}  [{self.blame_cycles:,}]"


@dataclass
class KernelVendorChains:
    kernel: str
    vendor: str
    stall_chains: List[Chain] = field(default_factory=list)   # from STALL ANALYSIS (1-hop)
    dep_chains: List[Chain] = field(default_factory=list)      # from DEPENDENCY CHAINS (multi-hop)

    @property
    def all_chains(self) -> List[Chain]:
        """All chains sorted by blame cycles descending."""
        combined = self.dep_chains + self.stall_chains
        return sorted(combined, key=lambda c: c.blame_cycles, reverse=True)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _detect_vendor(log_path: str) -> str:
    name = Path(log_path).name.lower()
    for pattern, vendor in LOG_VENDOR_MAP.items():
        if pattern in name:
            return vendor
    raise ValueError(f"Cannot detect vendor from filename: {log_path}")


def _parse_stall_chain(line: str) -> Optional[Chain]:
    """Parse a STALL ANALYSIS line: 'stall_loc stall_op <-- root_loc root_op  cycles % speedup'"""
    if "<--" not in line:
        return None
    parts = line.split("<--")
    if len(parts) != 2:
        return None

    stall_side = parts[0].strip()
    root_side = parts[1].strip()

    stall_tokens = stall_side.split()
    root_tokens = root_side.split()
    if len(stall_tokens) < 2 or len(root_tokens) < 2:
        return None

    stall_loc = stall_tokens[0]
    stall_op = stall_tokens[-1]
    root_loc = root_tokens[0]
    root_op = root_tokens[1]

    # Self-blame: skip
    if stall_op == root_op and stall_loc == root_loc:
        return None

    # Extract blame cycles (second-to-last number, or last)
    nums = re.findall(r'([\d,]+)', root_side)
    blame = 0
    if nums:
        blame = int(nums[-2].replace(",", "") if len(nums) >= 2 else nums[-1].replace(",", ""))

    return Chain(
        nodes=[(stall_op, stall_loc), (root_op, root_loc)],
        blame_cycles=blame,
    )


def _parse_dep_chain(line: str) -> Optional[Chain]:
    """Parse a DEPENDENCY CHAINS line: 'op @ loc ← op @ loc ← ... ← op @ loc  CYCLES'"""
    if "←" not in line:
        return None

    # Extract trailing blame cycles
    cycles_match = re.search(r'([\d,]+)\s*$', line.strip())
    blame = 0
    if cycles_match:
        blame = int(cycles_match.group(1).replace(",", ""))
        line_body = line[:cycles_match.start()].strip()
    else:
        line_body = line.strip()

    # Split by ←
    segments = line_body.split("←")
    nodes = []
    for seg in segments:
        seg = seg.strip()
        # Format: "op @ loc" or "op @ loc:col"
        at_match = re.match(r'(\S+)\s+@\s+(.+)', seg)
        if at_match:
            op = at_match.group(1)
            loc = at_match.group(2).strip()
            nodes.append((op, loc))

    if len(nodes) < 2:
        return None

    return Chain(nodes=nodes, blame_cycles=blame)


def parse_all_chains(log_paths: List[str]) -> Dict[str, Dict[str, KernelVendorChains]]:
    """Parse logs → {kernel_name: {vendor: KernelVendorChains}}"""
    result: Dict[str, Dict[str, KernelVendorChains]] = {}

    for log_path in log_paths:
        vendor = _detect_vendor(log_path)

        with open(log_path) as f:
            text = f.read()

        sections = re.split(r'\[(\d+)/(\d+)\]\s+(\S+)\s*/\s*(\S+)', text)

        i = 1
        while i + 4 <= len(sections):
            kernel_name = sections[i + 2]
            body = sections[i + 4]
            i += 5

            if kernel_name in EXCLUDED_KERNELS:
                continue

            if kernel_name not in result:
                result[kernel_name] = {}

            kvc = KernelVendorChains(kernel=kernel_name, vendor=vendor)

            # Parse STALL ANALYSIS
            stall_section = False
            stall_sep_count = 0
            for line in body.split("\n"):
                if "STALL ANALYSIS" in line:
                    stall_sep_count = 0
                    continue
                if stall_sep_count < 2 and line.strip().startswith("---"):
                    stall_sep_count += 1
                    if stall_sep_count == 2:
                        stall_section = True
                    continue
                if stall_section:
                    if line.startswith("===") or line.startswith("---"):
                        stall_section = False
                        continue
                    chain = _parse_stall_chain(line)
                    if chain:
                        kvc.stall_chains.append(chain)

            # Parse DEPENDENCY CHAINS
            chain_section = False
            saw_header = False
            for line in body.split("\n"):
                if "DEPENDENCY CHAINS" in line:
                    saw_header = True
                    continue
                if saw_header and line.strip().startswith("---"):
                    saw_header = False
                    chain_section = True
                    continue
                if chain_section:
                    if line.startswith("===") or line.startswith("---"):
                        chain_section = False
                        continue
                    chain = _parse_dep_chain(line)
                    if chain:
                        kvc.dep_chains.append(chain)

            result[kernel_name][vendor] = kvc

    return result


# ---------------------------------------------------------------------------
# Analysis: detect chain merges
# ---------------------------------------------------------------------------

def find_mergeable_chains(
    kvc: KernelVendorChains,
    top_n: int = 5,
) -> List[Tuple[int, Chain, Set[Tuple[str, str]]]]:
    """
    Check if any of chains #2..#top_n share nodes with chain #1.

    Returns list of (chain_index, chain, shared_nodes) for chains that merge.
    """
    chains = kvc.all_chains[:top_n]
    if len(chains) < 2:
        return []

    chain1 = chains[0]
    chain1_locs = chain1.loc_set
    chain1_nodes = chain1.node_set

    merges = []
    for idx in range(1, len(chains)):
        c = chains[idx]
        # Check shared locations (looser: same source line)
        shared_locs = chain1_locs & c.loc_set
        # Check shared nodes (stricter: same opcode + location)
        shared_nodes = chain1_nodes & c.node_set

        if shared_locs:
            merges.append((idx + 1, c, shared_locs))

    return merges


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detect chain merges across kernels")
    parser.add_argument("--logs", nargs="+", default=[
        "tests/data/pc/per-kernel/leo-amd.log",
        "tests/data/pc/per-kernel/leo-nvidia.log",
        "tests/data/pc/per-kernel/leo-intel.log",
    ])
    parser.add_argument("--top-n", type=int, default=5,
                        help="Check top N chains for mergeability")
    args = parser.parse_args()

    all_data = parse_all_chains(args.logs)

    # -----------------------------------------------------------------------
    # Per-kernel detailed output
    # -----------------------------------------------------------------------
    print(f"{'='*110}")
    print(f"Chain Merge Detection: Top-{args.top_n} chains sharing nodes with Chain #1")
    print(f"{'='*110}\n")

    # Store per-kernel-vendor results: (kernel, vendor) -> (n_merged, merged_rankings, total_chains)
    merge_info: Dict[str, Dict[str, Tuple[int, List[int], int]]] = {}

    for kernel_name in sorted(all_data.keys()):
        vendor_data = all_data[kernel_name]
        has_any_merge = False

        for vendor in VENDOR_ORDER:
            if vendor not in vendor_data:
                continue
            kvc = vendor_data[vendor]
            chains = kvc.all_chains
            top_chains = chains[:args.top_n]

            merges = find_mergeable_chains(kvc, top_n=args.top_n)
            rankings = [rank for rank, _, _ in merges]

            if kernel_name not in merge_info:
                merge_info[kernel_name] = {}
            merge_info[kernel_name][vendor] = (len(merges), rankings, len(top_chains))

            if merges:
                if not has_any_merge:
                    print(f"--- {kernel_name} ---")
                    has_any_merge = True

                print(f"  [{vendor.upper():>6}] {len(merges)} chain(s) merge with #1, rankings: {rankings}")
                print(f"           Chain #1: {chains[0]}")
                for rank, chain, shared in merges:
                    shared_str = ", ".join(sorted(shared))
                    print(f"           Chain #{rank}: {chain}")
                    print(f"                     shared: {shared_str}")
                print()

    # -----------------------------------------------------------------------
    # Summary table: all kernels × vendors, showing merge count + rankings
    # -----------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("COMPACT SUMMARY TABLE")
    print(f"{'='*110}")
    print(f"\n{'Kernel':<35} {'AMD':<25} {'NVIDIA':<25} {'INTEL':<25}")
    print(f"{'-'*110}")

    for kernel_name in sorted(merge_info.keys()):
        cols = []
        for vendor in VENDOR_ORDER:
            if vendor in merge_info[kernel_name]:
                n_merged, rankings, n_total = merge_info[kernel_name][vendor]
                if n_merged > 0:
                    r_str = ",".join(f"#{r}" for r in rankings)
                    cols.append(f"{n_merged} merged ({r_str})")
                else:
                    cols.append("--")
            else:
                cols.append("N/A")
        print(f"{kernel_name:<35} {cols[0]:<25} {cols[1]:<25} {cols[2]:<25}")

    # -----------------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*110}")

    for top_n_check in [args.top_n, 3]:
        print(f"\n--- Top-{top_n_check} chains ---")
        print(f"{'Vendor':<10} {'Has merge':<12} {'Total':<8} {'Ratio':<8} "
              f"{'Avg merged':<12} {'Max merged':<12} {'Most common rankings'}")
        print(f"{'-'*95}")

        for vendor in VENDOR_ORDER:
            n_has_merge = 0
            n_total = 0
            all_n_merged = []
            all_rankings = []

            for kernel_name in sorted(merge_info.keys()):
                if vendor not in merge_info[kernel_name]:
                    continue
                n_total += 1
                # Recompute for this top_n_check
                kvc = all_data[kernel_name][vendor]
                merges = find_mergeable_chains(kvc, top_n=top_n_check)
                rankings = [rank for rank, _, _ in merges]
                if merges:
                    n_has_merge += 1
                    all_n_merged.append(len(merges))
                    all_rankings.extend(rankings)

            ratio = f"{n_has_merge/n_total*100:.0f}%" if n_total > 0 else "N/A"
            avg_merged = f"{sum(all_n_merged)/len(all_n_merged):.1f}" if all_n_merged else "0"
            max_merged = str(max(all_n_merged)) if all_n_merged else "0"

            # Count ranking frequency
            rank_freq = {}
            for r in all_rankings:
                rank_freq[r] = rank_freq.get(r, 0) + 1
            top_ranks = sorted(rank_freq.items(), key=lambda x: -x[1])[:3]
            rank_str = ", ".join(f"#{r}({c}x)" for r, c in top_ranks)

            print(f"{vendor.upper():<10} {n_has_merge:<12} {n_total:<8} {ratio:<8} "
                  f"{avg_merged:<12} {max_merged:<12} {rank_str}")

    # -----------------------------------------------------------------------
    # Distribution: how many chains merge with #1
    # -----------------------------------------------------------------------
    print(f"\n{'='*110}")
    print(f"DISTRIBUTION: How many of top-{args.top_n} chains merge with #1?")
    print(f"{'='*110}")

    for vendor in VENDOR_ORDER:
        dist = {}  # n_merged -> count of kernels
        for kernel_name in sorted(merge_info.keys()):
            if vendor not in merge_info[kernel_name]:
                continue
            kvc = all_data[kernel_name][vendor]
            merges = find_mergeable_chains(kvc, top_n=args.top_n)
            n = len(merges)
            dist[n] = dist.get(n, 0) + 1

        print(f"\n  {vendor.upper()}:")
        for n in sorted(dist.keys()):
            bar = "#" * dist[n]
            label = f"{n} merged" if n > 0 else "0 (no merge)"
            print(f"    {label:<15} {dist[n]:>3} kernels  {bar}")

    # -----------------------------------------------------------------------
    # Cross-vendor highlights
    # -----------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("CROSS-VENDOR HIGHLIGHTS")
    print(f"{'='*110}")

    # Kernels with merges on all 3 vendors
    all3 = []
    any_merge = []
    for kernel_name in sorted(merge_info.keys()):
        vendors_with_merge = []
        for vendor in VENDOR_ORDER:
            if vendor in merge_info[kernel_name]:
                kvc = all_data[kernel_name][vendor]
                merges = find_mergeable_chains(kvc, top_n=args.top_n)
                if merges:
                    rankings = [r for r, _, _ in merges]
                    vendors_with_merge.append((vendor, len(merges), rankings))
        if len(vendors_with_merge) == 3:
            all3.append((kernel_name, vendors_with_merge))
        if vendors_with_merge:
            any_merge.append((kernel_name, vendors_with_merge))

    print(f"\nKernels with merges on ALL 3 vendors ({len(all3)}):")
    for kname, vmerges in all3:
        parts = []
        for v, n, ranks in vmerges:
            r_str = ",".join(f"#{r}" for r in ranks)
            parts.append(f"{v.upper()}: {n} merged ({r_str})")
        print(f"  {kname:<35} {' | '.join(parts)}")

    print(f"\nKernels with merges on ANY vendor ({len(any_merge)}):")
    for kname, vmerges in any_merge:
        parts = []
        for v, n, ranks in vmerges:
            r_str = ",".join(f"#{r}" for r in ranks)
            parts.append(f"{v.upper()}: {n}({r_str})")
        print(f"  {kname:<35} {' | '.join(parts)}")


if __name__ == "__main__":
    main()
