#!/usr/bin/env python3
"""
Compact cross-vendor dependency chain overview — all kernels on one page.

Each kernel gets a small subplot with 3 vertical chains (one per vendor).
Chain #1 is the main trunk; chains #2-#5 that share nodes with #1 are
drawn as small branches, showing the DAG structure Leo exposes.

Parses existing Leo log files (no re-analysis needed).

Usage:
    python scripts/plot_chain_overview.py \
        --logs tests/data/pc/per-kernel/leo-amd.log \
              tests/data/pc/per-kernel/leo-nvidia.log \
              tests/data/pc/per-kernel/leo-intel.log \
        -o chain_overview.pdf

    # Quick test with first 3 kernels:
    python scripts/plot_chain_overview.py --limit 3 --cols 3 -o /tmp/test3.pdf
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Vendor configuration
# ---------------------------------------------------------------------------
VENDORS = {
    "amd":    {"label": "AMD MI300",    "color": "#cc2936", "marker": "o"},
    "nvidia": {"label": "NVIDIA A100",  "color": "#4a8c1c", "marker": "D"},
    "intel":  {"label": "Intel PVC",    "color": "#1a6fb5", "marker": "s"},
}
VENDOR_ORDER = ["amd", "nvidia", "intel"]

# Map log filename patterns to vendor keys
LOG_VENDOR_MAP = {"amd": "amd", "nvidia": "nvidia", "intel": "intel"}

# RAJAPerf group order for display
GROUP_ORDER = ["Algorithm", "Apps", "Basic", "Lcals", "Polybench"]

# Kernels too simple for cross-vendor analysis (see docs/KERNEL_SELECTION.md)
EXCLUDED_KERNELS = {
    "Stream_ADD", "Stream_COPY", "Stream_DOT", "Stream_MUL", "Stream_TRIAD",
    "Basic_ARRAY_OF_PTRS", "Basic_COPY8", "Basic_DAXPY", "Basic_EMPTY",
    "Basic_INIT3", "Basic_INIT_VIEW1D", "Basic_INIT_VIEW1D_OFFSET",
    "Basic_MULADDSUB", "Basic_NESTED_INIT", "Basic_REDUCE3_INT",
}


@dataclass
class ParsedChain:
    """A parsed blame chain with node locations."""
    nodes: List[str]   # list of location strings, from stall end to root
    blame: int = 0

    @property
    def depth(self) -> int:
        return len(self.nodes)

    @property
    def loc_set(self) -> Set[str]:
        return set(self.nodes)


@dataclass
class VendorChainInfo:
    """All chain info for one vendor on one kernel."""
    chain1: Optional[ParsedChain] = None      # top-1 chain (highest blame)
    merged_chains: List[ParsedChain] = field(default_factory=list)  # chains that share nodes with #1
    branch_depths: List[int] = field(default_factory=list)  # depth in chain1 where each branch occurs
    stall_ratio: float = 0.0
    all_chains: List[ParsedChain] = field(default_factory=list)  # all parsed chains sorted by blame


@dataclass
class KernelSummary:
    name: str
    vendor_info: Dict[str, VendorChainInfo] = field(default_factory=dict)

    @property
    def group(self) -> str:
        if "_" in self.name:
            return self.name.split("_", 1)[0]
        return "Other"

    @property
    def short_name(self) -> str:
        if "_" in self.name:
            parts = self.name.split("_", 1)
            if parts[0] in GROUP_ORDER:
                return parts[1]
        return self.name


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _detect_vendor(log_path: str) -> str:
    name = Path(log_path).name.lower()
    for pattern, vendor in LOG_VENDOR_MAP.items():
        if pattern in name:
            return vendor
    raise ValueError(f"Cannot detect vendor from filename: {log_path}")


def _parse_stall_line(line: str) -> Optional[ParsedChain]:
    """Parse a STALL ANALYSIS line into a 2-node chain."""
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

    # Skip self-blame
    if stall_op == root_op and stall_loc == root_loc:
        return None

    nums = re.findall(r'([\d,]+)', root_side)
    blame = 0
    if nums:
        blame = int(nums[-2].replace(",", "") if len(nums) >= 2 else nums[-1].replace(",", ""))

    return ParsedChain(nodes=[stall_loc, root_loc], blame=blame)


def _parse_dep_line(line: str) -> Optional[ParsedChain]:
    """Parse a DEPENDENCY CHAINS line into a multi-node chain."""
    if "\u2190" not in line:  # ←
        return None

    cycles_match = re.search(r'([\d,]+)\s*$', line.strip())
    blame = 0
    if cycles_match:
        blame = int(cycles_match.group(1).replace(",", ""))
        line_body = line[:cycles_match.start()].strip()
    else:
        line_body = line.strip()

    segments = line_body.split("\u2190")
    nodes = []
    for seg in segments:
        seg = seg.strip()
        at_match = re.match(r'(\S+)\s+@\s+(.+)', seg)
        if at_match:
            nodes.append(at_match.group(2).strip())

    if len(nodes) < 2:
        return None
    return ParsedChain(nodes=nodes, blame=blame)


def parse_logs(log_paths: List[str], top_n: int = 5) -> List[KernelSummary]:
    """Parse Leo log files, extract chains, detect merges."""
    kernel_map: Dict[str, KernelSummary] = {}

    for log_path in log_paths:
        vendor = _detect_vendor(log_path)
        print(f"Parsing {log_path} (vendor={vendor})")

        with open(log_path) as f:
            text = f.read()

        sections = re.split(r'\[(\d+)/(\d+)\]\s+(\S+)\s*/\s*(\S+)', text)

        i = 1
        while i + 4 <= len(sections):
            kernel_name = sections[i + 2]
            body = sections[i + 4]
            i += 5

            if kernel_name not in kernel_map:
                kernel_map[kernel_name] = KernelSummary(name=kernel_name)
            ks = kernel_map[kernel_name]

            vinfo = VendorChainInfo()

            # Extract stall ratio
            stall_match = re.search(r'Overall Stall Ratio:\s*([\d.]+)%', body)
            if stall_match:
                vinfo.stall_ratio = float(stall_match.group(1)) / 100.0

            # Parse STALL ANALYSIS (1-hop chains)
            stall_chains = []
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
                    chain = _parse_stall_line(line)
                    if chain:
                        stall_chains.append(chain)

            # Parse DEPENDENCY CHAINS (multi-hop)
            dep_chains = []
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
                    chain = _parse_dep_line(line)
                    if chain:
                        dep_chains.append(chain)

            # Combine and sort by blame
            all_chains = sorted(dep_chains + stall_chains, key=lambda c: c.blame, reverse=True)
            vinfo.all_chains = all_chains

            # Find chain #1 and merges
            top_chains = all_chains[:top_n]
            if top_chains:
                vinfo.chain1 = top_chains[0]
                chain1_locs = vinfo.chain1.loc_set

                for c in top_chains[1:]:
                    shared = chain1_locs & c.loc_set
                    if shared:
                        vinfo.merged_chains.append(c)
                        # Find the deepest shared node position in chain1
                        # (depth = index in chain1.nodes, 0 = stall end)
                        best_depth = -1
                        for shared_loc in shared:
                            if shared_loc in vinfo.chain1.nodes:
                                idx = vinfo.chain1.nodes.index(shared_loc)
                                best_depth = max(best_depth, idx)
                        if best_depth >= 0:
                            vinfo.branch_depths.append(best_depth)
                        else:
                            vinfo.branch_depths.append(0)
            elif vinfo.stall_ratio > 0:
                # Self-blame: single node
                vinfo.chain1 = ParsedChain(nodes=["self"], blame=0)

            ks.vendor_info[vendor] = vinfo

    # Sort by group order, then name
    def sort_key(ks):
        g = ks.group
        order = GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
        return (order, ks.name)
    return sorted(kernel_map.values(), key=sort_key)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(
    kernels: List[KernelSummary],
    output: str,
    cols: int = 10,
):
    """Draw a single-page figure with all kernels as small chain subplots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Publication-quality defaults
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    n = len(kernels)
    rows = math.ceil(n / cols)

    cell_w = 1.2
    cell_h = 1.5
    fig_w = cols * cell_w + 0.8
    fig_h = rows * cell_h + 1.2

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    axes_flat = [axes[r][c] for r in range(rows) for c in range(cols)]

    # Global max depth for consistent y-axis
    max_depth = 0
    for ks in kernels:
        for vinfo in ks.vendor_info.values():
            if vinfo.chain1:
                max_depth = max(max_depth, vinfo.chain1.depth)
    max_depth = max(max_depth, 2)

    # Detect group boundaries
    groups = []
    prev_group = None
    for i, ks in enumerate(kernels):
        if ks.group != prev_group:
            groups.append((i, ks.group))
            prev_group = ks.group

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue
        _draw_kernel_cell(ax, kernels[i], max_depth)

    # Group labels
    for gi, (kidx, gname) in enumerate(groups):
        ax = axes_flat[kidx]
        ax.annotate(
            gname,
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-4, 8), textcoords="offset points",
            fontsize=5.5, fontweight="bold", color="#333",
            fontstyle="italic",
        )

    # Legend
    legend_handles = []
    for vk in VENDOR_ORDER:
        cfg = VENDORS[vk]
        handle = mlines.Line2D(
            [], [],
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=4,
            markeredgecolor=cfg["color"],
            markerfacecolor=cfg["color"],
            linewidth=0.8,
            label=cfg["label"],
        )
        legend_handles.append(handle)
    # Source kernel node
    handle_src = mlines.Line2D(
        [], [],
        color="#555", marker="o", markersize=4,
        markeredgecolor="#333", markeredgewidth=0.6,
        markerfacecolor="white", linewidth=0,
        label="Source kernel",
    )
    legend_handles.append(handle_src)
    # "No stall" marker
    handle_x = mlines.Line2D(
        [], [],
        color="#999", marker="x", markersize=4,
        markeredgewidth=0.7, linewidth=0,
        label="No stalls", alpha=0.5,
    )
    legend_handles.append(handle_x)
    # Branch marker
    handle_branch = mlines.Line2D(
        [], [],
        color="#888", marker="o", markersize=3,
        markeredgewidth=0.3, markerfacecolor="#888",
        linewidth=0.6, linestyle="--",
        label="Merged chain branch",
    )
    legend_handles.append(handle_branch)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        fontsize=6,
        frameon=True,
        fancybox=False,
        edgecolor="#ccc",
        facecolor="white",
        borderpad=0.4,
        columnspacing=1.2,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, 1.0 - 0.08 / fig_h),
    )

    plt.tight_layout(rect=[0, 0, 1, 1.0 - 0.55 / fig_h], h_pad=1.0, w_pad=0.3)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\nSaved overview to {output}")
    plt.close(fig)


def _draw_kernel_cell(ax, ks: KernelSummary, max_depth: int):
    """Draw one kernel's mini subplot with a shared source node and 3 vendor chains."""
    n_vendors = len(VENDOR_ORDER)
    x_positions = {vk: i for i, vk in enumerate(VENDOR_ORDER)}
    x_center = (n_vendors - 1) / 2.0  # center of vendor columns

    # The source node sits one level below the stall end (y=0),
    # since chains grow upward from stall (y=0) to root cause (y=depth).
    # The source node represents the shared kernel source code.
    source_y = -1

    ax.set_xlim(-0.8, n_vendors - 0.2)
    ax.set_ylim(source_y - 0.5, max_depth + 0.5)

    # Clean style
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#fafafa")
    ax.set_yticks([])
    ax.set_xticks([])

    # Grid lines (chain depth levels only, not the source level)
    for d in range(max_depth + 1):
        ax.axhline(y=d, color="#e8e8e8", linewidth=0.3, zorder=0)

    # Kernel name
    ax.set_title(
        ks.short_name, fontsize=4.5, pad=2, color="#333",
        fontfamily="sans-serif", fontweight="medium",
    )

    # Draw shared source node at the bottom (represents the kernel source code).
    # Chains grow upward from this common origin.
    ax.plot(x_center, source_y, "o", color="#555", markersize=4.5,
            markeredgecolor="#333", markeredgewidth=0.6,
            markerfacecolor="white", alpha=0.9, zorder=6)

    # Draw each vendor's chain and connect from source node upward
    for vk in VENDOR_ORDER:
        cfg = VENDORS[vk]
        color = cfg["color"]
        marker = cfg["marker"]
        x = x_positions[vk]

        if vk not in ks.vendor_info:
            ax.plot(x, 0, "x", color=color, markersize=3.5, markeredgewidth=0.6,
                    alpha=0.35, zorder=5)
            # Connect source to "no data" marker
            ax.plot([x_center, x], [source_y, 0], color="#ccc",
                    linewidth=0.4, alpha=0.3, linestyle=":", zorder=1)
            continue

        vinfo = ks.vendor_info[vk]

        if vinfo.chain1 is None or vinfo.chain1.depth == 0:
            ax.plot(x, 0, "x", color=color, markersize=3.5, markeredgewidth=0.6,
                    alpha=0.35, zorder=5)
            ax.plot([x_center, x], [source_y, 0], color="#ccc",
                    linewidth=0.4, alpha=0.3, linestyle=":", zorder=1)
            continue

        depth = vinfo.chain1.depth
        ys = list(range(depth))
        xs = [x] * depth

        # Edge from source node up to bottom of vendor chain (y=0, the stall point)
        ax.plot([x_center, x], [source_y, 0], color=color,
                linewidth=0.5, alpha=0.35, zorder=1)

        # Main trunk line
        ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.5, zorder=2)

        # Main trunk nodes
        for yi in ys:
            ax.plot(x, yi, marker, color=color, markersize=3.0,
                    markeredgecolor=color, markeredgewidth=0.4,
                    markerfacecolor=color, alpha=0.85, zorder=5)

        # Draw branches for merged chains
        n_branches = len(vinfo.branch_depths)
        if n_branches > 0:
            chain1_locs = vinfo.chain1.loc_set
            for bi in range(n_branches):
                branch_y = vinfo.branch_depths[bi]
                merged = vinfo.merged_chains[bi]

                unique_count = sum(1 for loc in merged.nodes if loc not in chain1_locs)
                unique_count = max(unique_count, 1)

                side = 1 if bi % 2 == 0 else -1
                bx = x + side * 0.3

                prev_x, prev_y = x, branch_y
                for ni in range(unique_count):
                    by = branch_y + (ni + 1)

                    ax.plot([prev_x, bx], [prev_y, by],
                            color=color, linewidth=0.5, alpha=0.45,
                            linestyle="--", zorder=3)

                    ax.plot(bx, by, marker, color=color, markersize=2.2,
                            markeredgecolor=color, markeredgewidth=0.3,
                            markerfacecolor=color, alpha=0.55, zorder=4)

                    prev_x, prev_y = bx, by


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_LOGS = [
    "tests/data/pc/per-kernel/leo-amd.log",
    "tests/data/pc/per-kernel/leo-nvidia.log",
    "tests/data/pc/per-kernel/leo-intel.log",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compact cross-vendor chain depth overview with branch detection")
    parser.add_argument("--logs", nargs="+", default=DEFAULT_LOGS,
                        help="Leo log files (one per vendor)")
    parser.add_argument("-o", "--output", default="/tmp/chain_overview.pdf",
                        help="Output file path (PDF or PNG)")
    parser.add_argument("--cols", type=int, default=10,
                        help="Number of columns in the grid")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N kernels (0 = all)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Check top N chains for merge detection")
    parser.add_argument("--no-filter", action="store_true",
                        help="Include all kernels (skip exclusion filter)")
    args = parser.parse_args()

    kernels = parse_logs(args.logs, top_n=args.top_n)
    if not kernels:
        print("No kernels found in logs.")
        sys.exit(1)
    print(f"Parsed {len(kernels)} kernels")

    if not args.no_filter:
        before = len(kernels)
        kernels = [k for k in kernels if k.name not in EXCLUDED_KERNELS]
        print(f"Filtered: {before} -> {len(kernels)} kernels ({before - len(kernels)} excluded)")

    if args.limit > 0:
        kernels = kernels[:args.limit]
        print(f"Limited to first {args.limit} kernels")

    # Print merge stats
    for ks in kernels:
        branches = []
        for vk in VENDOR_ORDER:
            if vk in ks.vendor_info:
                vi = ks.vendor_info[vk]
                d = vi.chain1.depth if vi.chain1 else 0
                nb = len(vi.merged_chains)
                branches.append(f"{vk}={d}d/{nb}b")
            else:
                branches.append(f"{vk}=?")
        print(f"  {ks.name:<35} {', '.join(branches)}")

    plot_overview(kernels, output=args.output, cols=args.cols)


if __name__ == "__main__":
    main()
