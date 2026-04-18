#!/usr/bin/env python3
"""
Plot cross-vendor dependency chains from Leo analysis.

Reads HPCToolkit per-kernel measurements for multiple GPU vendors, runs Leo
back-slicing analysis, and generates a matplotlib figure showing dependency
chains side by side.

Usage:
    # Inside Docker container (with data mounted):
    uv run python scripts/plot_cross_vendor_chains.py \
        --kernel Apps_CONVECTION3DPA \
        --output cross_vendor_chains.pdf

    # Custom data directory:
    uv run python scripts/plot_cross_vendor_chains.py \
        --kernel Apps_CONVECTION3DPA \
        --data-dir tests/data/pc/per-kernel \
        --top-chains 2 --output comparison.pdf
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vendor configuration
# ---------------------------------------------------------------------------
VENDORS = {
    "amd": {"label": "AMD MI300", "color": "#ed1c24", "arch": "amd", "dir": "amd"},
    "nvidia": {"label": "NVIDIA A100", "color": "#76b900", "arch": "nvidia", "dir": "nvidia-arm"},
    "intel": {"label": "Intel PVC", "color": "#0071c5", "arch": "intel", "dir": "intel"},
}
STALL_COLOR = "#dc322f"
BG_GRAY = "#f5f5f5"

# ---------------------------------------------------------------------------
# Data classes for extracted chain data
# ---------------------------------------------------------------------------

@dataclass
class NodeData:
    opcode: str
    source_file: str = ""
    source_line: int = 0
    is_stall: bool = False  # first node in chain (stall point)


@dataclass
class ChainData:
    nodes: List[NodeData] = field(default_factory=list)
    total_blame: float = 0.0
    depth: int = 0


@dataclass
class VendorResult:
    key: str
    label: str
    color: str
    kernel_name: str = ""
    stall_ratio: float = 0.0
    total_stall_cycles: float = 0.0
    num_chains: int = 0
    max_depth: int = 0
    chains: List[ChainData] = field(default_factory=list)
    occupancy_str: str = ""


# ---------------------------------------------------------------------------
# Opcode formatting helpers
# ---------------------------------------------------------------------------

_OPCODE_ABBREV = {
    "global_load_dwordx2": "global_load",
    "global_load_dwordx4": "global_load",
    "ds_read2_b64": "ds_read2",
    "ds_read_b64": "ds_read",
    "LDG.E.64.CONSTANT": "LDG.64.CONST",
    "LDG.E.64": "LDG.64",
    "IMAD.MOV.U32": "IMAD.MOV",
    "ISETP.GE.U32.AND.EX": "ISETP.GE",
    "ISETP.GE.U32.AND": "ISETP.GE",
    "SHF.L.U64.HI": "SHF.L.U64",
}


def _short_opcode(opcode: str, max_len: int = 18) -> str:
    """Shorten an ISA opcode for display."""
    # Strip waitcnt arguments: "s_waitcnt vmcnt(2) lgkmcnt(0)" -> "s_waitcnt"
    if opcode.startswith("s_waitcnt"):
        return "s_waitcnt"
    for long, short in _OPCODE_ABBREV.items():
        if long in opcode:
            opcode = opcode.replace(long, short)
    return opcode[:max_len]


def _fmt_cycles(c: float) -> str:
    if c >= 1e9:
        return f"{c / 1e9:.1f}B"
    if c >= 1e6:
        return f"{c / 1e6:.1f}M"
    if c >= 1e3:
        return f"{c / 1e3:.0f}K"
    return f"{c:.0f}"


# ---------------------------------------------------------------------------
# Run Leo analysis and extract chains
# ---------------------------------------------------------------------------

def _extract_chains(analysis_result, top_n: int = 10) -> List[ChainData]:
    """Extract chain data from a Leo AnalysisResult.

    Sorted by blame descending (highest blame first).  Display-time selection
    handles picking diverse depths.
    """
    chains: List[ChainData] = []
    blame_result = analysis_result.backslice_result

    # Sort by blame descending — display selection picks depth variety later
    sorted_chains = sorted(
        blame_result.blame_chains,
        key=lambda c: c.total_blame,
        reverse=True,
    )

    for chain in sorted_chains[:top_n]:
        cd = ChainData(total_blame=chain.total_blame, depth=chain.depth)
        for i, node in enumerate(chain.nodes):
            loc = analysis_result.get_source_location(node.pc, nearest=True)
            sf = Path(loc[0]).name if loc and loc[0] else ""
            sl = loc[1] if loc and len(loc) > 1 else 0
            cd.nodes.append(NodeData(
                opcode=node.opcode,
                source_file=sf,
                source_line=sl,
                is_stall=(i == 0),
            ))
        chains.append(cd)
    return chains


def analyze_vendor(measurements_dir: str, vendor_key: str) -> Optional[VendorResult]:
    """Run Leo analysis for one vendor and return extracted data."""
    from leo.program_analysis import analyze_program

    cfg = VENDORS[vendor_key]
    print(f"  Analyzing {cfg['label']} …")

    result = analyze_program(
        measurements_dir=measurements_dir,
        arch=cfg["arch"],
        top_n_kernels=1,
        run_full_analysis=True,
        skip_failed_kernels=True,
    )

    if not result.per_kernel_results:
        print(f"    No kernels found")
        return None

    k = result.per_kernel_results[0]
    if not k.analyzed:
        print(f"    Kernel analysis failed: {k.error}")
        return None

    blame = k.analysis_result.backslice_result
    vr = VendorResult(
        key=vendor_key,
        label=cfg["label"],
        color=cfg["color"],
        kernel_name=k.short_name,
        stall_ratio=k.stall_ratio,
        total_stall_cycles=k.stall_cycles,
        num_chains=len(blame.blame_chains),
        max_depth=max((c.depth for c in blame.blame_chains), default=0),
        chains=_extract_chains(k.analysis_result),
    )
    if k.occupancy:
        vr.occupancy_str = f"{k.occupancy.occupancy_pct:.0f}%"
    print(f"    {vr.stall_ratio * 100:.1f}% stall, {vr.num_chains} chains, max {vr.max_depth} hops")
    return vr


# ---------------------------------------------------------------------------
# Matplotlib plotting
# ---------------------------------------------------------------------------

def _draw_node(ax, x, y, text, w, h, *, facecolor, edgecolor, linewidth=0.6,
               fontsize=4.5, fontweight="normal", fontcolor="#333"):
    """Draw a rounded-rectangle node with centered text."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontfamily="monospace",
            fontweight=fontweight, color=fontcolor, zorder=4)


def _draw_edge(ax, x1, y1, x2, y2, color, lw=0.6):
    """Draw a downward arrow between two nodes."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        shrinkA=1, shrinkB=1),
        zorder=2,
    )


def _select_display_chains(chains: List[ChainData], n: int) -> List[ChainData]:
    """Select the top N chains by blame (most critical first).

    Chains are already sorted by total_blame descending from _extract_chains().
    """
    return chains[:n]


def plot_page(
    vendors: List[VendorResult],
    kernel_name: str = "",
    top_chains: int = 2,
    figsize: Tuple[float, float] = (14, 5.5),
):
    """Generate one page of the cross-vendor dependency chain figure.

    Returns the matplotlib Figure (caller is responsible for saving/closing).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    n = len(vendors)
    col_w = figsize[0] / n      # width per vendor column
    nw = 1.9                    # node width
    nh = 0.32                   # node height
    vs = 0.58                   # vertical step between hops
    y_top = 4.8
    y_chain = 3.6               # y where chains start

    # Max depth across displayed chains (not all chains — avoids empty rows)
    max_d = 0
    for v in vendors:
        for ch in v.chains[:top_chains]:
            max_d = max(max_d, ch.depth)
    max_d = min(max_d, 5)

    # --- Kernel title ---
    title = kernel_name or vendors[0].kernel_name
    ax.text(figsize[0] / 2, y_top + 0.5, f"{title} — same source code, three GPU architectures",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG_GRAY, ec="#cccccc", lw=0.5))

    # --- Depth guide lines ---
    for d in range(max_d + 1):
        yy = y_chain - d * vs
        ax.axhline(y=yy, color="#ebebeb", lw=0.3, ls=":", zorder=0)
        labels = {0: "root cause", 1: "hop 1", 2: "hop 2", 3: "hop 3", 4: "hop 4"}
        if d in labels:
            ax.text(-0.2, yy, labels[d], ha="right", va="center",
                    fontsize=3, color="#c0c0c0")

    # --- Draw each vendor column ---
    for vi, vd in enumerate(vendors):
        cx = col_w * (vi + 0.5)

        # Header
        ax.text(cx, y_top - 0.05, vd.label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=vd.color)
        stats = f"{vd.stall_ratio * 100:.1f}% stall · {vd.num_chains} chains · max {vd.max_depth} hops"
        ax.text(cx, y_top - 0.35, stats, ha="center", va="center",
                fontsize=5, color="#999")

        # Dashed line from title
        ax.plot([cx, cx], [y_top + 0.15, y_top + 0.05],
                color="#ddd", lw=0.5, ls="--")

        # Select chains to display: pick top-blame, then deepest different chain
        display_chains = _select_display_chains(vd.chains, top_chains)
        nc = len(display_chains)
        if nc == 0:
            continue
        offsets = [0.0] if nc == 1 else [
            -col_w * 0.22 + i * (col_w * 0.44 / (nc - 1)) for i in range(nc)
        ]

        for ci, chain in enumerate(display_chains):
            chain_x = cx + offsets[ci]
            # Reverse node order: chain.nodes[0]=stall, [-1]=root → display top-down
            nodes_display = list(reversed(chain.nodes))

            for ni, nd in enumerate(nodes_display):
                ny = y_chain - ni * vs
                op_short = _short_opcode(nd.opcode)
                line_str = f" :{nd.source_line}" if nd.source_line else ""
                label = f"{op_short}{line_str}"

                if nd.is_stall:
                    fc = STALL_COLOR + "15"
                    ec = STALL_COLOR
                    lw = 1.0
                    fw = "bold"
                    ftc = STALL_COLOR
                else:
                    fc = vd.color + "0c"
                    ec = vd.color + "80"
                    lw = 0.6
                    fw = "normal"
                    ftc = "#444"

                _draw_node(ax, chain_x, ny, label, nw, nh,
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           fontweight=fw, fontcolor=ftc)

                # Edge to next node
                if ni < len(nodes_display) - 1:
                    next_is_stall = nodes_display[ni + 1].is_stall
                    ec2 = STALL_COLOR + "80" if next_is_stall else vd.color + "50"
                    _draw_edge(ax, chain_x, ny - nh / 2,
                               cx + offsets[ci], y_chain - (ni + 1) * vs + nh / 2,
                               color=ec2)

            # Blame label — below the stall node to avoid overlap
            last_y = y_chain - (len(nodes_display) - 1) * vs
            pct = (chain.total_blame / vd.total_stall_cycles * 100
                   if vd.total_stall_cycles > 0 else 0)
            blame_str = _fmt_cycles(chain.total_blame)
            if ci == 0:
                ax.text(chain_x, last_y - nh / 2 - 0.06,
                        f"{blame_str} ({pct:.1f}%)",
                        ha="center", va="top", fontsize=3.8,
                        fontweight="bold", color=vd.color)
            else:
                ax.text(chain_x, last_y - nh / 2 - 0.06,
                        blame_str, ha="center", va="top",
                        fontsize=3.5, color=vd.color)

        # --- Bottom summary box ---
        summary_y = y_chain - (max_d + 1) * vs - 0.25
        # Auto-derive dominant root cause: weight by blame cycles
        from collections import defaultdict
        root_blame: Dict[str, float] = defaultdict(float)
        for ch in vd.chains[:10]:
            if ch.nodes:
                root_blame[ch.nodes[-1].opcode] += ch.total_blame
        if root_blame:
            dominant = _short_opcode(max(root_blame, key=root_blame.get))
        else:
            dominant = "unknown"

        summary = (
            f"Dominant root cause: {dominant}\n"
            f"Total stall cycles: {_fmt_cycles(vd.total_stall_cycles)}\n"
            f"Stall ratio: {vd.stall_ratio * 100:.1f}%"
        )
        ax.text(cx, summary_y, summary, ha="center", va="top",
                fontsize=4.2, color="#666", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc=vd.color + "08", ec=vd.color + "25", lw=0.5))

    # --- Axis limits ---
    ax.set_xlim(-0.5, figsize[0] + 0.5)
    y_bot = y_chain - (max_d + 2) * vs - 0.6
    ax.set_ylim(y_bot, y_top + 1.0)

    plt.tight_layout(pad=0.3)
    return fig


def plot_figure(
    vendors: List[VendorResult],
    output: str,
    kernel_name: str = "",
    top_chains: int = 2,
    figsize: Tuple[float, float] = (14, 5.5),
):
    """Generate and save a single-page cross-vendor figure."""
    import matplotlib.pyplot as plt

    fig = plot_page(vendors, kernel_name, top_chains, figsize)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _analyze_kernel(kernel: str, data_dir: Path, vendor_keys: List[str]) -> List[VendorResult]:
    """Run Leo analysis for one kernel across all vendors."""
    results: List[VendorResult] = []
    for vk in vendor_keys:
        cfg = VENDORS[vk]
        vdir = data_dir / kernel / cfg["dir"]
        meas_dirs = sorted(vdir.glob("hpctoolkit-*-measurements"))
        if not meas_dirs:
            print(f"  [{vk}] No measurements at {vdir}, skipping")
            continue
        vr = analyze_vendor(str(meas_dirs[0]), vk)
        if vr:
            results.append(vr)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot cross-vendor dependency chains from Leo analysis")
    parser.add_argument("--kernel", default=None,
                        help="Kernel name (subfolder in per-kernel data). "
                             "Omit or use --all for all kernels.")
    parser.add_argument("--all", action="store_true",
                        help="Generate one page per kernel in a multi-page PDF")
    parser.add_argument("--data-dir", default="tests/data/pc/per-kernel",
                        help="Base directory with per-kernel measurement data")
    parser.add_argument("--top-chains", type=int, default=2,
                        help="Number of chains to display per vendor")
    parser.add_argument("-o", "--output", default="cross_vendor_chains.pdf",
                        help="Output file path (PDF)")
    parser.add_argument("--figsize", default="14,5.5",
                        help="Figure size as 'width,height' in inches")
    parser.add_argument("--vendors", nargs="+", default=list(VENDORS.keys()),
                        choices=list(VENDORS.keys()),
                        help="Vendors to include")
    args = parser.parse_args()

    figsize = tuple(float(x) for x in args.figsize.split(","))
    data_dir = Path(args.data_dir)

    # Discover kernels
    if args.all or args.kernel is None:
        kernels = sorted(
            d.name for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not kernels:
            print(f"No kernel directories found in {data_dir}")
            sys.exit(1)
        print(f"Found {len(kernels)} kernels in {data_dir}")
    else:
        kernels = [args.kernel]

    if len(kernels) == 1:
        # Single kernel — same as before
        print(f"Kernel: {kernels[0]}")
        vendor_results = _analyze_kernel(kernels[0], data_dir, args.vendors)
        if not vendor_results:
            print("No vendor data available.")
            sys.exit(1)
        plot_figure(
            vendor_results,
            output=args.output,
            kernel_name=kernels[0],
            top_chains=args.top_chains,
            figsize=figsize,
        )
    else:
        # Multi-kernel — one page per kernel in a single PDF
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(args.output) as pdf:
            for ki, kernel in enumerate(kernels):
                print(f"\n[{ki + 1}/{len(kernels)}] {kernel}")
                vendor_results = _analyze_kernel(kernel, data_dir, args.vendors)
                if not vendor_results:
                    print(f"  Skipping {kernel}: no vendor data")
                    continue
                fig = plot_page(
                    vendor_results,
                    kernel_name=kernel,
                    top_chains=args.top_chains,
                    figsize=figsize,
                )
                pdf.savefig(fig, dpi=300, bbox_inches="tight")
                plt.close(fig)

        print(f"\nSaved {len(kernels)}-page PDF to {args.output}")


if __name__ == "__main__":
    main()
