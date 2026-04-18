#!/usr/bin/env python3
"""Generate multiple SDC chart styles for paper. Pick your favorite."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from bkm5c85lw output (typed baseline)
KERNELS = [
    "MASS3DEA", "LTIMES_NV", "LTIMES", "3MM", "2MM", "GEMM",
    "PRESSURE", "ENERGY", "FIR", "ZONAL_3D", "VOL3D", "DEL_DOT",
    "DIFF3DPA", "CONV3DPA", "MASS3DPA",
    "miniBUDE", "XSBench", "LULESH", "QuickSilv", "llama.cpp", "Kripke",
]

# (before, after) per vendor
NV = [
    (29.6,69.3),(64.4,76.6),(57.4,64.3),(56.6,78.8),(52.5,81.3),(52.2,82.4),
    (74.2,84.8),(72.3,82.9),(68.9,84.4),(71.0,93.8),(63.0,80.5),(67.6,85.3),
    (73.1,86.2),(68.4,78.0),(71.4,87.7),
    (31.7,68.0),(57.8,72.1),(67.2,78.9),(61.5,74.9),(67.4,83.8),(52.6,64.2),
]
AMD = [
    (49.6,96.9),(68.8,87.5),(82.6,88.9),(75.0,87.5),(70.6,94.1),(75.0,88.9),
    (81.8,86.7),(91.9,81.5),(100.,85.7),(80.0,79.3),(83.3,88.6),(83.0,84.3),
    (85.7,85.1),(81.8,94.0),(85.3,94.0),
    (51.0,97.0),(75.4,89.3),(89.5,87.0),(71.2,92.3),(85.3,91.6),(78.6,85.7),
]
INTEL = [
    (82.3,82.3),(71.1,71.3),(72.9,72.9),(79.7,79.7),(77.7,77.7),(60.6,60.6),
    (86.1,86.1),(75.6,77.6),(88.6,88.6),(88.1,88.1),(74.5,74.5),(73.0,74.6),
    (84.4,84.6),(85.7,86.2),(81.5,82.7),
    (70.8,73.1),(71.7,72.0),(77.9,78.7),(None,None),(None,None),(None,None),
]

N = len(KERNELS)
RAJA_N = 15

NVIDIA_GREEN = '#76B900'
AMD_RED = '#ED1C24'
INTEL_BLUE = '#0071C5'

import os
OUT = os.environ.get("LEO_SDC_OUT", "sdc_plots")
os.makedirs(OUT, exist_ok=True)


def style1_grouped_bars():
    """GPA-style: grouped bars, before/after per vendor, all kernels."""
    fig, ax = plt.subplots(figsize=(18, 5))
    x = np.arange(N)
    w = 0.13

    for i, (data, color, label) in enumerate([
        (NV, NVIDIA_GREEN, "NVIDIA"),
        (AMD, AMD_RED, "AMD"),
        (INTEL, INTEL_BLUE, "Intel"),
    ]):
        befores = [d[0] if d[0] is not None else 0 for d in data]
        afters = [d[1] if d[1] is not None else 0 for d in data]
        offset = (i - 1) * 2 * w
        ax.bar(x + offset - w/2, befores, w, color=color, alpha=0.3, edgecolor=color)
        ax.bar(x + offset + w/2, afters, w, color=color, alpha=0.9, edgecolor=color,
               label=f"{label} (after)" if i == 0 else f"{label}")

    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axvline(x=14.5, color='black', linestyle=':', alpha=0.3)
    ax.set_ylabel('Single Dependency Coverage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(KERNELS, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, fontsize=8, loc='upper left')
    ax.set_title('Style 1: Grouped Bars (GPA-style, before=faded, after=solid)')
    plt.tight_layout()
    fig.savefig(f'{OUT}/style1_grouped.pdf', dpi=150)
    fig.savefig(f'{OUT}/style1_grouped.png', dpi=150)
    plt.close()


def style2_after_only():
    """After-only bars, 3 vendors side by side. Clean and simple."""
    fig, ax = plt.subplots(figsize=(16, 4))
    x = np.arange(N)
    w = 0.25

    for i, (data, color, label) in enumerate([
        (NV, NVIDIA_GREEN, "NVIDIA H100"),
        (AMD, AMD_RED, "AMD MI300A"),
        (INTEL, INTEL_BLUE, "Intel PVC"),
    ]):
        afters = [d[1] if d[1] is not None else 0 for d in data]
        mask = [d[1] is not None for d in data]
        bars = ax.bar(x + (i-1)*w, afters, w, color=color, alpha=0.85, label=label)
        # Gray out N/A bars
        for j, b in enumerate(bars):
            if not mask[j]:
                b.set_alpha(0)

    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=14.5, color='black', linestyle=':', alpha=0.3)
    ax.text(7, 82, '80% threshold', fontsize=7, color='gray', ha='center')
    ax.text(7, 102, 'RAJAPerf Kernels', fontsize=8, ha='center', style='italic')
    ax.text(18, 102, 'HPC Apps', fontsize=8, ha='center', style='italic')
    ax.set_ylabel('SDC after analysis (%)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(KERNELS, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 108)
    ax.legend(ncol=3, fontsize=8, loc='lower right')
    ax.set_title('Style 2: After-Only (clean, no before)')
    plt.tight_layout()
    fig.savefig(f'{OUT}/style2_after_only.pdf', dpi=150)
    fig.savefig(f'{OUT}/style2_after_only.png', dpi=150)
    plt.close()


def style3_stacked_delta():
    """Stacked: base + delta, showing improvement."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for ax, (data, color, title) in zip(axes, [
        (NV, NVIDIA_GREEN, "NVIDIA H100"),
        (AMD, AMD_RED, "AMD MI300A"),
        (INTEL, INTEL_BLUE, "Intel PVC"),
    ]):
        befores = [d[0] if d[0] is not None else 0 for d in data]
        afters = [d[1] if d[1] is not None else 0 for d in data]
        deltas = [a - b for a, b in zip(afters, befores)]
        mask = [d[0] is not None for d in data]

        x = np.arange(N)
        ax.bar(x, befores, color=color, alpha=0.3, label='Before')
        bars = ax.bar(x, [max(0, d) for d in deltas], bottom=befores, color=color, alpha=0.9, label='Improvement')
        # Negative deltas (AMD ENERGY, FIR, ZONAL)
        for j in range(N):
            if deltas[j] < 0:
                ax.bar(j, abs(deltas[j]), bottom=afters[j], color='orange', alpha=0.7)
            if not mask[j]:
                bars[j].set_alpha(0)

        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=14.5, color='black', linestyle=':', alpha=0.3)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(KERNELS, rotation=90, fontsize=6)
        ax.set_ylim(0, 105)
        if ax == axes[0]:
            ax.set_ylabel('SDC (%)')

    axes[0].legend(fontsize=7)
    fig.suptitle('Style 3: Stacked (before + improvement delta)', fontsize=11)
    plt.tight_layout()
    fig.savefig(f'{OUT}/style3_stacked.pdf', dpi=150)
    fig.savefig(f'{OUT}/style3_stacked.png', dpi=150)
    plt.close()


def style4_heatmap():
    """Heatmap: kernels × vendors, color = SDC after."""
    vendors = ['NVIDIA H100', 'AMD MI300A', 'Intel PVC']
    matrix = []
    for data in [NV, AMD, INTEL]:
        row = [d[1] if d[1] is not None else np.nan for d in data]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')
    ax.set_xticks(range(N))
    ax.set_xticklabels(KERNELS, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(vendors, fontsize=9)

    # Annotate cells
    for i in range(3):
        for j in range(N):
            val = matrix[i][j]
            if not np.isnan(val):
                color = 'white' if val < 70 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6, color=color)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=6, color='gray')

    ax.axvline(x=14.5, color='white', linewidth=2)
    plt.colorbar(im, ax=ax, label='SDC (%)', shrink=0.8)
    ax.set_title('Style 4: Heatmap (SDC after analysis)')
    plt.tight_layout()
    fig.savefig(f'{OUT}/style4_heatmap.pdf', dpi=150)
    fig.savefig(f'{OUT}/style4_heatmap.png', dpi=150)
    plt.close()


def style5_compact_table():
    """Compact: only after values, sorted, with 80% line. Paper-friendly single column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(N)
    w = 0.25

    for i, (data, color, label) in enumerate([
        (NV, NVIDIA_GREEN, "NVIDIA"),
        (AMD, AMD_RED, "AMD"),
        (INTEL, INTEL_BLUE, "Intel"),
    ]):
        afters = [d[1] if d[1] is not None else 0 for d in data]
        mask = [d[1] is not None for d in data]
        bars = ax.barh(x + (i-1)*w, afters, w, color=color, alpha=0.85, label=label)
        for j, b in enumerate(bars):
            if not mask[j]:
                b.set_alpha(0)

    ax.axvline(x=80, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=14.5, color='black', linestyle=':', alpha=0.3)
    ax.set_yticks(x)
    ax.set_yticklabels(KERNELS, fontsize=7)
    ax.set_xlabel('SDC (%)')
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8, loc='lower right')
    ax.invert_yaxis()
    ax.set_title('Style 5: Horizontal bars (compact, single-column)')
    plt.tight_layout()
    fig.savefig(f'{OUT}/style5_horizontal.pdf', dpi=150)
    fig.savefig(f'{OUT}/style5_horizontal.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    style1_grouped_bars()
    style2_after_only()
    style3_stacked_delta()
    style4_heatmap()
    style5_compact_table()
    print(f"Generated 5 styles in {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")
