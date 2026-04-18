#!/usr/bin/env python3
"""Generate paper-quality SDC stacked chart as PGF for SC26."""

import os
import matplotlib
matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": "",
    "font.size": 6,
    "axes.labelsize": 5,
    "axes.titlesize": 6,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4.5,
    "legend.fontsize": 3.5,
})
import matplotlib.pyplot as plt
import numpy as np

# Data from bkm5c85lw output (typed baseline, before sync tracing → after pruning)
KERNELS_RAJA = [
    "MASS3DEA", "LTIMES\\_NV", "LTIMES", "3MM", "2MM", "GEMM",
    "PRESSURE", "ENERGY", "FIR", "ZONAL\\_3D", "VOL3D", "DEL\\_DOT",
    "DIFF3DPA", "CONV3DPA", "MASS3DPA",
]
KERNELS_HPC = ["miniBUDE", "XSBench", "LULESH", "QuickSilv", "llama.cpp", "Kripke"]

# (before, after) — NVIDIA
NV_R = [(29.6,69.3),(64.4,76.6),(57.4,64.3),(56.6,78.8),(52.5,81.3),(52.2,82.4),
        (74.2,84.8),(72.3,82.9),(68.9,84.4),(71.0,93.8),(63.0,80.5),(67.6,85.3),
        (73.1,86.2),(68.4,78.0),(71.4,87.7)]
NV_H = [(31.7,68.0),(57.8,72.1),(67.2,78.9),(61.5,74.9),(67.4,83.8),(52.6,64.2)]

# AMD
AMD_R = [(49.6,96.9),(68.8,87.5),(82.6,88.9),(75.0,87.5),(70.6,94.1),(75.0,88.9),
         (81.8,86.7),(91.9,81.5),(100.,85.7),(80.0,79.3),(83.3,88.6),(83.0,84.3),
         (85.7,85.1),(81.8,94.0),(85.3,94.0)]
AMD_H = [(51.0,97.0),(75.4,89.3),(89.5,87.0),(71.2,92.3),(85.3,91.6),(78.6,85.7)]

# Intel
INT_R = [(82.3,82.3),(71.1,71.3),(72.9,72.9),(79.7,79.7),(77.7,77.7),(60.6,60.6),
         (86.1,86.1),(75.6,77.6),(88.6,88.6),(88.1,88.1),(74.5,74.5),(73.0,74.6),
         (84.4,84.6),(85.7,86.2),(81.5,82.7)]
INT_H = [(70.8,73.1),(71.7,72.0),(77.9,78.7),(None,None),(None,None),(None,None)]

NVIDIA_GREEN = '#76B900'
AMD_RED = '#ED1C24'
INTEL_BLUE = '#0071C5'

OUT = os.environ.get("LEO_FIG_OUT", "figures")


def plot_vendor(ax, kernels, data, color, title, show_ylabel=True):
    N = len(kernels)
    x = np.arange(N)
    befores = [d[0] if d[0] is not None else 0 for d in data]
    afters = [d[1] if d[1] is not None else 0 for d in data]
    deltas = [a - b for a, b in zip(afters, befores)]

    # Before (faded)
    ax.bar(x, befores, color=color, alpha=0.25, edgecolor='none', label='Before')
    # Positive delta (solid)
    pos_deltas = [max(0, d) for d in deltas]
    ax.bar(x, pos_deltas, bottom=befores, color=color, alpha=0.85, edgecolor='none', label='Improvement')
    # Negative delta (orange)
    for j in range(N):
        if deltas[j] < 0:
            ax.bar(j, abs(deltas[j]), bottom=afters[j], color='#FF8C00', alpha=0.7, edgecolor='none')

    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.6, linewidth=0.6)
    ax.set_title(r'\textbf{' + title + '}', fontsize=6, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=50, fontsize=3.5, ha='right', va='top', rotation_mode='anchor')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=4.5, pad=0.5)
    ax.tick_params(axis='x', pad=0.5)
    if show_ylabel:
        ax.set_ylabel(r'SDC (\%)', fontsize=5, labelpad=1)


fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.15), sharey=True)

# Combine RAJAPerf + HPC for each vendor
all_kernels = KERNELS_RAJA + KERNELS_HPC

plot_vendor(axes[0], all_kernels, NV_R + NV_H, NVIDIA_GREEN, 'NVIDIA GH200', show_ylabel=True)
plot_vendor(axes[1], all_kernels, AMD_R + AMD_H, AMD_RED, 'AMD MI300A', show_ylabel=False)
plot_vendor(axes[2], all_kernels, INT_R + INT_H, INTEL_BLUE, 'Intel PVC', show_ylabel=False)

# Add legend only to first subplot
axes[0].legend(fontsize=3.5, loc='lower left', framealpha=0.8, handlelength=0.8, handletextpad=0.2, borderpad=0.2)

plt.tight_layout(pad=0.1, w_pad=0.2)
fig.savefig(f'{OUT}/sdc_coverage.pgf', bbox_inches='tight', pad_inches=0)
fig.savefig(f'{OUT}/sdc_coverage.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
print(f"Saved to {OUT}/sdc_coverage.pgf and .pdf")
plt.close()
