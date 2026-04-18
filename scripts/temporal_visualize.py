#!/usr/bin/env python3
"""
Temporal PC Sampling Visualization (HPCToolkit)

Reads cube.bin produced by scripts/temporal_cube_builder.cpp and generates
summary plots. Optionally reads hwid.csv (histogram output) and plots
hardware ID distributions.
"""

import argparse
import csv
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

STALL_NAMES = [
    "NONE", "IFETCH", "IDEPEND", "MEM", "GMEM", "TMEM", "SYNC",
    "CMEM", "PIPE_BUSY", "MEM_THROTTLE", "OTHER", "SLEEP", "HIDDEN", "INVALID"
]

STALL_COLORS = {
    "NONE": "#2ecc71",
    "IFETCH": "#3498db",
    "IDEPEND": "#9b59b6",
    "MEM": "#e74c3c",
    "GMEM": "#c0392b",
    "TMEM": "#d35400",
    "SYNC": "#f39c12",
    "CMEM": "#16a085",
    "PIPE_BUSY": "#27ae60",
    "MEM_THROTTLE": "#8e44ad",
    "OTHER": "#7f8c8d",
    "SLEEP": "#34495e",
    "HIDDEN": "#1abc9c",
    "INVALID": "#95a5a6",
}

HWID_FIELDS = [
    "chiplet",
    "cu_or_wgp_id",
    "simd_id",
    "wave_id",
    "pipe_id",
    "workgroup_id",
    "shader_engine_id",
    "shader_array_id",
    "vm_id",
    "queue_id",
    "microengine_id",
]

KEY_HWID_FIELDS = [
    "chiplet",
    "cu_or_wgp_id",
    "simd_id",
    "shader_engine_id",
]

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_cube(filepath: Path):
    """Load cube data from binary file produced by C++ builder."""
    with filepath.open("rb") as f:
        cct_count, stall_count, time_bins = struct.unpack("<iii", f.read(12))
        min_ts, max_ts = struct.unpack("<qq", f.read(16))
        cct_ids = np.frombuffer(f.read(cct_count * 4), dtype=np.uint32)
        cube = np.frombuffer(
            f.read(cct_count * stall_count * time_bins * 8), dtype=np.uint64
        )
        cube = cube.reshape((cct_count, stall_count, time_bins))

    metadata = {
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "time_bins": time_bins,
    }
    return cube, cct_ids, metadata


def load_hwid_csv(path: Path):
    """Load hw_id histogram CSV into dict: field -> list[(value,count)]."""
    data = {field: [] for field in HWID_FIELDS}
    if not path.exists():
        return data

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row["field"]
            if field not in data:
                continue
            data[field].append((int(row["value"]), int(row["count"])))

    # Sort each field by value
    for field in data:
        data[field].sort(key=lambda x: x[0])
    return data


def load_hwid_time_csv(path: Path):
    """Load hw_id time series CSV into dict: field -> {value -> array of counts per time bin}."""
    if not path.exists():
        return {}, 0

    # First pass: determine max time bin
    max_time_bin = 0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["time_bin"])
            if t > max_time_bin:
                max_time_bin = t

    num_time_bins = max_time_bin + 1

    # Second pass: populate data
    data = {field: {} for field in HWID_FIELDS}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row["field"]
            if field not in data:
                continue
            value = int(row["value"])
            time_bin = int(row["time_bin"])
            count = int(row["count"])

            if value not in data[field]:
                data[field][value] = np.zeros(num_time_bins, dtype=np.uint64)
            data[field][value][time_bin] = count

    return data, num_time_bins


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _ensure_outdir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_stall_stack(cube, out_path: Path):
    stall_time = cube.sum(axis=0)  # (R, T)
    x = np.arange(stall_time.shape[1])

    active_mask = stall_time.sum(axis=1) > 0
    active_data = stall_time[active_mask]
    active_labels = [STALL_NAMES[i] for i in range(len(STALL_NAMES)) if active_mask[i]]
    active_colors = [STALL_COLORS.get(label, "#7f8c8d") for label in active_labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(x, active_data, labels=active_labels, colors=active_colors, alpha=0.85)
    ax.set_xlabel("Time Bin (GPU cycles)")
    ax.set_ylabel("Sample Count")
    ax.set_title("Stall Reason Composition Over Time")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hwid_histograms(hwid_data: dict, out_path: Path, top_n: int):
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, field in enumerate(HWID_FIELDS):
        ax = axes[i]
        items = hwid_data.get(field, [])
        if not items:
            ax.set_title(f"{field} (no data)")
            ax.axis("off")
            continue

        # Sort by count desc for top_n
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
        values = [v for v, _ in items_sorted]
        counts = [c for _, c in items_sorted]
        ax.bar(values, counts, color="#4c72b0")
        ax.set_title(field)
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    # Turn off unused axes if any
    for j in range(len(HWID_FIELDS), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hwid_temporal_normalized(
    hwid_time_data: dict,
    num_time_bins: int,
    out_path: Path,
    top_n: int,
):
    """Plot normalized temporal trends (percentage per time bin)."""
    key_fields = KEY_HWID_FIELDS

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    x = np.arange(num_time_bins)

    for i, field in enumerate(key_fields):
        ax = axes[i]
        field_data = hwid_time_data.get(field, {})
        if not field_data:
            ax.set_title(f"{field} (no data)")
            ax.axis("off")
            continue

        totals = [(v, arr.sum()) for v, arr in field_data.items()]
        totals_sorted = sorted(totals, key=lambda x: x[1], reverse=True)[:top_n]
        top_values = [v for v, _ in totals_sorted]

        total_per_bin = np.zeros(num_time_bins, dtype=np.float64)
        for arr in field_data.values():
            total_per_bin += arr
        total_per_bin[total_per_bin == 0] = 1.0

        stack_data = []
        labels = []
        for v in sorted(top_values):
            stack_data.append(field_data[v].astype(np.float64))
            labels.append(f"{field}={v}")

        if stack_data:
            stack_data = np.array(stack_data)
            other = total_per_bin - stack_data.sum(axis=0)
            if np.any(other > 0):
                stack_data = np.vstack([stack_data, other])
                labels.append("other")

            stack_pct = stack_data / total_per_bin * 100.0
            colors = [plt.cm.tab20(i % 20) for i in range(len(labels))]
            if labels[-1] == "other":
                colors[-1] = "#cccccc"
            ax.stackplot(x, stack_pct, labels=labels, colors=colors, alpha=0.85)
            ax.set_xlabel("Time Bin")
            ax.set_ylabel("Percent (%)")
            ax.set_title(f"{field} Distribution Over Time (Normalized)")
            ax.set_ylim(0, 100)
            ax.legend(loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hwid_temporal_heatmap(
    hwid_time_data: dict,
    num_time_bins: int,
    out_path: Path,
    top_n: int,
):
    """Plot hw_id value x time heatmaps for key fields."""
    key_fields = KEY_HWID_FIELDS
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, field in enumerate(key_fields):
        ax = axes[i]
        field_data = hwid_time_data.get(field, {})
        if not field_data:
            ax.set_title(f"{field} (no data)")
            ax.axis("off")
            continue

        totals = [(v, arr.sum()) for v, arr in field_data.items()]
        totals_sorted = sorted(totals, key=lambda x: x[1], reverse=True)[:top_n]
        top_values = [v for v, _ in totals_sorted]

        data = np.vstack([field_data[v] for v in top_values])

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="magma",
        )
        ax.set_title(f"{field} Heatmap")
        ax.set_xlabel("Time Bin")
        ax.set_ylabel("Value")
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_values, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize temporal cube and hw_id data")
    parser.add_argument("cube", type=Path, help="Path to cube.bin")
    parser.add_argument("--hwid", type=Path, default=None, help="Path to hwid.csv")
    parser.add_argument("--hwid-time", type=Path, default=None, help="Path to hwid_time.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--hwid-top", type=int, default=32, help="Top-N values per hw_id field")
    args = parser.parse_args()

    _ensure_outdir(args.out_dir)

    cube, cct_ids, _ = load_cube(args.cube)

    plot_stall_stack(cube, args.out_dir / "temporal_stall_stack.png")

    if args.hwid and args.hwid.exists():
        hwid_data = load_hwid_csv(args.hwid)
        plot_hwid_histograms(hwid_data, args.out_dir / "temporal_hwid_hist.png", args.hwid_top)

    if args.hwid_time and args.hwid_time.exists():
        hwid_time_data, num_time_bins = load_hwid_time_csv(args.hwid_time)
        plot_hwid_temporal_normalized(
            hwid_time_data, num_time_bins,
            args.out_dir / "temporal_hwid_time_pct.png", args.hwid_top
        )
        plot_hwid_temporal_heatmap(
            hwid_time_data, num_time_bins,
            args.out_dir / "temporal_hwid_time_heat.png", args.hwid_top
        )

    print(f"Plots saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
