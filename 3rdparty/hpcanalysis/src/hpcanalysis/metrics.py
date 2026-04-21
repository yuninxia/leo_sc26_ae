# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

TIME_METRICS = [
    "cputime",
    "realtime",
    "cycles",
]

GPU_METRICS = {
    "gpuop": "GPU all operations",
    "gker": "GPU kernel execution",
    "gmem": "GPU memory allocation/deallocation",
    "gmset": "GPU memory set",
    "gxcopy": "GPU explicit data copy",
    "gicopy": "GPU implicit data copy",
    "gsync": "GPU synchronization",
    "gins": "GPU instruction execution and stall metrics",
    # AMD GPU PC sampling metrics
    "gcycles": "AMD GPU cycles (issue, stall, wave, thread metrics)",
    "gpipe": "AMD GPU pipeline (issue and stall cycles)",
}
