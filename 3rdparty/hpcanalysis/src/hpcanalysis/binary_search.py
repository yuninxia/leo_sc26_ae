# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

import struct
from typing import Tuple

from hpcanalysis.parsing import safe_unpack


def cct_metrics_binary_search(
    format: str, data: bytes, low: int, high: int, target: int
) -> int | Tuple[int, int, int | float]:

    if high >= low:

        mid = (low + high) // 2
        (id, idValue) = safe_unpack(format, data, 0, mid)

        if id == target:
            return (mid, id, idValue)

        elif id > target:
            return cct_metrics_binary_search(format, data, low, mid - 1, target)

        else:
            return cct_metrics_binary_search(format, data, mid + 1, high, target)

    else:
        return -1


def trace_binary_search(data: bytes, low: int, high: int, target: int) -> int:
    sample_format = "<QL"
    sample_size = struct.calcsize(sample_format)

    while low < high:
        mid = (low + high) // 2

        (timestamp,) = safe_unpack("<Q", data, mid * sample_size)
        if timestamp < target:
            low = mid + 1
        else:
            high = mid

    return low
