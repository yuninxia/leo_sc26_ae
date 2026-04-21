// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_search.hpp"
#include "data_read.hpp"

uint64_t trace_binary_search(const char *data, uint64_t num_samples,
                             uint64_t target) {
  uint64_t low = 0;
  uint64_t high = num_samples;

  while (low < high) {
    uint64_t mid = (low + high) / 2;
    uint64_t timestamp = safe_unpack<uint64_t>(data, mid * EVENT_TUPLE_SIZE);

    if (timestamp < target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}
