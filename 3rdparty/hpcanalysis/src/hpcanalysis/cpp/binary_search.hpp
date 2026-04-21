// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "parsing.hpp"

#include <cstdint>
#include <optional>
#include <tuple>

template <typename T1, typename T2>
std::optional<std::tuple<uint64_t, T1, T2>>
cct_metrics_binary_search(const char *data, uint64_t low, uint64_t high,
                          T1 target) {
  if (high < low) {
    return std::nullopt;
  }

  uint64_t mid = (low + high) / 2;
  const auto tuple_size = sizeof(T1) + sizeof(T2);
  const auto base_offset = tuple_size * mid;

  T1 id = safe_unpack<T1>(data, base_offset);
  T2 idValue = safe_unpack<T2>(data, base_offset + sizeof(T1));

  if (id == target) {
    return std::tuple<uint64_t, T1, T2>{mid, id, idValue};
  } else if (id > target) {
    return cct_metrics_binary_search<T1, T2>(data, low, mid - 1, target);
  } else {
    return cct_metrics_binary_search<T1, T2>(data, mid + 1, high, target);
  }
}

uint64_t trace_binary_search(const char *data, uint64_t num_samples,
                             uint64_t target);
