// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

template <typename T> T safe_unpack(const char *data_ptr, int offset) {
  T value;
  std::memcpy(&value, data_ptr + offset, sizeof(T));
  return value;
}
