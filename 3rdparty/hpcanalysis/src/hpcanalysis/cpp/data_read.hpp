// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <omp.h>
#include <pybind11/stl.h>
#include <regex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace py = pybind11;

const size_t FILE_HEADER_OFFSET = 16;
const size_t CCT_TUPLE_SIZE = sizeof(uint32_t) + sizeof(uint64_t);
const size_t METRIC_TUPLE_SIZE = sizeof(uint16_t) + sizeof(double);
const size_t EVENT_TUPLE_SIZE = sizeof(uint64_t) + sizeof(uint32_t);

const std::map<std::string, std::string> METRIC_SCOPE_MAPPING = {
    {"execution", "i"}, {"function", "e"}, {"point", "p"}, {"lex_aware", "c"}};

const std::map<int, std::string> SUMMARY_METRIC_MAPPING = {
    {0, "sum"}, {1, "min"}, {2, "max"}};

using TraceProfileMap =
    std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>>;

struct ProfileRow {
  uint32_t profile_id;
  uint32_t cct_id;
  uint16_t metric_id;
  double value;
};

struct TraceRow {
  uint32_t profile_id;
  uint32_t cct_id;
  uint64_t start_timestamp;
  uint64_t end_timestamp;
};

class DataRead {
public:
  DataRead(const std::string &dir_path);

  py::list read_metric_descriptions();
  py::list read_profile_descriptions();
  py::list read_profile_slices(const std::vector<uint32_t> &profile_indices,
                               const std::vector<uint32_t> &cct_indices,
                               const std::vector<uint16_t> &metric_indices,
                               const size_t thread_count);
  py::list read_trace_slices(const TraceProfileMap &profile_indices,
                             const size_t thread_count);

private:
  std::string _meta_file;
  std::string _profile_file;
  std::string _cct_file;
  std::string _trace_file;
};
