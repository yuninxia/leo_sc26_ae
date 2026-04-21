// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#include "data_read.hpp"
#include "binary_search.hpp"
#include "parsing.hpp"

std::string normalize_string(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  s.erase(std::remove_if(s.begin(), s.end(),
                         [](char c) {
                           return c == '\0' ||
                                  std::iscntrl(static_cast<unsigned char>(c));
                         }),
          s.end());

  size_t first = s.find_first_not_of(' ');
  if (std::string::npos == first) {
    return "";
  }

  size_t last = s.find_last_not_of(' ');
  return s.substr(first, (last - first + 1));
}

DataRead::DataRead(const std::string &dir_path) {
  _meta_file = dir_path + "/meta.db";
  _profile_file = dir_path + "/profile.db";
  _cct_file = dir_path + "/cct.db";
  _trace_file = dir_path + "/trace.db";
}

py::list DataRead::read_metric_descriptions() {
  int fd = open(_meta_file.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Failed to open meta file: " + _meta_file);
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    throw std::runtime_error("Failed to stat meta file: " + _meta_file);
  }
  size_t file_size = st.st_size;

  const char *mapped_data_ptr =
      (const char *)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped_data_ptr == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to mmap meta file: " + _meta_file);
  }

  py::list rows;

  uint64_t szMetrics = safe_unpack<uint64_t>(
      mapped_data_ptr, FILE_HEADER_OFFSET + 4 * sizeof(uint64_t));
  uint64_t pMetrics = safe_unpack<uint64_t>(
      mapped_data_ptr,
      FILE_HEADER_OFFSET + 4 * sizeof(uint64_t) + sizeof(uint64_t));

  const char *metrics_db = mapped_data_ptr + pMetrics;

  uint64_t pMetrics_array = safe_unpack<uint64_t>(metrics_db, 0);
  uint32_t nMetrics = safe_unpack<uint32_t>(metrics_db, sizeof(uint64_t));
  uint8_t szMetric =
      safe_unpack<uint8_t>(metrics_db, sizeof(uint64_t) + sizeof(uint32_t));
  uint8_t szScopeInst = safe_unpack<uint8_t>(
      metrics_db, sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint8_t));
  uint8_t szSummary = safe_unpack<uint8_t>(
      metrics_db, sizeof(uint64_t) + sizeof(uint32_t) + 2 * sizeof(uint8_t));

  for (uint32_t i = 0; i < nMetrics; ++i) {
    const auto base_offset = pMetrics_array - pMetrics + i * szMetric;

    uint64_t pName = safe_unpack<uint64_t>(metrics_db, base_offset);
    uint64_t pScopeInsts =
        safe_unpack<uint64_t>(metrics_db, base_offset + sizeof(uint64_t));
    uint64_t pSummaries =
        safe_unpack<uint64_t>(metrics_db, base_offset + 2 * sizeof(uint64_t));
    uint16_t nScopeInsts =
        safe_unpack<uint16_t>(metrics_db, base_offset + 3 * sizeof(uint64_t));
    uint16_t nSummaries = safe_unpack<uint16_t>(
        metrics_db, base_offset + 3 * sizeof(uint64_t) + sizeof(uint16_t));

    std::string name = std::string(mapped_data_ptr + pName);
    std::string unit = "";

    if (name.length() > 0 && name.back() == ')') {
      auto open_paren = name.find_last_of('(');
      if (open_paren != std::string::npos) {
        unit = name.substr(open_paren + 1, name.length() - open_paren - 2);
        name = name.substr(0, open_paren);
      }
    }

    name = normalize_string(name);
    unit = normalize_string(unit);

    for (uint16_t j = 0; j < nScopeInsts; ++j) {
      const auto base_offset = pScopeInsts - pMetrics + j * szScopeInst;

      uint64_t pScope = safe_unpack<uint64_t>(metrics_db, base_offset);
      uint16_t propMetricId =
          safe_unpack<uint16_t>(metrics_db, base_offset + sizeof(uint64_t));

      const auto pScope_offset_rel = pScope - pMetrics;
      uint64_t pScopeName =
          safe_unpack<uint64_t>(metrics_db, pScope_offset_rel);

      std::string raw_scope_name =
          normalize_string(std::string(mapped_data_ptr + pScopeName));
      std::string scope_name = "unknown_scope";
      if (METRIC_SCOPE_MAPPING.count(raw_scope_name)) {
        scope_name = METRIC_SCOPE_MAPPING.at(raw_scope_name);
      }

      py::dict metric_dict;
      metric_dict["id"] = propMetricId;
      metric_dict["name"] = name;
      metric_dict["aggregation"] = "prop";
      metric_dict["scope"] = scope_name;
      metric_dict["unit"] = unit;
      rows.append(metric_dict);
    }

    for (uint16_t j = 0; j < nSummaries; ++j) {
      const auto base_offset = pSummaries - pMetrics + j * szSummary;

      uint64_t pScope = safe_unpack<uint64_t>(metrics_db, base_offset);
      uint8_t combine =
          safe_unpack<uint8_t>(metrics_db, base_offset + 2 * sizeof(uint64_t));
      uint16_t statMetricId = safe_unpack<uint16_t>(
          metrics_db, base_offset + 2 * sizeof(uint64_t) + 2 * sizeof(uint8_t));

      const auto pScope_offset_rel = pScope - pMetrics;
      uint64_t pScopeName =
          safe_unpack<uint64_t>(metrics_db, pScope_offset_rel);

      std::string raw_scope_name =
          normalize_string(std::string(mapped_data_ptr + pScopeName));
      std::string scope_name = "unknown_scope";
      if (METRIC_SCOPE_MAPPING.count(raw_scope_name)) {
        scope_name = METRIC_SCOPE_MAPPING.at(raw_scope_name);
      }

      std::string summary_name = "unknown_summary";
      if (SUMMARY_METRIC_MAPPING.count(combine)) {
        summary_name = SUMMARY_METRIC_MAPPING.at(combine);
      }

      py::dict metric_dict;
      metric_dict["id"] = statMetricId;
      metric_dict["name"] = name;
      metric_dict["aggregation"] = summary_name;
      metric_dict["scope"] = scope_name;
      metric_dict["unit"] = unit;
      rows.append(metric_dict);
    }
  }

  munmap((void *)mapped_data_ptr, file_size);
  close(fd);

  return rows;
}

py::list DataRead::read_profile_descriptions() {
  int profile_fd = open(_profile_file.c_str(), O_RDONLY);
  if (profile_fd == -1) {
    throw std::runtime_error("Failed to open profile file: " + _profile_file);
  }

  struct stat st;
  if (fstat(profile_fd, &st) == -1) {
    close(profile_fd);
    throw std::runtime_error("Failed to stat profile file: " + _profile_file);
  }
  size_t profile_file_size = st.st_size;

  const char *profile_db = (const char *)mmap(
      NULL, profile_file_size, PROT_READ, MAP_SHARED, profile_fd, 0);
  if (profile_db == MAP_FAILED) {
    close(profile_fd);
    throw std::runtime_error("Failed to mmap profile file: " + _profile_file);
  }

  int meta_fd = open(_meta_file.c_str(), O_RDONLY);
  if (meta_fd == -1) {
    throw std::runtime_error("Failed to open meta file: " + _meta_file);
  }

  if (fstat(meta_fd, &st) == -1) {
    close(meta_fd);
    throw std::runtime_error("Failed to stat meta file: " + _meta_file);
  }
  size_t meta_file_size = st.st_size;

  const char *meta_db = (const char *)mmap(NULL, meta_file_size, PROT_READ,
                                           MAP_SHARED, meta_fd, 0);
  if (meta_db == MAP_FAILED) {
    close(meta_fd);
    throw std::runtime_error("Failed to mmap meta file: " + _meta_file);
  }

  int trace_fd = open(_trace_file.c_str(), O_RDONLY);
  if (trace_fd == -1) {
    throw std::runtime_error("Failed to open trace file: " + _trace_file);
  }

  if (fstat(trace_fd, &st) == -1) {
    close(trace_fd);
    throw std::runtime_error("Failed to stat trace file: " + _trace_file);
  }
  size_t trace_file_size = st.st_size;

  const char *trace_db = (const char *)mmap(NULL, trace_file_size, PROT_READ,
                                            MAP_SHARED, trace_fd, 0);
  if (trace_db == MAP_FAILED) {
    close(trace_fd);
    throw std::runtime_error("Failed to mmap trace file: " + _profile_file);
  }

  uint64_t pCtxTraces = 0;
  uint64_t pTraces = 0;
  uint8_t szTrace = 0;

  if (trace_db != nullptr && trace_db != MAP_FAILED) {
    uint64_t szCtxTraces = safe_unpack<uint64_t>(trace_db, FILE_HEADER_OFFSET);
    pCtxTraces =
        safe_unpack<uint64_t>(trace_db, FILE_HEADER_OFFSET + sizeof(uint64_t));
    pTraces = safe_unpack<uint64_t>(trace_db, pCtxTraces);
    szTrace = safe_unpack<uint8_t>(trace_db, pCtxTraces + sizeof(uint64_t) +
                                                 sizeof(uint32_t));
  }

  uint64_t szIdNames =
      safe_unpack<uint64_t>(meta_db, FILE_HEADER_OFFSET + 2 * sizeof(uint64_t));
  uint64_t pIdNames =
      safe_unpack<uint64_t>(meta_db, FILE_HEADER_OFFSET + 3 * sizeof(uint64_t));

  uint64_t ppNames = safe_unpack<uint64_t>(meta_db, pIdNames);
  uint8_t nKinds = safe_unpack<uint8_t>(meta_db, pIdNames + sizeof(uint64_t));

  std::map<uint8_t, std::string> identifiers;

  for (uint8_t i = 0; i < nKinds; ++i) {
    uint64_t pName = safe_unpack<uint64_t>(
        meta_db, pIdNames + ppNames - pIdNames + i * sizeof(uint64_t));
    identifiers[i] = normalize_string(std::string(meta_db + pName));
  }

  uint64_t szProfileInfos =
      safe_unpack<uint64_t>(profile_db, FILE_HEADER_OFFSET);
  uint64_t pProfileInfos =
      safe_unpack<uint64_t>(profile_db, FILE_HEADER_OFFSET + sizeof(uint64_t));
  uint64_t szIdTuples = safe_unpack<uint64_t>(
      profile_db, FILE_HEADER_OFFSET + 2 * sizeof(uint64_t));
  uint64_t pIdTuples = safe_unpack<uint64_t>(
      profile_db, FILE_HEADER_OFFSET + 3 * sizeof(uint64_t));

  uint64_t pProfiles = safe_unpack<uint64_t>(profile_db, pProfileInfos);
  uint8_t szProfile = safe_unpack<uint8_t>(
      profile_db, pProfileInfos + sizeof(uint64_t) + sizeof(uint32_t));

  py::list rows;

  uint64_t current_offset = 0;
  int current_index = 1;

  while (current_offset < szIdTuples) {
    py::dict row;
    row["id"] = current_index;

    const auto base_offset =
        pProfiles - pProfileInfos + current_index * szProfile;

    uint64_t nValues =
        safe_unpack<uint64_t>(profile_db, pProfileInfos + base_offset);
    uint32_t nCtxs = safe_unpack<uint32_t>(
        profile_db, pProfileInfos + base_offset + 2 * sizeof(uint64_t));

    row["ctx_samples"] = nCtxs;
    row["metric_samples"] = nValues;

    if (trace_db != nullptr && trace_db != MAP_FAILED) {
      const auto trace_base_offset =
          pTraces - pCtxTraces + (current_index - 1) * szTrace;

      uint64_t pStart = safe_unpack<uint64_t>(
          trace_db + pCtxTraces, trace_base_offset + 2 * sizeof(uint32_t));
      uint64_t pEnd = safe_unpack<uint64_t>(
          trace_db + pCtxTraces,
          trace_base_offset + 2 * sizeof(uint32_t) + sizeof(uint64_t));

      row["trace_samples"] = (pEnd - pStart) / EVENT_TUPLE_SIZE;
    }

    uint16_t nIds =
        safe_unpack<uint16_t>(profile_db, pIdTuples + current_offset);
    current_offset += 8;

    for (uint16_t i = 0; i < nIds; ++i) {
      uint8_t kind =
          safe_unpack<uint8_t>(profile_db, pIdTuples + current_offset);
      uint32_t logicalId = safe_unpack<uint32_t>(
          profile_db,
          pIdTuples + current_offset + 2 * sizeof(uint8_t) + sizeof(uint16_t));
      uint64_t physicalId = safe_unpack<uint64_t>(
          profile_db, pIdTuples + current_offset + 2 * sizeof(uint8_t) +
                          sizeof(uint16_t) + sizeof(uint32_t));

      std::string id_name = identifiers.at(kind);
      uint64_t id_value;

      if (kind == 1 || kind == 7) {
        id_value = physicalId;
      } else {
        id_value = logicalId;
      }

      row[id_name.c_str()] = id_value;
      current_offset += 16;
    }

    current_index += 1;
    rows.append(row);
  }

  munmap((void *)profile_db, profile_file_size);
  close(profile_fd);
  munmap((void *)meta_db, meta_file_size);
  close(meta_fd);
  munmap((void *)trace_db, trace_file_size);
  close(trace_fd);

  return rows;
}

void extract_metrics(uint32_t profile_index, uint32_t cct_id,
                     std::vector<uint16_t> metric_indices,
                     const char *values_sub_db, uint64_t startIndex,
                     uint64_t end_index, std::vector<ProfileRow> &rows) {
  if (!metric_indices.empty()) {
    while (!metric_indices.empty()) {
      uint16_t metric_id = metric_indices.back();
      metric_indices.pop_back();

      auto result_2 = cct_metrics_binary_search<uint16_t, double>(
          values_sub_db, startIndex, end_index, metric_id);

      if (result_2.has_value()) {
        uint64_t mid_2 = std::get<0>(result_2.value());
        uint16_t metricId = std::get<1>(result_2.value());
        double value = std::get<2>(result_2.value());

        ProfileRow row;
        row.profile_id = profile_index;
        row.cct_id = cct_id;
        row.metric_id = metricId;
        row.value = value;
        rows.push_back(row);
        end_index = mid_2;
      }
    }
  } else {
    for (uint64_t j = startIndex; j < end_index; ++j) {
      uint16_t metricId =
          safe_unpack<uint16_t>(values_sub_db, j * METRIC_TUPLE_SIZE);
      double value = safe_unpack<double>(values_sub_db, j * METRIC_TUPLE_SIZE +
                                                            sizeof(uint16_t));

      ProfileRow row;
      row.profile_id = profile_index;
      row.cct_id = cct_id;
      row.metric_id = metricId;
      row.value = value;
      rows.push_back(row);
    }
  }
}

std::vector<ProfileRow>
parse_profile(const char *profile_db, uint32_t profile_index,
              std::vector<uint32_t> cct_indices,
              const std::vector<uint16_t> &metric_indices) {
  std::vector<ProfileRow> rows;

  uint64_t szProfileInfos =
      safe_unpack<uint64_t>(profile_db, FILE_HEADER_OFFSET);
  uint64_t pProfileInfos =
      safe_unpack<uint64_t>(profile_db, FILE_HEADER_OFFSET + sizeof(uint64_t));

  uint64_t pProfiles = safe_unpack<uint64_t>(profile_db, pProfileInfos);
  uint8_t szProfile = safe_unpack<uint8_t>(
      profile_db, pProfileInfos + sizeof(uint64_t) + sizeof(uint32_t));

  const auto base_offset = pProfiles + profile_index * szProfile;

  uint64_t nValues = safe_unpack<uint64_t>(profile_db, base_offset);
  uint64_t pValues =
      safe_unpack<uint64_t>(profile_db, base_offset + sizeof(uint64_t));
  uint32_t nCtxs =
      safe_unpack<uint32_t>(profile_db, base_offset + 2 * sizeof(uint64_t));
  uint64_t pCtxIndices = safe_unpack<uint64_t>(
      profile_db, base_offset + 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t));

  const char *cct_sub_db = profile_db + pCtxIndices;
  const char *values_sub_db = profile_db + pValues;

  if (!cct_indices.empty()) {
    uint64_t low = 0;
    uint64_t high = nCtxs - 1;

    while (!cct_indices.empty()) {
      uint32_t cct_id = cct_indices.back();
      cct_indices.pop_back();

      auto result = cct_metrics_binary_search<uint32_t, uint64_t>(
          cct_sub_db, low, high, cct_id);

      if (result.has_value()) {
        uint64_t mid = std::get<0>(result.value());
        uint32_t ctxId = std::get<1>(result.value());
        uint64_t startIndex = std::get<2>(result.value());
        uint64_t end_index;

        if (mid == nCtxs - 1) {
          end_index = nValues;
        } else {
          end_index = safe_unpack<uint64_t>(
              cct_sub_db, (mid + 1) * CCT_TUPLE_SIZE + sizeof(uint32_t));
        }

        high = mid;
        extract_metrics(profile_index, ctxId, metric_indices, values_sub_db,
                        startIndex, end_index, rows);
      }
    }
  } else {
    for (uint32_t i = 0; i < nCtxs; ++i) {
      uint32_t ctxId = safe_unpack<uint32_t>(cct_sub_db, i * CCT_TUPLE_SIZE);
      uint64_t startIndex = safe_unpack<uint64_t>(
          cct_sub_db, i * CCT_TUPLE_SIZE + sizeof(uint32_t));
      uint64_t end_index;

      if (i == nCtxs - 1) {
        end_index = nValues;
      } else {
        end_index = safe_unpack<uint64_t>(cct_sub_db, (i + 1) * CCT_TUPLE_SIZE +
                                                          sizeof(uint32_t));
      }

      extract_metrics(profile_index, ctxId, metric_indices, values_sub_db,
                      startIndex, end_index, rows);
    }
  }

  return rows;
}

py::list
DataRead::read_profile_slices(const std::vector<uint32_t> &profile_indices,
                              const std::vector<uint32_t> &cct_indices,
                              const std::vector<uint16_t> &metric_indices,
                              const size_t thread_count) {
  int fd = open(_profile_file.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Failed to open profile file: " + _profile_file);
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    throw std::runtime_error("Failed to stat profile file: " + _profile_file);
  }
  size_t file_size = st.st_size;

  const char *mapped_data =
      (const char *)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped_data == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to mmap profile file: " + _profile_file);
  }

  omp_set_num_threads(thread_count);

  const size_t profiles_count = profile_indices.size();
  std::vector<std::vector<ProfileRow>> profile_results(profiles_count);

#pragma omp parallel for default(none)                                         \
    shared(profile_results, mapped_data, profile_indices, cct_indices,         \
           metric_indices, profiles_count)
  for (size_t task_index = 0; task_index < profiles_count; ++task_index) {
    profile_results[task_index] = parse_profile(
        mapped_data, profile_indices[task_index], cct_indices, metric_indices);
  }

  py::list final_results = py::list();

  for (const auto &profile_result : profile_results) {
    for (const ProfileRow &profile_row : profile_result) {
      py::dict py_row;
      py_row["profile_id"] = profile_row.profile_id;
      py_row["cct_id"] = profile_row.cct_id;
      py_row["metric_id"] = profile_row.metric_id;
      py_row["value"] = profile_row.value;
      final_results.append(py_row);
    }
  }

  munmap((void *)mapped_data, file_size);
  close(fd);

  return final_results;
}

std::vector<TraceRow>
parse_trace(const char *trace_db, uint32_t profile_index,
            const std::vector<std::pair<uint64_t, uint64_t>> &time_frames) {
  std::vector<TraceRow> rows;

  uint64_t szCtxTraces = safe_unpack<uint64_t>(trace_db, FILE_HEADER_OFFSET);
  uint64_t pCtxTraces =
      safe_unpack<uint64_t>(trace_db, FILE_HEADER_OFFSET + sizeof(uint64_t));

  const char *trace_header_sub_db = trace_db + pCtxTraces;

  uint64_t pTraces = safe_unpack<uint64_t>(trace_header_sub_db, 0);
  uint8_t szTrace = safe_unpack<uint8_t>(trace_header_sub_db,
                                         sizeof(uint64_t) + sizeof(uint32_t));

  uint64_t minTimestamp =
      safe_unpack<uint64_t>(trace_header_sub_db, 2 * sizeof(uint64_t));
  uint64_t maxTimestamp =
      safe_unpack<uint64_t>(trace_header_sub_db, 3 * sizeof(uint64_t));

  const auto base_offset = pTraces - pCtxTraces + (profile_index - 1) * szTrace;
  uint64_t pStart = safe_unpack<uint64_t>(trace_header_sub_db,
                                          base_offset + 2 * sizeof(uint32_t));
  uint64_t pEnd = safe_unpack<uint64_t>(trace_header_sub_db,
                                        base_offset + 2 * sizeof(uint32_t) +
                                            sizeof(uint64_t));

  std::vector<std::pair<uint64_t, uint64_t>> frames_to_process;
  if (time_frames.empty()) {
    frames_to_process.push_back({minTimestamp, maxTimestamp});
  } else {
    frames_to_process = time_frames;
  }

  const uint64_t num_samples = (pEnd - pStart) / EVENT_TUPLE_SIZE;
  const char *trace_events_sub_db = trace_db + pStart;

  for (const auto &time_frame : frames_to_process) {
    std::vector<TraceRow> frame_rows;
    TraceRow *last_row_ptr = nullptr;

    uint64_t start_timestamp = time_frame.first;
    uint64_t end_timestamp = time_frame.second;

    uint64_t sample_index =
        trace_binary_search(trace_events_sub_db, num_samples, start_timestamp);

    while (sample_index < num_samples) {
      uint64_t timestamp = safe_unpack<uint64_t>(
          trace_events_sub_db, sample_index * EVENT_TUPLE_SIZE);
      uint32_t ctxId = safe_unpack<uint32_t>(trace_events_sub_db,
                                             sample_index * EVENT_TUPLE_SIZE +
                                                 sizeof(uint64_t));

      if (timestamp > end_timestamp) {
        if (last_row_ptr != nullptr) {
          last_row_ptr->end_timestamp = timestamp;
        }
        break;
      }

      if (last_row_ptr != nullptr) {
        last_row_ptr->end_timestamp = timestamp;
      }

      if (frame_rows.empty() || last_row_ptr->cct_id != ctxId) {
        TraceRow new_row;
        new_row.profile_id = profile_index;
        new_row.cct_id = ctxId;
        new_row.start_timestamp = timestamp;
        new_row.end_timestamp = timestamp;
        frame_rows.push_back(std::move(new_row));
        last_row_ptr = &frame_rows.back();
      }

      sample_index += 1;
    }

    if (sample_index >= num_samples) {
      if (last_row_ptr != nullptr) {
        last_row_ptr->end_timestamp = maxTimestamp;
      }
    }

    rows.insert(rows.end(), std::make_move_iterator(frame_rows.begin()),
                std::make_move_iterator(frame_rows.end()));
  }

  return rows;
}

py::list DataRead::read_trace_slices(const TraceProfileMap &profile_indices,
                                     const size_t thread_count) {
  int fd = open(_trace_file.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Failed to open trace file: " + _trace_file);
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    throw std::runtime_error("Failed to stat trace file: " + _trace_file);
  }
  size_t file_size = st.st_size;

  const char *mapped_data =
      (const char *)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped_data == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to mmap trace file: " + _profile_file);
  }

  std::vector<uint32_t> profile_keys;
  for (const auto &pair : profile_indices) {
    profile_keys.push_back(pair.first);
  }

  omp_set_num_threads(thread_count);

  const size_t profiles_count = profile_keys.size();
  std::vector<std::vector<TraceRow>> trace_results(profiles_count);

#pragma omp parallel for default(none)                                         \
    shared(trace_results, mapped_data, profile_indices, profile_keys,          \
           profiles_count)
  for (size_t task_index = 0; task_index < profiles_count; ++task_index) {
    uint32_t profile_id = profile_keys[task_index];
    const auto &time_frames = profile_indices.at(profile_id);
    trace_results[task_index] =
        parse_trace(mapped_data, profile_id, time_frames);
  }

  py::list final_results = py::list();

  for (const auto &trace_result : trace_results) {
    for (const TraceRow &trace_row : trace_result) {
      py::dict py_row;
      py_row["profile_id"] = trace_row.profile_id;
      py_row["cct_id"] = trace_row.cct_id;
      py_row["start_timestamp"] = trace_row.start_timestamp;
      py_row["end_timestamp"] = trace_row.end_timestamp;
      final_results.append(py_row);
    }
  }

  munmap((void *)mapped_data, file_size);
  close(fd);

  return final_results;
}
