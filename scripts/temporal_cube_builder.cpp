/**
 * Temporal PC Sampling Cube Builder (HPCToolkit)
 *
 * High-performance C++ + OpenMP implementation for processing
 * HPCToolkit's temporal PC sampling data.
 *
 * Reads temporal-*.bin files and builds a 3D data cube:
 *   (CCT_node × StallReason × TimeBin) -> sample_count
 *
 * Supports temporal file format v1 and v2:
 *   v1: 16-byte samples (no hw_id)
 *   v2: 38-byte samples (includes ROCprofiler hw_id fields)
 *
 * Optional: emit hw_id histograms (global counts) to CSV.
 *
 * Build:
 *   g++ -O3 -fopenmp -std=c++17 -o temporal_cube_builder scripts/temporal_cube_builder.cpp
 *
 * Usage:
 *   ./temporal_cube_builder <measurements_dir> [--time-bins N] [--output cube.bin]
 *   ./temporal_cube_builder <measurements_dir> --hwid-out hwid.csv
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

constexpr uint32_t TEMPORAL_MAGIC = 0x54435048;  // "HPCT"
constexpr uint32_t TEMPORAL_VERSION_V1 = 1;
constexpr uint32_t TEMPORAL_VERSION_V2 = 2;
constexpr int NUM_STALL_REASONS = 14;  // 0-13 in HPCToolkit
constexpr int DEFAULT_TIME_BINS = 100;

// hw_id bit widths (from ROCprofiler PC sampling)
constexpr size_t HW_CHIPLET_SIZE = 1u << 6;        // 6 bits
constexpr size_t HW_CU_WGP_SIZE = 1u << 4;         // 4 bits
constexpr size_t HW_SIMD_SIZE = 1u << 2;           // 2 bits
constexpr size_t HW_WAVE_ID_SIZE = 1u << 7;        // 7 bits
constexpr size_t HW_PIPE_SIZE = 1u << 4;           // 4 bits
constexpr size_t HW_WORKGROUP_ID_SIZE = 1u << 7;   // 7 bits
constexpr size_t HW_SHADER_ENGINE_SIZE = 1u << 5;  // 5 bits
constexpr size_t HW_SHADER_ARRAY_SIZE = 1u << 1;   // 1 bit
constexpr size_t HW_VM_ID_SIZE = 1u << 6;          // 6 bits
constexpr size_t HW_QUEUE_ID_SIZE = 1u << 4;       // 4 bits
constexpr size_t HW_MICROENGINE_SIZE = 1u << 2;    // 2 bits

// Stall reason names for output
const char* STALL_NAMES[] = {
    "NONE", "IFETCH", "IDEPEND", "MEM", "GMEM", "TMEM", "SYNC",
    "CMEM", "PIPE_BUSY", "MEM_THROTTLE", "OTHER", "SLEEP", "HIDDEN", "INVALID"
};

//------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------

#pragma pack(push, 1)
struct TemporalFileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t sample_count;
    uint64_t start_timestamp;
    uint64_t end_timestamp;
};

// v1 sample (16 bytes)
struct TemporalSampleV1 {
    uint32_t cct_node_id;
    uint64_t timestamp;
    uint16_t stall_reason;
    uint16_t inst_type;
};

// v2 sample (38 bytes) - includes hw_id fields
struct TemporalSampleV2 {
    uint32_t cct_node_id;
    uint64_t timestamp;
    uint16_t stall_reason;
    uint16_t inst_type;
    uint16_t hw_chiplet;
    uint16_t hw_cu_or_wgp_id;
    uint16_t hw_simd_id;
    uint16_t hw_wave_id;
    uint16_t hw_pipe_id;
    uint16_t hw_workgroup_id;
    uint16_t hw_shader_engine_id;
    uint16_t hw_shader_array_id;
    uint16_t hw_vm_id;
    uint16_t hw_queue_id;
    uint16_t hw_microengine_id;
};
#pragma pack(pop)

static_assert(sizeof(TemporalFileHeader) == 32, "Header size mismatch");
static_assert(sizeof(TemporalSampleV1) == 16, "v1 sample size mismatch");
static_assert(sizeof(TemporalSampleV2) == 38, "v2 sample size mismatch");

using TemporalSample = TemporalSampleV2;

struct HwIdHist {
    std::vector<uint64_t> chiplet = std::vector<uint64_t>(HW_CHIPLET_SIZE, 0);
    std::vector<uint64_t> cu_or_wgp = std::vector<uint64_t>(HW_CU_WGP_SIZE, 0);
    std::vector<uint64_t> simd = std::vector<uint64_t>(HW_SIMD_SIZE, 0);
    std::vector<uint64_t> wave_id = std::vector<uint64_t>(HW_WAVE_ID_SIZE, 0);
    std::vector<uint64_t> pipe = std::vector<uint64_t>(HW_PIPE_SIZE, 0);
    std::vector<uint64_t> workgroup = std::vector<uint64_t>(HW_WORKGROUP_ID_SIZE, 0);
    std::vector<uint64_t> shader_engine = std::vector<uint64_t>(HW_SHADER_ENGINE_SIZE, 0);
    std::vector<uint64_t> shader_array = std::vector<uint64_t>(HW_SHADER_ARRAY_SIZE, 0);
    std::vector<uint64_t> vm_id = std::vector<uint64_t>(HW_VM_ID_SIZE, 0);
    std::vector<uint64_t> queue_id = std::vector<uint64_t>(HW_QUEUE_ID_SIZE, 0);
    std::vector<uint64_t> microengine = std::vector<uint64_t>(HW_MICROENGINE_SIZE, 0);
    uint64_t samples = 0;
};

struct HwIdTime {
    int time_bins = 0;
    std::vector<uint64_t> chiplet;
    std::vector<uint64_t> cu_or_wgp;
    std::vector<uint64_t> simd;
    std::vector<uint64_t> wave_id;
    std::vector<uint64_t> pipe;
    std::vector<uint64_t> workgroup;
    std::vector<uint64_t> shader_engine;
    std::vector<uint64_t> shader_array;
    std::vector<uint64_t> vm_id;
    std::vector<uint64_t> queue_id;
    std::vector<uint64_t> microengine;

    explicit HwIdTime(int bins)
        : time_bins(bins),
          chiplet(HW_CHIPLET_SIZE * bins, 0),
          cu_or_wgp(HW_CU_WGP_SIZE * bins, 0),
          simd(HW_SIMD_SIZE * bins, 0),
          wave_id(HW_WAVE_ID_SIZE * bins, 0),
          pipe(HW_PIPE_SIZE * bins, 0),
          workgroup(HW_WORKGROUP_ID_SIZE * bins, 0),
          shader_engine(HW_SHADER_ENGINE_SIZE * bins, 0),
          shader_array(HW_SHADER_ARRAY_SIZE * bins, 0),
          vm_id(HW_VM_ID_SIZE * bins, 0),
          queue_id(HW_QUEUE_ID_SIZE * bins, 0),
          microengine(HW_MICROENGINE_SIZE * bins, 0) {}
};

//------------------------------------------------------------------------------
// File I/O
//------------------------------------------------------------------------------

static int compute_time_bin(uint64_t timestamp,
                            uint64_t min_ts,
                            uint64_t max_ts,
                            int time_bins) {
    uint64_t range = max_ts - min_ts;
    if (range == 0) range = 1;
    uint64_t rel = timestamp - min_ts;
    int t_idx = static_cast<int>((rel * (time_bins - 1)) / range);
    if (t_idx < 0) t_idx = 0;
    if (t_idx >= time_bins) t_idx = time_bins - 1;
    return t_idx;
}

static void update_hw_hist(HwIdHist& hist, const TemporalSample& s) {
    auto safe_inc = [](std::vector<uint64_t>& vec, uint16_t idx) {
        size_t sidx = static_cast<size_t>(idx);
        if (sidx < vec.size()) vec[sidx]++;
    };

    safe_inc(hist.chiplet, s.hw_chiplet);
    safe_inc(hist.cu_or_wgp, s.hw_cu_or_wgp_id);
    safe_inc(hist.simd, s.hw_simd_id);
    safe_inc(hist.wave_id, s.hw_wave_id);
    safe_inc(hist.pipe, s.hw_pipe_id);
    safe_inc(hist.workgroup, s.hw_workgroup_id);
    safe_inc(hist.shader_engine, s.hw_shader_engine_id);
    safe_inc(hist.shader_array, s.hw_shader_array_id);
    safe_inc(hist.vm_id, s.hw_vm_id);
    safe_inc(hist.queue_id, s.hw_queue_id);
    safe_inc(hist.microengine, s.hw_microengine_id);
    hist.samples++;
}

static void update_hw_time(HwIdTime& time_data, const TemporalSample& s, int t_idx) {
    const size_t t = static_cast<size_t>(t_idx);
    auto safe_inc_time = [&](std::vector<uint64_t>& vec, uint16_t idx) {
        size_t sidx = static_cast<size_t>(idx);
        size_t bins = static_cast<size_t>(time_data.time_bins);
        if (sidx < vec.size() / bins) {
            vec[sidx * bins + t]++;
        }
    };

    safe_inc_time(time_data.chiplet, s.hw_chiplet);
    safe_inc_time(time_data.cu_or_wgp, s.hw_cu_or_wgp_id);
    safe_inc_time(time_data.simd, s.hw_simd_id);
    safe_inc_time(time_data.wave_id, s.hw_wave_id);
    safe_inc_time(time_data.pipe, s.hw_pipe_id);
    safe_inc_time(time_data.workgroup, s.hw_workgroup_id);
    safe_inc_time(time_data.shader_engine, s.hw_shader_engine_id);
    safe_inc_time(time_data.shader_array, s.hw_shader_array_id);
    safe_inc_time(time_data.vm_id, s.hw_vm_id);
    safe_inc_time(time_data.queue_id, s.hw_queue_id);
    safe_inc_time(time_data.microengine, s.hw_microengine_id);
}

static void write_hist_section(std::ofstream& out,
                               const char* field,
                               const std::vector<uint64_t>& counts) {
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] == 0) continue;
        out << field << "," << i << "," << counts[i] << "\n";
    }
}

static void write_hw_hist_csv(const std::string& path, const HwIdHist& hist) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Error: Cannot write hw_id histogram to " << path << std::endl;
        return;
    }

    out << "field,value,count\n";
    write_hist_section(out, "chiplet", hist.chiplet);
    write_hist_section(out, "cu_or_wgp_id", hist.cu_or_wgp);
    write_hist_section(out, "simd_id", hist.simd);
    write_hist_section(out, "wave_id", hist.wave_id);
    write_hist_section(out, "pipe_id", hist.pipe);
    write_hist_section(out, "workgroup_id", hist.workgroup);
    write_hist_section(out, "shader_engine_id", hist.shader_engine);
    write_hist_section(out, "shader_array_id", hist.shader_array);
    write_hist_section(out, "vm_id", hist.vm_id);
    write_hist_section(out, "queue_id", hist.queue_id);
    write_hist_section(out, "microengine_id", hist.microengine);

    out.close();
    std::cout << "hw_id histogram written to: " << path
              << " (samples=" << hist.samples << ")" << std::endl;
}

static void write_hwid_time_section(std::ofstream& out,
                                    const char* field,
                                    const std::vector<uint64_t>& counts,
                                    int time_bins) {
    const size_t value_count = counts.size() / static_cast<size_t>(time_bins);
    for (size_t value = 0; value < value_count; ++value) {
        const size_t base = value * static_cast<size_t>(time_bins);
        for (int t = 0; t < time_bins; ++t) {
            uint64_t count = counts[base + static_cast<size_t>(t)];
            if (count == 0) continue;
            out << field << "," << value << "," << t << "," << count << "\n";
        }
    }
}

static void write_hwid_time_csv(const std::string& path, const HwIdTime& data) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Error: Cannot write hw_id time series to " << path << std::endl;
        return;
    }

    out << "field,value,time_bin,count\n";
    write_hwid_time_section(out, "chiplet", data.chiplet, data.time_bins);
    write_hwid_time_section(out, "cu_or_wgp_id", data.cu_or_wgp, data.time_bins);
    write_hwid_time_section(out, "simd_id", data.simd, data.time_bins);
    write_hwid_time_section(out, "wave_id", data.wave_id, data.time_bins);
    write_hwid_time_section(out, "pipe_id", data.pipe, data.time_bins);
    write_hwid_time_section(out, "workgroup_id", data.workgroup, data.time_bins);
    write_hwid_time_section(out, "shader_engine_id", data.shader_engine, data.time_bins);
    write_hwid_time_section(out, "shader_array_id", data.shader_array, data.time_bins);
    write_hwid_time_section(out, "vm_id", data.vm_id, data.time_bins);
    write_hwid_time_section(out, "queue_id", data.queue_id, data.time_bins);
    write_hwid_time_section(out, "microengine_id", data.microengine, data.time_bins);

    out.close();
    std::cout << "hw_id time series written to: " << path << std::endl;
}

static std::vector<TemporalSample>
read_temporal_file(const std::string& filepath,
                   uint64_t& min_ts,
                   uint64_t& max_ts,
                   bool& file_has_hw_id) {
    std::vector<TemporalSample> samples;
    file_has_hw_id = false;

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Warning: Cannot open " << filepath << std::endl;
        return samples;
    }

    TemporalFileHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        std::cerr << "Warning: Failed to read header from " << filepath << std::endl;
        return samples;
    }

    if (header.magic != TEMPORAL_MAGIC) {
        std::cerr << "Warning: Invalid magic in " << filepath << std::endl;
        return samples;
    }

    if (header.sample_count == 0) return samples;

    // Update timestamp range
    if (header.start_timestamp < min_ts) min_ts = header.start_timestamp;
    if (header.end_timestamp > max_ts) max_ts = header.end_timestamp;

    if (header.version == TEMPORAL_VERSION_V1) {
        std::vector<TemporalSampleV1> v1(header.sample_count);
        file.read(reinterpret_cast<char*>(v1.data()),
                  header.sample_count * sizeof(TemporalSampleV1));
        if (!file) {
            std::cerr << "Warning: Failed to read v1 samples from " << filepath << std::endl;
            return samples;
        }
        samples.resize(header.sample_count);
        for (size_t i = 0; i < v1.size(); ++i) {
            samples[i] = TemporalSample{
                v1[i].cct_node_id,
                v1[i].timestamp,
                v1[i].stall_reason,
                v1[i].inst_type,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };
        }
        file_has_hw_id = false;
    } else if (header.version == TEMPORAL_VERSION_V2) {
        samples.resize(header.sample_count);
        file.read(reinterpret_cast<char*>(samples.data()),
                  header.sample_count * sizeof(TemporalSample));
        if (!file) {
            std::cerr << "Warning: Failed to read v2 samples from " << filepath << std::endl;
            samples.clear();
            return samples;
        }
        file_has_hw_id = true;
    } else {
        std::cerr << "Warning: Unsupported temporal version "
                  << header.version << " in " << filepath << std::endl;
        return samples;
    }

    return samples;
}

static std::vector<std::string> find_temporal_files(const std::string& dir) {
    std::vector<std::string> files;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (name.rfind("temporal-", 0) == 0 &&
                name.size() >= 4 && name.substr(name.size() - 4) == ".bin") {
                files.push_back(entry.path().string());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

//------------------------------------------------------------------------------
// Cube Building
//------------------------------------------------------------------------------

struct CubeBuilder {
    int num_time_bins;
    uint64_t min_timestamp;
    uint64_t max_timestamp;

    // Mapping: cct_id -> index
    std::unordered_map<uint32_t, int> cct_to_idx;
    std::vector<uint32_t> idx_to_cct;

    // 3D Cube: [cct_idx][stall_reason][time_bin]
    std::vector<std::vector<std::vector<uint64_t>>> cube;

    CubeBuilder(int time_bins)
        : num_time_bins(time_bins), min_timestamp(UINT64_MAX), max_timestamp(0) {}

    void collect_cct_ids(const std::vector<TemporalSample>& samples) {
        std::unordered_set<uint32_t> cct_set;
        for (const auto& s : samples) {
            cct_set.insert(s.cct_node_id);
        }

        idx_to_cct.assign(cct_set.begin(), cct_set.end());
        std::sort(idx_to_cct.begin(), idx_to_cct.end());

        for (size_t i = 0; i < idx_to_cct.size(); ++i) {
            cct_to_idx[idx_to_cct[i]] = static_cast<int>(i);
        }
    }

    void allocate_cube() {
        int C = static_cast<int>(idx_to_cct.size());
        int R = NUM_STALL_REASONS;
        int T = num_time_bins;

        cube.resize(C);
        for (int c = 0; c < C; ++c) {
            cube[c].resize(R);
            for (int r = 0; r < R; ++r) {
                cube[c][r].resize(T, 0);
            }
        }
    }

    void fill_cube_parallel(const std::vector<TemporalSample>& samples) {
        uint64_t time_range = max_timestamp - min_timestamp;
        if (time_range == 0) time_range = 1;

        int C = static_cast<int>(idx_to_cct.size());
        int R = NUM_STALL_REASONS;
        int T = num_time_bins;

        // Thread-local cubes to avoid race conditions
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::vector<std::vector<uint64_t>>>> local_cubes(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // Allocate local cube
            local_cubes[tid].resize(C);
            for (int c = 0; c < C; ++c) {
                local_cubes[tid][c].resize(R);
                for (int r = 0; r < R; ++r) {
                    local_cubes[tid][c][r].resize(T, 0);
                }
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < samples.size(); ++i) {
                const auto& s = samples[i];

                auto it = cct_to_idx.find(s.cct_node_id);
                if (it == cct_to_idx.end()) continue;

                int c_idx = it->second;
                int r_idx = std::min(static_cast<int>(s.stall_reason), R - 1);

                // Compute time bin
                uint64_t rel_time = s.timestamp - min_timestamp;
                int t_idx = static_cast<int>((rel_time * (T - 1)) / time_range);
                t_idx = std::min(t_idx, T - 1);

                local_cubes[tid][c_idx][r_idx][t_idx]++;
            }
        }

        // Merge local cubes into global cube
        #pragma omp parallel for collapse(3) schedule(static)
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int t = 0; t < T; ++t) {
                    for (int tid = 0; tid < num_threads; ++tid) {
                        cube[c][r][t] += local_cubes[tid][c][r][t];
                    }
                }
            }
        }
    }

    void write_output(const std::string& output_path) {
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Cannot write to " << output_path << std::endl;
            return;
        }

        int C = static_cast<int>(idx_to_cct.size());
        int R = NUM_STALL_REASONS;
        int T = num_time_bins;

        // Write header
        // Format: [C, R, T] dimensions + CCT IDs + flattened cube data
        out.write(reinterpret_cast<const char*>(&C), sizeof(int));
        out.write(reinterpret_cast<const char*>(&R), sizeof(int));
        out.write(reinterpret_cast<const char*>(&T), sizeof(int));

        // Write timestamp range
        out.write(reinterpret_cast<const char*>(&min_timestamp), sizeof(uint64_t));
        out.write(reinterpret_cast<const char*>(&max_timestamp), sizeof(uint64_t));

        // Write CCT ID mapping
        out.write(reinterpret_cast<const char*>(idx_to_cct.data()),
                  C * sizeof(uint32_t));

        // Write cube data (row-major: C × R × T)
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                out.write(reinterpret_cast<const char*>(cube[c][r].data()),
                          T * sizeof(uint64_t));
            }
        }

        out.close();
        std::cout << "Output written to: " << output_path << std::endl;
    }

    void print_summary(size_t total_samples) {
        int C = static_cast<int>(idx_to_cct.size());
        int R = NUM_STALL_REASONS;
        int T = num_time_bins;

        std::cout << "\n========================================" << std::endl;
        std::cout << "TEMPORAL CUBE BUILDER SUMMARY" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total samples: " << total_samples << std::endl;
        std::cout << "Time range: " << (max_timestamp - min_timestamp) / 1e6
                  << "M GPU cycles" << std::endl;
        std::cout << "\nCube dimensions:" << std::endl;
        std::cout << "  CCT nodes:     " << C << std::endl;
        std::cout << "  Stall reasons: " << R << std::endl;
        std::cout << "  Time bins:     " << T << std::endl;
        std::cout << "  Total cells:   " << (size_t)C * R * T << std::endl;

        // Compute stall distribution
        std::vector<uint64_t> stall_totals(R, 0);
        uint64_t grand_total = 0;

        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int t = 0; t < T; ++t) {
                    stall_totals[r] += cube[c][r][t];
                    grand_total += cube[c][r][t];
                }
            }
        }

        std::cout << "\nStall Reason Distribution:" << std::endl;
        for (int r = 0; r < R; ++r) {
            if (stall_totals[r] > 0) {
                double pct = 100.0 * stall_totals[r] / grand_total;
                printf("  %-12s: %8lu (%5.1f%%)\n",
                       STALL_NAMES[r], stall_totals[r], pct);
            }
        }

        // Top CCT nodes
        std::vector<std::pair<uint64_t, int>> cct_totals;
        for (int c = 0; c < C; ++c) {
            uint64_t total = 0;
            for (int r = 0; r < R; ++r) {
                for (int t = 0; t < T; ++t) {
                    total += cube[c][r][t];
                }
            }
            cct_totals.push_back({total, c});
        }
        std::sort(cct_totals.rbegin(), cct_totals.rend());

        std::cout << "\nTop 5 CCT Nodes:" << std::endl;
        for (int i = 0; i < std::min(5, C); ++i) {
            auto [count, idx] = cct_totals[i];
            double pct = 100.0 * count / grand_total;
            printf("  CCT#%-8u: %8lu (%5.1f%%)\n",
                   idx_to_cct[idx], count, pct);
        }
    }
};

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <measurements_dir> [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --time-bins N    Number of time bins (default: 100)" << std::endl;
    std::cerr << "  --output FILE    Output file path (default: cube.bin)" << std::endl;
    std::cerr << "  --threads N      Number of OpenMP threads" << std::endl;
    std::cerr << "  --hwid-out FILE  Output CSV with hw_id histograms" << std::endl;
    std::cerr << "  --hwid-time-out FILE  Output CSV with hw_id time-series" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string measurements_dir;
    int num_time_bins = DEFAULT_TIME_BINS;
    std::string output_path = "cube.bin";
    std::string hwid_out_path;
    std::string hwid_time_out_path;
    int num_threads = omp_get_max_threads();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--time-bins" && i + 1 < argc) {
            num_time_bins = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
            omp_set_num_threads(num_threads);
        } else if (arg == "--hwid-out" && i + 1 < argc) {
            hwid_out_path = argv[++i];
        } else if (arg == "--hwid-time-out" && i + 1 < argc) {
            hwid_time_out_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        } else if (measurements_dir.empty()) {
            measurements_dir = arg;
        } else {
            std::cerr << "Unexpected argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (measurements_dir.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Temporal Cube Builder (C++ + OpenMP)" << std::endl;
    std::cout << "Measurements dir: " << measurements_dir << std::endl;
    std::cout << "Time bins: " << num_time_bins << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    if (!hwid_out_path.empty()) {
        std::cout << "hw_id histogram: " << hwid_out_path << std::endl;
    }
    if (!hwid_time_out_path.empty()) {
        std::cout << "hw_id time series: " << hwid_time_out_path << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Find temporal files
    auto temporal_files = find_temporal_files(measurements_dir);
    if (temporal_files.empty()) {
        std::cerr << "Error: No temporal-*.bin files found in " << measurements_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << temporal_files.size() << " temporal file(s)" << std::endl;

    // Read all samples
    CubeBuilder builder(num_time_bins);
    std::vector<TemporalSample> all_samples;
    std::vector<uint8_t> sample_has_hwid;
    HwIdHist hw_hist;
    bool any_hw_id = false;

    for (const auto& filepath : temporal_files) {
        std::cout << "Reading " << fs::path(filepath).filename() << "..." << std::endl;
        bool file_has_hw_id = false;
        auto samples = read_temporal_file(filepath,
                                          builder.min_timestamp,
                                          builder.max_timestamp,
                                          file_has_hw_id);
        if (file_has_hw_id) {
            any_hw_id = true;
            if (!hwid_out_path.empty()) {
                for (const auto& s : samples) {
                    update_hw_hist(hw_hist, s);
                }
            }
        }
        for (const auto& s : samples) {
            all_samples.push_back(s);
            sample_has_hwid.push_back(file_has_hw_id ? 1 : 0);
        }
    }

    std::cout << "Total samples loaded: " << all_samples.size() << std::endl;

    if (all_samples.empty()) {
        std::cerr << "Error: No samples found!" << std::endl;
        return 1;
    }

    // Build cube
    std::cout << "Collecting CCT IDs..." << std::endl;
    builder.collect_cct_ids(all_samples);

    std::cout << "Allocating cube..." << std::endl;
    builder.allocate_cube();

    std::cout << "Building cube (parallel)..." << std::endl;
    builder.fill_cube_parallel(all_samples);

    // Write output
    builder.write_output(output_path);

    // Optional hw_id histogram
    if (!hwid_out_path.empty()) {
        if (!any_hw_id) {
            std::cerr << "Warning: No v2 temporal files found; hw_id histogram skipped."
                      << std::endl;
        } else {
            write_hw_hist_csv(hwid_out_path, hw_hist);
        }
    }

    // Optional hw_id time series
    if (!hwid_time_out_path.empty()) {
        if (!any_hw_id) {
            std::cerr << "Warning: No v2 temporal files found; hw_id time series skipped."
                      << std::endl;
        } else {
            HwIdTime hw_time(num_time_bins);
            int max_threads = omp_get_max_threads();
            std::vector<HwIdTime> local_times;
            local_times.reserve(max_threads);
            for (int i = 0; i < max_threads; ++i) {
                local_times.emplace_back(num_time_bins);
            }

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                HwIdTime& local = local_times[tid];
                #pragma omp for schedule(static)
                for (size_t i = 0; i < all_samples.size(); ++i) {
                    if (sample_has_hwid[i] == 0) continue;
                    const auto& s = all_samples[i];
                    int t_idx = compute_time_bin(s.timestamp,
                                                 builder.min_timestamp,
                                                 builder.max_timestamp,
                                                 num_time_bins);
                    update_hw_time(local, s, t_idx);
                }
            }

            auto merge_field = [&](std::vector<uint64_t>& dst,
                                   const std::vector<uint64_t>& src) {
                for (size_t i = 0; i < dst.size(); ++i) {
                    dst[i] += src[i];
                }
            };

            for (int i = 0; i < max_threads; ++i) {
                merge_field(hw_time.chiplet, local_times[i].chiplet);
                merge_field(hw_time.cu_or_wgp, local_times[i].cu_or_wgp);
                merge_field(hw_time.simd, local_times[i].simd);
                merge_field(hw_time.wave_id, local_times[i].wave_id);
                merge_field(hw_time.pipe, local_times[i].pipe);
                merge_field(hw_time.workgroup, local_times[i].workgroup);
                merge_field(hw_time.shader_engine, local_times[i].shader_engine);
                merge_field(hw_time.shader_array, local_times[i].shader_array);
                merge_field(hw_time.vm_id, local_times[i].vm_id);
                merge_field(hw_time.queue_id, local_times[i].queue_id);
                merge_field(hw_time.microengine, local_times[i].microengine);
            }

            write_hwid_time_csv(hwid_time_out_path, hw_time);
        }
    }

    // Summary
    builder.print_summary(all_samples.size());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nTotal time: " << duration.count() << " ms" << std::endl;

    return 0;
}
