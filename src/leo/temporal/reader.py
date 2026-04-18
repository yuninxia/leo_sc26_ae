"""Reader for HPCToolkit temporal PC sampling binary files.

Reads temporal-*.bin files produced by HPCToolkit's GPU PC sampling
with hardware ID fields (chiplet, CU, SIMD, wavefront, etc.) from
ROCprofiler.

Binary format:
    Header (32 bytes): magic, version, sample_count, start_ts, end_ts
    V1 samples (16 bytes each): cct_node_id, timestamp, stall_reason, inst_type
    V2 samples (38 bytes each): V1 fields + 11 hardware ID fields
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# File format constants
TEMPORAL_MAGIC = 0x54435048  # "HPCT" in little-endian
TEMPORAL_VERSION_V1 = 1
TEMPORAL_VERSION_V2 = 2

# Struct formats (little-endian)
HEADER_FMT = "<IIQqq"  # magic(u32), version(u32), count(u64), start_ts(i64), end_ts(i64)
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 32 bytes

SAMPLE_V1_FMT = "<IqHH"  # cct_id(u32), timestamp(i64), stall(u16), inst_type(u16)
SAMPLE_V1_SIZE = struct.calcsize(SAMPLE_V1_FMT)  # 16 bytes

SAMPLE_V2_FMT = "<IqHH11H"  # V1 fields + 11 hw_id fields (u16 each)
SAMPLE_V2_SIZE = struct.calcsize(SAMPLE_V2_FMT)  # 38 bytes


@dataclass
class TemporalSample:
    """A single PC sample with optional hardware ID fields."""

    cct_node_id: int
    timestamp: int
    stall_reason: int
    inst_type: int
    # Hardware ID fields (v2 only, 0 for v1)
    chiplet: int = 0
    cu_or_wgp_id: int = 0
    simd_id: int = 0
    wave_id: int = 0
    pipe_id: int = 0
    workgroup_id: int = 0
    shader_engine_id: int = 0
    shader_array_id: int = 0
    vm_id: int = 0
    queue_id: int = 0
    microengine_id: int = 0


@dataclass
class TemporalFileInfo:
    """Metadata from a temporal binary file header."""

    path: str
    version: int
    sample_count: int
    start_timestamp: int
    end_timestamp: int
    has_hw_ids: bool


def find_temporal_files(measurements_dir: str) -> List[Path]:
    """Find all temporal-*.bin files in a measurements directory.

    Args:
        measurements_dir: Path to HPCToolkit measurements directory.

    Returns:
        List of paths to temporal binary files, sorted by name.
    """
    mdir = Path(measurements_dir)
    if not mdir.is_dir():
        return []
    files = sorted(mdir.glob("temporal-*.bin"))
    return files


def read_temporal_header(filepath: str) -> Optional[TemporalFileInfo]:
    """Read only the header from a temporal binary file.

    Args:
        filepath: Path to temporal-*.bin file.

    Returns:
        TemporalFileInfo with header metadata, or None if invalid.
    """
    path = Path(filepath)
    if not path.is_file() or path.stat().st_size < HEADER_SIZE:
        return None

    with open(path, "rb") as f:
        data = f.read(HEADER_SIZE)

    magic, version, count, start_ts, end_ts = struct.unpack(HEADER_FMT, data)

    if magic != TEMPORAL_MAGIC:
        return None

    if version not in (TEMPORAL_VERSION_V1, TEMPORAL_VERSION_V2):
        return None

    return TemporalFileInfo(
        path=str(path),
        version=version,
        sample_count=count,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        has_hw_ids=(version == TEMPORAL_VERSION_V2),
    )


def read_temporal_samples(filepath: str) -> Tuple[List[TemporalSample], bool]:
    """Read all samples from a temporal binary file.

    Args:
        filepath: Path to temporal-*.bin file.

    Returns:
        Tuple of (samples list, has_hw_ids bool).
        Returns ([], False) if the file is invalid.
    """
    info = read_temporal_header(filepath)
    if info is None or info.sample_count == 0:
        return [], False

    samples: List[TemporalSample] = []

    with open(filepath, "rb") as f:
        f.seek(HEADER_SIZE)  # Skip header

        if info.version == TEMPORAL_VERSION_V1:
            for _ in range(info.sample_count):
                data = f.read(SAMPLE_V1_SIZE)
                if len(data) < SAMPLE_V1_SIZE:
                    break
                cct_id, ts, stall, inst_type = struct.unpack(SAMPLE_V1_FMT, data)
                samples.append(TemporalSample(
                    cct_node_id=cct_id,
                    timestamp=ts,
                    stall_reason=stall,
                    inst_type=inst_type,
                ))

        elif info.version == TEMPORAL_VERSION_V2:
            for _ in range(info.sample_count):
                data = f.read(SAMPLE_V2_SIZE)
                if len(data) < SAMPLE_V2_SIZE:
                    break
                fields = struct.unpack(SAMPLE_V2_FMT, data)
                samples.append(TemporalSample(
                    cct_node_id=fields[0],
                    timestamp=fields[1],
                    stall_reason=fields[2],
                    inst_type=fields[3],
                    chiplet=fields[4],
                    cu_or_wgp_id=fields[5],
                    simd_id=fields[6],
                    wave_id=fields[7],
                    pipe_id=fields[8],
                    workgroup_id=fields[9],
                    shader_engine_id=fields[10],
                    shader_array_id=fields[11],
                    vm_id=fields[12],
                    queue_id=fields[13],
                    microengine_id=fields[14],
                ))

    return samples, info.has_hw_ids
