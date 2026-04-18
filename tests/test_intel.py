"""Comprehensive tests for Intel GPU support in Leo.

Tests cover all Intel-specific components:
- Architecture: PonteVecchio latencies and opcode classification
- Parser: Intel zebin (ELF) format parsing
- Disassembler: GED library wrapper
- Metrics: Intel stall metric mapping
- Integration: Full analysis pipeline

Test Data Paths:
- Database: tests/data/pc/intel/hpctoolkit-single.sycloffload.icpx.intelgpu-database/
- GPU binary: tests/data/pc/intel/hpctoolkit-single.sycloffload.icpx.intelgpu-measurements/gpubins/73a6cf2fd4844baf239d0e2c13911c34.gpubin
"""

import pytest
from pathlib import Path

from leo.arch import get_architecture, get_vendor, GPUArchitecture
from leo.arch.intel import (
    IntelArchitecture,
    PonteVecchio,
    INTEL_MEMORY_OPCODES,
    INTEL_SYNC_OPCODES,
    INTEL_CONTROL_OPCODES,
    INTEL_ALU_OPCODES,
)
from leo.analysis.metrics import (
    INTEL_METRICS,
    get_mem_dep_stall,
    get_memory_stall_cycles,
    get_exec_dep_stall,
)
from leo.utils.vendor import detect_vendor_from_arch_name


# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data/pc/intel"
INTEL_DB_PATH = TEST_DATA_DIR / "hpctoolkit-single.sycloffload.icpx.intelgpu-database"
INTEL_GPUBIN_PATH = (
    TEST_DATA_DIR
    / "hpctoolkit-single.sycloffload.icpx.intelgpu-measurements"
    / "gpubins"
    / "73a6cf2fd4844baf239d0e2c13911c34.gpubin"
)


# =============================================================================
# Vendor Detection Tests
# =============================================================================


class TestIntelVendorDetection:
    """Test vendor detection for Intel architecture names."""

    @pytest.mark.parametrize(
        "arch_name,expected_vendor",
        [
            ("pvc", "intel"),
            ("PVC", "intel"),
            ("pontevecchio", "intel"),
            ("PonteVecchio", "intel"),
            ("xe_hpc", "intel"),
            ("XE_HPC", "intel"),
            ("max1100", "intel"),
            ("MAX1100", "intel"),
        ],
    )
    def test_detect_intel_from_arch_name(self, arch_name, expected_vendor):
        """Test that Intel architecture names are correctly detected."""
        vendor = detect_vendor_from_arch_name(arch_name, default=None)
        assert vendor == expected_vendor

    def test_get_vendor_function(self):
        """Test get_vendor() returns 'intel' for Intel architectures."""
        assert get_vendor("pvc") == "intel"
        assert get_vendor("xe_hpc") == "intel"
        assert get_vendor("pontevecchio") == "intel"


# =============================================================================
# Architecture Tests
# =============================================================================


class TestIntelArchitecture:
    """Test Intel GPU architecture classes."""

    def test_get_architecture_pvc(self):
        """Test getting PonteVecchio architecture by various names."""
        for name in ["pvc", "pontevecchio", "xe_hpc", "max1100"]:
            arch = get_architecture(name)
            assert isinstance(arch, PonteVecchio)
            assert arch.vendor == "intel"
            assert arch.name == "PonteVecchio"

    def test_architecture_properties(self):
        """Test basic architecture properties."""
        arch = get_architecture("pvc")
        assert arch.inst_size == 16  # 16-byte instructions (uncompacted)
        assert arch.warp_size == 16  # SIMD width
        assert arch.sms == 128  # Xe cores
        assert arch.schedulers == 8  # Vector engines per Xe core
        assert arch.warps_per_sm == 64
        assert arch.frequency == 1.6

    def test_simd_width_alias(self):
        """Test simd_width property (Intel terminology)."""
        arch = get_architecture("pvc")
        assert arch.simd_width == arch.warp_size == 16

    def test_latency_memory(self):
        """Test latency for memory operations (IGC LSC_UNTYPED_L1 to HBM)."""
        arch = get_architecture("pvc")
        min_lat, max_lat = arch.latency("send")
        assert min_lat == 45   # IGC LSC_UNTYPED_L1
        assert max_lat == 500  # HBM worst-case

    def test_latency_alu(self):
        """Test latency for ALU operations (IGC FPU_ACC to FPU+DELTA)."""
        arch = get_architecture("pvc")
        min_lat, max_lat = arch.latency("add")
        assert min_lat == 6   # IGC FPU_ACC
        assert max_lat == 13  # IGC FPU + 3*DELTA

    def test_latency_math(self):
        """Test latency for math (transcendental) operations."""
        arch = get_architecture("pvc")
        min_lat, max_lat = arch.latency("math")
        assert min_lat == 17  # IGC MATH
        assert max_lat == 29  # IGC MATH + 3*DELTA_MATH

    def test_latency_dpas(self):
        """Test latency for DPAS (matrix) operations."""
        arch = get_architecture("pvc")
        min_lat, max_lat = arch.latency("dpas")
        assert min_lat == 21  # IGC DPAS base (RepeatCount=1)
        assert max_lat == 28  # PVC: DPAS + RepeatCount=8 - 1

    def test_latency_sync(self):
        """Test latency for sync operations (IGC SLM_FENCE to TLP)."""
        arch = get_architecture("pvc")
        min_lat, max_lat = arch.latency("sync")
        assert min_lat == 23  # IGC SLM_FENCE
        assert max_lat == 100  # Conservative TLP upper bound

    def test_latency_control(self):
        """Test latency for control flow operations (IGC ARF/BRANCH)."""
        arch = get_architecture("pvc")
        min_lat, max_lat = arch.latency("jmpi")
        assert min_lat == 16  # IGC ARF
        assert max_lat == 23  # IGC BRANCH

    def test_issue_rate(self):
        """Test instruction issue rates."""
        arch = get_architecture("pvc")
        assert arch.issue("add") == 1
        assert arch.issue("send") >= 1


class TestOpcodeClassification:
    """Test Intel opcode classification."""

    def test_classify_memory_opcodes(self):
        """Test that memory opcodes are classified correctly."""
        arch = get_architecture("pvc")
        for opcode in INTEL_MEMORY_OPCODES:
            assert arch.classify_opcode(opcode) == "MEMORY"
            assert arch.is_memory_op(opcode) is True

    def test_classify_sync_opcodes(self):
        """Test that sync opcodes are classified correctly."""
        arch = get_architecture("pvc")
        for opcode in INTEL_SYNC_OPCODES:
            assert arch.classify_opcode(opcode) == "SYNC"
            assert arch.is_sync_op(opcode) is True

    def test_classify_control_opcodes(self):
        """Test that control flow opcodes are classified correctly."""
        arch = get_architecture("pvc")
        for opcode in INTEL_CONTROL_OPCODES:
            assert arch.classify_opcode(opcode) == "CONTROL"

    def test_classify_alu_opcodes(self):
        """Test that ALU opcodes are classified correctly."""
        arch = get_architecture("pvc")
        for opcode in INTEL_ALU_OPCODES:
            if opcode not in INTEL_MEMORY_OPCODES | INTEL_SYNC_OPCODES | INTEL_CONTROL_OPCODES:
                assert arch.classify_opcode(opcode) == "ALU"

    def test_case_insensitive(self):
        """Test that opcode classification is case-insensitive."""
        arch = get_architecture("pvc")
        assert arch.classify_opcode("SEND") == "MEMORY"
        assert arch.classify_opcode("Send") == "MEMORY"
        assert arch.classify_opcode("send") == "MEMORY"


# =============================================================================
# Metrics Tests
# =============================================================================


class TestIntelMetrics:
    """Test Intel-specific metric handling."""

    def test_intel_metrics_dictionary(self):
        """Test that INTEL_METRICS contains expected keys."""
        # Intel uses standard HPCToolkit metric names (mem, gmem, idep, sync)
        expected_keys = {"exec_dep", "mem_dep", "mem_dep_alt", "sync", "total_stall"}
        assert expected_keys.issubset(set(INTEL_METRICS.keys()))

    def test_get_mem_dep_stall_intel(self):
        """Test memory dependency stall lookup for Intel."""
        # Intel uses gcycles:stl:mem as primary, gcycles:stl:gmem as fallback
        metrics = {
            "gcycles:stl:mem": 1000.0,
            "gcycles:stl:gmem": 500.0,  # Fallback - not used if mem is present
        }
        result = get_mem_dep_stall(metrics, vendor="intel")
        assert result == 1000.0

    def test_get_memory_stall_cycles_intel(self):
        """Test total memory stall cycles for Intel."""
        # Intel uses gcycles:stl:mem + gcycles:stl:lmem
        metrics = {
            "gcycles:stl:mem": 2000.0,
            "gcycles:stl:lmem": 100.0,
        }
        result = get_memory_stall_cycles(metrics, vendor="intel")
        assert result == 2100.0  # mem + lmem

    def test_get_exec_dep_stall(self):
        """Test execution dependency stall lookup."""
        metrics = {
            "gcycles:stl:idep": 500.0,
            "gcycles:stl:sbid": 300.0,
        }
        # Note: get_exec_dep_stall uses idep for all vendors currently
        result = get_exec_dep_stall(metrics, vendor="intel")
        assert result == 500.0


# =============================================================================
# Parser Tests
# =============================================================================


class TestZebinParser:
    """Test Intel zebin parser."""

    @pytest.fixture
    def intel_gpubin_path(self):
        """Path to Intel GPU binary."""
        if not INTEL_GPUBIN_PATH.exists():
            pytest.skip("Intel GPU binary not available")
        return INTEL_GPUBIN_PATH

    def test_parser_import(self):
        """Test that ZebinParser can be imported."""
        from leo.binary.parser import ZebinParser, get_parser

        assert ZebinParser is not None

    def test_parser_factory(self, intel_gpubin_path):
        """Test get_parser returns ZebinParser for Intel."""
        from leo.binary.parser import get_parser

        parser = get_parser("intel", str(intel_gpubin_path))
        assert parser.vendor == "intel"
        assert parser.format == "zebin"

    def test_parse_sections(self, intel_gpubin_path):
        """Test parsing sections from Intel zebin."""
        from leo.binary.parser import ZebinParser

        parser = ZebinParser(intel_gpubin_path)
        assert len(parser.sections) >= 1

        # Should have at least one .text section
        text_sections = [s for s in parser.sections if ".text" in s]
        assert len(text_sections) >= 1

    def test_parse_functions(self, intel_gpubin_path):
        """Test parsing function symbols."""
        from leo.binary.parser import ZebinParser

        parser = ZebinParser(intel_gpubin_path)
        # May have functions if symbol table is present
        # Just verify it doesn't crash
        assert isinstance(parser.functions, dict)

    def test_gpubin_is_elf(self, intel_gpubin_path):
        """Verify Intel GPU binary is valid ELF."""
        with open(intel_gpubin_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x7fELF", "GPU binary should be ELF format"


# =============================================================================
# Disassembler Tests
# =============================================================================


class TestIntelDisassembler:
    """Test Intel GED disassembler."""

    @pytest.fixture
    def intel_gpubin_path(self):
        """Path to Intel GPU binary."""
        if not INTEL_GPUBIN_PATH.exists():
            pytest.skip("Intel GPU binary not available")
        return INTEL_GPUBIN_PATH

    def test_disassembler_import(self):
        """Test that IntelDisassembler can be imported."""
        from leo.binary.disasm import IntelDisassembler, get_disassembler

        assert IntelDisassembler is not None

    def test_get_disassembler(self):
        """Test get_disassembler returns IntelDisassembler for Intel."""
        from leo.binary.disasm import get_disassembler

        disasm = get_disassembler("intel")
        assert disasm.vendor == "intel"
        assert disasm.tool_name == "ged"

    def test_check_available(self):
        """Test availability check (may skip if GED not available)."""
        from leo.binary.disasm import IntelDisassembler

        disasm = IntelDisassembler()
        # Just verify the method exists and returns a boolean
        result = disasm.check_available()
        assert isinstance(result, bool)

    def test_disassemble(self, intel_gpubin_path):
        """Test disassembling Intel GPU binary."""
        from leo.binary.disasm import IntelDisassembler

        disasm = IntelDisassembler()
        if not disasm.check_available():
            pytest.skip("GED library not available")

        output = disasm.disassemble(str(intel_gpubin_path))
        assert len(output) > 0

    def test_disassemble_and_parse_all(self, intel_gpubin_path):
        """Test full disassembly and parsing pipeline."""
        from leo.binary.disasm import IntelDisassembler

        disasm = IntelDisassembler()
        if not disasm.check_available():
            pytest.skip("GED library not available")

        functions = disasm.disassemble_and_parse_all(str(intel_gpubin_path))
        assert len(functions) >= 1
        assert all(len(f.instructions) > 0 for f in functions)


# =============================================================================
# AnalysisConfig Tests
# =============================================================================


class TestIntelAnalysisConfig:
    """Test AnalysisConfig with Intel GPU settings."""

    def test_config_from_arch_name(self):
        """Test vendor detection from Intel architecture name."""
        from leo.analyzer import AnalysisConfig

        config = AnalysisConfig(
            db_path="/tmp/test",
            gpubin_path="/tmp/test.gpubin",
            arch="pvc",
        )
        assert config.vendor == "intel"

    def test_config_from_zebin_extension(self):
        """Test vendor detection from .zebin file extension."""
        from leo.analyzer import AnalysisConfig

        config = AnalysisConfig(
            db_path="/tmp/test",
            gpubin_path="/tmp/kernel.zebin",
            arch="unknown",
        )
        assert config.vendor == "intel"

    def test_config_explicit_vendor(self):
        """Test explicit vendor override."""
        from leo.analyzer import AnalysisConfig

        config = AnalysisConfig(
            db_path="/tmp/test",
            gpubin_path="/tmp/test.gpubin",
            arch="pvc",
            vendor="intel",
        )
        assert config.vendor == "intel"


# =============================================================================
# VMA Property Tests
# =============================================================================


class TestIntelVMAProperty:
    """Test VMAProperty with Intel stall metrics."""

    def test_intel_stall_breakdown(self):
        """Test stall breakdown for Intel vendor."""
        from leo.analysis.vma_property import VMAProperty

        # Intel uses standard HPCToolkit metric names
        metrics = {
            "gcycles:stl:mem": 1000.0,
            "gcycles:stl:idep": 500.0,
            "gcycles:stl:sync": 200.0,
            "gcycles:stl:pipe": 100.0,
        }
        prop = VMAProperty(
            vma=0x100,
            prof_metrics=metrics,
            vendor="intel",
            has_profile_data=True,
        )

        breakdown = prop.get_stall_breakdown()
        assert "gcycles:stl:mem" in breakdown
        assert "gcycles:stl:idep" in breakdown
        assert breakdown["gcycles:stl:mem"] == 1000.0

    def test_memory_stall_cycles_intel(self):
        """Test memory_stall_cycles property for Intel."""
        from leo.analysis.vma_property import VMAProperty

        # Intel uses gcycles:stl:mem + gcycles:stl:lmem
        metrics = {
            "gcycles:stl:mem": 2000.0,
            "gcycles:stl:lmem": 100.0,
        }
        prop = VMAProperty(
            vma=0x100,
            prof_metrics=metrics,
            vendor="intel",
            has_profile_data=True,
        )

        # Intel uses mem + lmem for memory stall cycles
        assert prop.memory_stall_cycles == 2100.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntelIntegration:
    """Integration tests for Intel GPU support."""

    @pytest.fixture
    def intel_db_path(self):
        """Path to Intel HPCToolkit database."""
        if not INTEL_DB_PATH.exists():
            pytest.skip("Intel test database not available")
        return INTEL_DB_PATH

    @pytest.fixture
    def intel_gpubin_path(self):
        """Path to Intel GPU binary."""
        if not INTEL_GPUBIN_PATH.exists():
            pytest.skip("Intel GPU binary not available")
        return INTEL_GPUBIN_PATH

    def test_database_exists(self, intel_db_path):
        """Verify Intel test database structure."""
        assert (intel_db_path / "cct.db").exists()
        assert (intel_db_path / "meta.db").exists()
        assert (intel_db_path / "profile.db").exists()

    def test_imports_work(self):
        """Test that all Intel-related imports work."""
        from leo.arch import PonteVecchio, get_architecture, IntelArchitecture
        from leo.binary.parser import ZebinParser
        from leo.binary.disasm import IntelDisassembler
        from leo.analysis.metrics import INTEL_METRICS

        assert PonteVecchio is not None
        assert ZebinParser is not None
        assert IntelDisassembler is not None
        assert len(INTEL_METRICS) > 0

    def test_full_parser_pipeline(self, intel_gpubin_path):
        """Test parsing Intel binary with both parser and disassembler."""
        from leo.binary.parser import get_parser
        from leo.binary.disasm import get_disassembler

        # Parse binary
        parser = get_parser("intel", str(intel_gpubin_path))
        assert len(parser.sections) >= 1

        # Get disassembler
        disasm = get_disassembler("intel")
        assert disasm.vendor == "intel"

    def test_analyzer_creation(self, intel_db_path, intel_gpubin_path):
        """Test creating KernelAnalyzer for Intel GPU."""
        from leo.analyzer import AnalysisConfig, KernelAnalyzer
        from leo.binary.disasm import get_disassembler

        # Check if GED is available first
        disasm = get_disassembler("intel")
        if not disasm.check_available():
            pytest.skip("GED library not available for full analysis")

        config = AnalysisConfig(
            db_path=str(intel_db_path),
            gpubin_path=str(intel_gpubin_path),
            arch="pvc",
        )

        # Should create without errors
        analyzer = KernelAnalyzer(config)
        assert analyzer.config.vendor == "intel"

    def test_full_analysis_pipeline(self, intel_db_path, intel_gpubin_path):
        """Test complete analysis pipeline with Intel GPU binary."""
        from leo.binary.disasm import get_disassembler

        disasm = get_disassembler("intel")
        if not disasm.check_available():
            pytest.skip("GED library not available for full analysis")

        from leo.analyzer import AnalysisConfig, KernelAnalyzer

        config = AnalysisConfig(
            db_path=str(intel_db_path),
            gpubin_path=str(intel_gpubin_path),
            arch="pvc",
        )

        analyzer = KernelAnalyzer(config)
        result = analyzer.analyze()

        # Verify results
        assert result is not None
        assert len(result.instructions) > 0
        assert result.cfg is not None
        assert result.vma_map is not None
        assert result.stats["num_instructions"] > 0

        # Verify summary can be generated
        summary = result.summary()
        assert "Leo GPU Performance Analysis" in summary


# =============================================================================
# Multi-Vendor Coexistence Tests
# =============================================================================


class TestMultiVendorCoexistence:
    """Test that Intel support coexists with NVIDIA and AMD."""

    def test_all_vendors_importable(self):
        """All vendor architectures can be imported together."""
        from leo.arch import V100, A100, MI300, PonteVecchio

        assert V100 is not None
        assert A100 is not None
        assert MI300 is not None
        assert PonteVecchio is not None

    def test_all_vendors_instantiable(self):
        """All vendor architectures can be instantiated."""
        nvidia = get_architecture("a100")
        amd = get_architecture("mi250")
        intel = get_architecture("pvc")

        assert nvidia.vendor == "nvidia"
        assert amd.vendor == "amd"
        assert intel.vendor == "intel"

    def test_all_parsers_importable(self):
        """All vendor parsers can be imported."""
        from leo.binary.parser import CubinParser, CodeObjectParser, ZebinParser

        assert CubinParser is not None
        assert CodeObjectParser is not None
        assert ZebinParser is not None

    def test_all_disassemblers_importable(self):
        """All vendor disassemblers can be imported."""
        from leo.binary.disasm import NVIDIADisassembler, AMDDisassembler, IntelDisassembler

        assert NVIDIADisassembler is not None
        assert AMDDisassembler is not None
        assert IntelDisassembler is not None

    def test_get_architecture_all_vendors(self):
        """get_architecture works for all vendors."""
        assert get_architecture("a100").vendor == "nvidia"
        assert get_architecture("mi250").vendor == "amd"
        assert get_architecture("pvc").vendor == "intel"

    def test_vendor_detection_all(self):
        """Vendor detection works for all vendors."""
        assert detect_vendor_from_arch_name("a100") == "nvidia"
        assert detect_vendor_from_arch_name("mi250") == "amd"
        assert detect_vendor_from_arch_name("pvc") == "intel"
