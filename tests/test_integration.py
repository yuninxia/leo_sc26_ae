"""Integration tests for multi-vendor GPU support.

Tests the complete M5 integration including:
- Unified get_architecture() function
- Vendor-aware disassembler and parser selection
- AMD and NVIDIA architecture support coexistence
"""

import pytest
from pathlib import Path

from leo.arch import (
    GPUArchitecture,
    get_architecture,
    get_vendor,
    NVIDIAArchitecture,
    AMDArchitecture,
    V100,
    A100,
    MI300,
)
from leo.analyzer import AnalysisConfig


class TestUnifiedGetArchitecture:
    """Test the unified get_architecture() function."""

    @pytest.mark.parametrize(
        "name,expected_class,expected_vendor",
        [
            # NVIDIA architectures
            ("v100", V100, "nvidia"),
            ("volta", V100, "nvidia"),
            ("sm_70", V100, "nvidia"),
            ("a100", A100, "nvidia"),
            ("ampere", A100, "nvidia"),
            ("sm_80", A100, "nvidia"),
            # AMD architectures
            ("mi100", MI300, "amd"),
            ("gfx908", MI300, "amd"),
            ("cdna1", MI300, "amd"),
            ("mi250", MI300, "amd"),
            ("mi250x", MI300, "amd"),
            ("gfx90a", MI300, "amd"),
            ("cdna2", MI300, "amd"),
            ("mi300", MI300, "amd"),
            ("mi300a", MI300, "amd"),
            ("mi300x", MI300, "amd"),
            ("gfx940", MI300, "amd"),
            ("gfx941", MI300, "amd"),
            ("gfx942", MI300, "amd"),
            ("cdna3", MI300, "amd"),
        ],
    )
    def test_architecture_lookup(self, name, expected_class, expected_vendor):
        """Test that all supported architectures can be looked up."""
        arch = get_architecture(name)
        assert isinstance(arch, expected_class)
        assert arch.vendor == expected_vendor
        assert isinstance(arch, GPUArchitecture)

    @pytest.mark.parametrize("name", ["v100", "V100", "VOLTA", "A100", "a100"])
    def test_case_insensitive(self, name):
        """Test that architecture names are case-insensitive."""
        arch = get_architecture(name)
        assert arch is not None

    def test_unknown_nvidia_architecture(self):
        """Test error message for unknown NVIDIA-like architecture."""
        with pytest.raises(ValueError) as exc_info:
            get_architecture("sm_99")
        # Error includes list of supported architectures
        assert "Unknown architecture" in str(exc_info.value)
        assert "NVIDIA:" in str(exc_info.value)

    def test_unknown_amd_architecture(self):
        """Test error message for unknown AMD-like architecture."""
        with pytest.raises(ValueError) as exc_info:
            get_architecture("gfx999")
        # Error includes list of supported architectures
        assert "Unknown architecture" in str(exc_info.value)
        assert "AMD:" in str(exc_info.value)

    def test_unknown_vendor_architecture(self):
        """Test error message for completely unknown architecture."""
        with pytest.raises(ValueError) as exc_info:
            get_architecture("xyz123")
        assert "Unknown architecture" in str(exc_info.value)
        assert "NVIDIA:" in str(exc_info.value)
        assert "AMD:" in str(exc_info.value)


class TestGetVendor:
    """Test the get_vendor() function."""

    @pytest.mark.parametrize(
        "name,expected_vendor",
        [
            ("a100", "nvidia"),
            ("v100", "nvidia"),
            ("sm_80", "nvidia"),
            ("mi250", "amd"),
            ("gfx90a", "amd"),
            ("cdna3", "amd"),
        ],
    )
    def test_vendor_detection(self, name, expected_vendor):
        """Test that vendor is correctly detected from architecture name."""
        vendor = get_vendor(name)
        assert vendor == expected_vendor


class TestArchitectureProperties:
    """Test architecture-specific properties."""

    def test_nvidia_inst_size(self):
        """NVIDIA uses 16-byte instructions."""
        arch = get_architecture("a100")
        assert arch.inst_size == 16

    def test_amd_inst_size(self):
        """AMD uses 4-byte instructions."""
        arch = get_architecture("mi250")
        assert arch.inst_size == 4

    def test_nvidia_warp_size(self):
        """NVIDIA uses 32-thread warps."""
        arch = get_architecture("a100")
        assert arch.warp_size == 32

    def test_amd_wave_size(self):
        """AMD uses 64-thread wavefronts."""
        arch = get_architecture("mi250")
        assert arch.warp_size == 64  # Aliased to wave_size

    def test_latency_returns_tuple(self):
        """Latency method returns (min, max) tuple."""
        for name in ["a100", "mi250"]:
            arch = get_architecture(name)
            lat = arch.latency("FADD")
            assert isinstance(lat, tuple)
            assert len(lat) == 2
            assert lat[0] <= lat[1]


class TestAnalysisConfigVendorDetection:
    """Test AnalysisConfig vendor auto-detection."""

    def test_vendor_from_nvidia_arch(self):
        """Vendor detected from NVIDIA architecture name."""
        config = AnalysisConfig(
            db_path="/tmp/test", gpubin_path="/tmp/test.gpubin", arch="a100"
        )
        assert config.vendor == "nvidia"

    def test_vendor_from_amd_arch(self):
        """Vendor detected from AMD architecture name."""
        config = AnalysisConfig(
            db_path="/tmp/test", gpubin_path="/tmp/test.co", arch="mi250"
        )
        assert config.vendor == "amd"

    def test_vendor_from_file_extension_co(self):
        """Vendor detected from .co file extension when arch unknown."""
        config = AnalysisConfig(
            db_path="/tmp/test", gpubin_path="/tmp/kernel.co", arch="unknown"
        )
        assert config.vendor == "amd"

    def test_vendor_from_file_extension_hsaco(self):
        """Vendor detected from .hsaco file extension when arch unknown."""
        config = AnalysisConfig(
            db_path="/tmp/test", gpubin_path="/tmp/kernel.hsaco", arch="unknown"
        )
        assert config.vendor == "amd"

    def test_vendor_explicit_override(self):
        """Explicit vendor overrides auto-detection."""
        config = AnalysisConfig(
            db_path="/tmp/test",
            gpubin_path="/tmp/test.co",
            arch="mi250",
            vendor="nvidia",  # Explicit override
        )
        assert config.vendor == "nvidia"


class TestMultiVendorCoexistence:
    """Test that both vendors can coexist without conflicts."""

    def test_import_both_vendors(self):
        """Both NVIDIA and AMD classes can be imported together."""
        from leo.arch import V100, A100, MI300

        assert V100 is not None
        assert A100 is not None
        assert MI300 is not None

    def test_instantiate_both_vendors(self):
        """Both NVIDIA and AMD architectures can be instantiated."""
        nvidia = get_architecture("a100")
        amd = get_architecture("mi250")

        assert nvidia.vendor == "nvidia"
        assert amd.vendor == "amd"
        assert nvidia.name != amd.name

    def test_disassembler_imports(self):
        """Both disassemblers can be imported."""
        from leo.binary.disasm import get_disassembler, NVIDIADisassembler, AMDDisassembler

        assert NVIDIADisassembler is not None
        assert AMDDisassembler is not None

    def test_parser_imports(self):
        """Both parsers can be imported."""
        from leo.binary.parser import get_parser, CubinParser, CodeObjectParser

        assert CubinParser is not None
        assert CodeObjectParser is not None


class TestAMDTestData:
    """Test with actual AMD test data if available."""

    @pytest.fixture
    def amd_db_path(self):
        """Path to AMD HPCToolkit test database."""
        path = Path(__file__).parent / "data/pc/amd/hpctoolkit-single.hipoffload.amdclang.rocmgpu-database"
        if not path.exists():
            pytest.skip("AMD test database not available")
        return path

    @pytest.fixture
    def amd_gpubin_path(self):
        """Path to AMD GPU binary."""
        gpubin = Path(__file__).parent / "data/pc/amd/hpctoolkit-single.hipoffload.amdclang.rocmgpu-measurements/gpubins/9f7f9be695af6f36f2b56450611127c6.gpubin"
        if not gpubin.exists():
            pytest.skip("AMD GPU binary not available")
        return gpubin

    @pytest.fixture
    def amd_large_gpubin_path(self):
        """Path to larger AMD GPU binary with multiple kernels."""
        # Use the same gpubin for now - can be updated if a larger binary is added
        gpubin = Path(__file__).parent / "data/pc/amd/hpctoolkit-single.hipoffload.amdclang.rocmgpu-measurements/gpubins/9f7f9be695af6f36f2b56450611127c6.gpubin"
        if not gpubin.exists():
            pytest.skip("AMD GPU binary not available")
        return gpubin

    def test_amd_database_exists(self, amd_db_path):
        """Verify AMD test database structure."""
        assert (amd_db_path / "cct.db").exists()
        assert (amd_db_path / "meta.db").exists()
        assert (amd_db_path / "profile.db").exists()

    def test_amd_gpubin_is_elf(self, amd_gpubin_path):
        """Verify AMD GPU binary is valid ELF."""
        with open(amd_gpubin_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x7fELF", "GPU binary should be ELF format"

    def test_amd_config_creation(self, amd_db_path, amd_gpubin_path):
        """Test creating AnalysisConfig for AMD."""
        config = AnalysisConfig(
            db_path=str(amd_db_path),
            gpubin_path=str(amd_gpubin_path),
            arch="mi300",  # MI300 family (gfx942)
        )
        assert config.vendor == "amd"
        assert config.arch == "mi300"

    def test_amd_code_object_parser(self, amd_large_gpubin_path):
        """Test AMD CodeObjectParser with real GPU binary."""
        from leo.binary.parser import get_parser

        parser = get_parser("amd", str(amd_large_gpubin_path))

        # Verify parser properties
        assert parser.format == "codeobj"
        assert parser.vendor == "amd"
        assert "gfx" in parser.target  # e.g., gfx942 for MI300, gfx908 for MI100

        # Should have sections
        assert len(parser.sections) >= 1
        assert ".text" in parser.sections

        # Should have functions
        assert len(parser.functions) >= 1

        # Should have kernel metadata
        assert len(parser.kernels) >= 1
        for name, info in parser.kernels.items():
            assert info.vgpr_count > 0
            assert info.sgpr_count > 0
            assert info.wavefront_size in [32, 64]

    def test_amd_disassembler(self, amd_gpubin_path):
        """Test AMD disassembler with real GPU binary."""
        from leo.binary.disasm import get_disassembler

        disasm = get_disassembler("amd")
        assert disasm.check_available(), "llvm-objdump not available"

        # Disassemble the binary
        output = disasm.disassemble(str(amd_gpubin_path))
        assert len(output) > 0, "Disassembly should produce output"

        # Parse functions
        functions = disasm.parse_all_functions(output)
        assert len(functions) >= 1, "Should parse at least one function"

        # Verify function structure
        func = functions[0]
        assert func.name, "Function should have a name"
        assert len(func.instructions) > 0, "Function should have instructions"

        # Verify instruction structure
        inst = func.instructions[0]
        assert inst.pc >= 0, "Instruction should have valid PC"
        assert inst.op, "Instruction should have opcode"

    def test_amd_full_analysis_pipeline(self, amd_db_path, amd_gpubin_path):
        """Test full analysis pipeline with AMD GPU binary."""
        from leo.analyzer import AnalysisConfig, KernelAnalyzer

        config = AnalysisConfig(
            db_path=str(amd_db_path),
            gpubin_path=str(amd_gpubin_path),
            arch="mi300",  # MI300 family (gfx942)
        )

        analyzer = KernelAnalyzer(config)
        result = analyzer.analyze()

        # Pipeline should complete without errors
        assert result is not None

        # Should have parsed instructions
        assert len(result.instructions) > 0

        # CFG should be built
        assert result.cfg is not None

        # VMA map should exist (even if no profile matches)
        assert result.vma_map is not None

        # Stats should be populated
        assert result.stats["num_instructions"] > 0
        assert result.stats["num_functions"] >= 1

        # Summary should be generatable
        summary = result.summary()
        assert "Leo GPU Performance Analysis" in summary


class TestNVIDIABackwardCompatibility:
    """Ensure NVIDIA analysis still works after AMD support added."""

    def test_nvidia_config_unchanged(self):
        """NVIDIA config creation unchanged."""
        config = AnalysisConfig(
            db_path="/tmp/test",
            gpubin_path="/tmp/kernel.cubin",
            arch="a100",
        )
        assert config.vendor == "nvidia"
        assert config.arch == "a100"

    def test_nvidia_default_arch(self):
        """Default arch is still a100 (NVIDIA)."""
        config = AnalysisConfig(
            db_path="/tmp/test",
            gpubin_path="/tmp/kernel.cubin",
        )
        assert config.arch == "a100"
        assert config.vendor == "nvidia"
