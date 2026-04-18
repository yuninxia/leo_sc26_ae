"""Tests for GPU binary parser module.

Tests the parser factory pattern, CUBIN parser, and AMD Code Object parser.
"""

import pytest
from pathlib import Path

# Test imports work
from leo.binary.parser import (
    # Abstract base
    BinaryParser,
    BinarySection,
    BinaryFunction,
    KernelInfo,
    ParserError,
    # Factory
    get_parser,
    register_parser,
    # NVIDIA CUBIN
    CubinParser,
    CubinSection,
    CubinFunction,
    extract_control_from_bytes,
    parse_cubin_control_fields,
    MASK_STALL,
    # AMD Code Object
    CodeObjectParser,
    CodeObjectSection,
    CodeObjectFunction,
    AMDKernelInfo,
)

# Also test backward-compatible imports
from leo.binary import (
    CubinParser as CubinParserBackward,
    CubinSection as CubinSectionBackward,
    CubinFunction as CubinFunctionBackward,
    CodeObjectParser as CodeObjectParserBackward,
    AMDKernelInfo as AMDKernelInfoBackward,
    get_parser as get_parser_backward,
)

# Test old cubin module still works
from leo.binary.cubin import (
    CubinParser as CubinParserLegacy,
    CubinSection as CubinSectionLegacy,
    extract_control_from_bytes as extract_control_from_bytes_legacy,
)


class TestParserImports:
    """Test that all parser imports work correctly."""

    def test_parser_module_imports(self):
        """Verify all expected symbols are importable from leo.binary.parser."""
        assert BinaryParser is not None
        assert BinarySection is not None
        assert BinaryFunction is not None
        assert KernelInfo is not None
        assert ParserError is not None
        assert get_parser is not None
        assert register_parser is not None

    def test_cubin_imports(self):
        """Verify CUBIN parser imports."""
        assert CubinParser is not None
        assert CubinSection is not None
        assert CubinFunction is not None
        assert extract_control_from_bytes is not None
        assert parse_cubin_control_fields is not None
        assert MASK_STALL == 0x00001E0000000000

    def test_codeobj_imports(self):
        """Verify AMD Code Object parser imports."""
        assert CodeObjectParser is not None
        assert CodeObjectSection is not None
        assert CodeObjectFunction is not None
        assert AMDKernelInfo is not None

    def test_backward_compatible_imports(self):
        """Verify backward-compatible imports from leo.binary."""
        assert CubinParserBackward is CubinParser
        assert CubinSectionBackward is CubinSection
        assert CubinFunctionBackward is CubinFunction
        assert CodeObjectParserBackward is CodeObjectParser
        assert AMDKernelInfoBackward is AMDKernelInfo
        assert get_parser_backward is get_parser

    def test_legacy_cubin_module_imports(self):
        """Verify legacy leo.binary.cubin module still works."""
        assert CubinParserLegacy is CubinParser
        assert CubinSectionLegacy is CubinSection
        assert extract_control_from_bytes_legacy is extract_control_from_bytes


class TestParserFactory:
    """Test parser factory function."""

    def test_get_cubin_parser_by_format(self):
        """Test getting CUBIN parser by format name."""
        from leo.binary.parser.base import get_parser_class

        parser_cls = get_parser_class("cubin")
        assert parser_cls == CubinParser

    def test_get_cubin_parser_by_vendor(self):
        """Test getting CUBIN parser by vendor name."""
        # get_parser should map 'nvidia' to 'cubin'
        with pytest.raises(FileNotFoundError):
            get_parser("nvidia", "/nonexistent/path.cubin")

    def test_get_codeobj_parser_by_format(self):
        """Test getting Code Object parser by format name."""
        with pytest.raises(FileNotFoundError):
            get_parser("codeobj", "/nonexistent/path.co")

    def test_get_codeobj_parser_by_vendor(self):
        """Test getting Code Object parser by vendor name."""
        # get_parser should map 'amd' to 'codeobj'
        with pytest.raises(FileNotFoundError):
            get_parser("amd", "/nonexistent/path.co")

    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_parser("invalid", "/nonexistent/path")
        assert "Unknown format" in str(exc_info.value)
        assert "cubin" in str(exc_info.value)  # Should list supported formats


class TestCubinParser:
    """Test CUBIN parser functionality."""

    def test_parser_interface(self):
        """Verify CubinParser has expected interface."""
        assert hasattr(CubinParser, "sections")
        assert hasattr(CubinParser, "functions")
        assert hasattr(CubinParser, "extract_control_at_offset")
        assert hasattr(CubinParser, "populate_instruction_controls")
        assert hasattr(CubinParser, "get_section_for_function")

    def test_parser_properties(self):
        """Test parser format and vendor properties exist."""
        # Can't instantiate without file, but can check class attributes
        assert CubinParser.format.fget is not None
        assert CubinParser.vendor.fget is not None

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            CubinParser("/nonexistent/path/kernel.cubin")

    def test_cubin_section_dataclass(self):
        """Test CubinSection dataclass."""
        section = CubinSection(
            name=".text._Z10testKernel",
            offset=0x1000,
            size=0x200,
            vaddr=0x5000,
            data=b"\x00" * 0x200,
        )
        assert section.name == ".text._Z10testKernel"
        assert section.offset == 0x1000
        assert section.size == 0x200
        assert section.vaddr == 0x5000
        assert len(section.data) == 0x200

    def test_cubin_function_dataclass(self):
        """Test CubinFunction dataclass."""
        func = CubinFunction(
            name="_Z10testKernel",
            offset=0x100,
            size=0x80,
            section=".text._Z10testKernel",
        )
        assert func.name == "_Z10testKernel"
        assert func.offset == 0x100
        assert func.size == 0x80
        assert func.section == ".text._Z10testKernel"


class TestCodeObjectParser:
    """Test AMD Code Object parser functionality."""

    def test_msgpack_available(self):
        """Verify msgpack is installed."""
        import msgpack

        assert hasattr(msgpack, "unpackb")
        assert hasattr(msgpack, "packb")

    def test_parser_interface(self):
        """Verify CodeObjectParser has expected interface."""
        assert hasattr(CodeObjectParser, "sections")
        assert hasattr(CodeObjectParser, "functions")
        assert hasattr(CodeObjectParser, "kernels")
        assert hasattr(CodeObjectParser, "get_kernel_info")
        assert hasattr(CodeObjectParser, "get_all_kernel_info")
        assert hasattr(CodeObjectParser, "get_section_for_function")
        assert hasattr(CodeObjectParser, "get_gpu_arch")
        assert hasattr(CodeObjectParser, "metadata")
        assert hasattr(CodeObjectParser, "target")

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            CodeObjectParser("/nonexistent/path/kernel.co")

    def test_amd_kernel_info_dataclass(self):
        """Test AMDKernelInfo dataclass."""
        info = AMDKernelInfo(
            name="testKernel",
            symbol="testKernel.kd",
            vgpr_count=64,
            sgpr_count=32,
            agpr_count=16,
            vgpr_spill_count=2,
            sgpr_spill_count=1,
            shared_mem_size=2048,  # LDS
            local_mem_size=128,  # Scratch
            wavefront_size=64,
            max_flat_workgroup_size=256,
            kernarg_segment_size=56,
        )
        assert info.name == "testKernel"
        assert info.symbol == "testKernel.kd"
        assert info.vgpr_count == 64
        assert info.sgpr_count == 32
        assert info.agpr_count == 16
        assert info.vgpr_spill_count == 2
        assert info.sgpr_spill_count == 1
        assert info.shared_mem_size == 2048
        assert info.local_mem_size == 128
        assert info.wavefront_size == 64
        assert info.max_flat_workgroup_size == 256
        assert info.kernarg_segment_size == 56

    def test_codeobj_section_dataclass(self):
        """Test CodeObjectSection dataclass."""
        section = CodeObjectSection(
            name=".text._Z10testKernel",
            offset=0x1000,
            size=0x200,
            vaddr=0x5000,
            data=b"\x00" * 0x200,
        )
        assert section.name == ".text._Z10testKernel"
        assert section.offset == 0x1000
        assert section.size == 0x200
        assert section.vaddr == 0x5000

    def test_codeobj_function_dataclass(self):
        """Test CodeObjectFunction dataclass."""
        func = CodeObjectFunction(
            name="_Z10testKernel",
            offset=0x100,
            size=0x80,
            section=".text._Z10testKernel",
        )
        assert func.name == "_Z10testKernel"
        assert func.offset == 0x100
        assert func.size == 0x80


class TestMsgpackMetadataParsing:
    """Test MessagePack metadata parsing functionality."""

    def test_msgpack_encode_decode(self):
        """Test basic msgpack encoding and decoding."""
        import msgpack

        # Sample AMD metadata structure
        metadata = {
            "amdhsa.version": [1, 2],
            "amdhsa.target": "amdgcn-amd-amdhsa--gfx90a",
            "amdhsa.kernels": [
                {
                    ".name": "testKernel",
                    ".symbol": "testKernel.kd",
                    ".vgpr_count": 64,
                    ".sgpr_count": 32,
                    ".group_segment_fixed_size": 2048,
                    ".private_segment_fixed_size": 128,
                    ".wavefront_size": 64,
                    ".max_flat_workgroup_size": 256,
                }
            ],
        }

        # Encode and decode
        packed = msgpack.packb(metadata)
        unpacked = msgpack.unpackb(packed, raw=False)

        assert unpacked["amdhsa.version"] == [1, 2]
        assert unpacked["amdhsa.target"] == "amdgcn-amd-amdhsa--gfx90a"
        assert len(unpacked["amdhsa.kernels"]) == 1
        kernel = unpacked["amdhsa.kernels"][0]
        assert kernel[".name"] == "testKernel"
        assert kernel[".vgpr_count"] == 64
        assert kernel[".sgpr_count"] == 32

    def test_decode_with_safety_limits(self):
        """Test msgpack decoding with safety limits."""
        import msgpack

        metadata = {"key": "value" * 100}
        packed = msgpack.packb(metadata)

        # Should work with reasonable limits
        result = msgpack.unpackb(
            packed,
            raw=False,
            max_str_len=10000000,
            max_array_len=100000,
            max_map_len=50000,
        )
        assert "key" in result


class TestControlFieldExtraction:
    """Test NVIDIA control field extraction."""

    def test_extract_control_from_bytes_basic(self):
        """Test extracting control fields from instruction bytes."""
        from leo.binary.parser.cubin import (
            extract_control_from_bytes,
            SHIFT_STALL,
            SHIFT_YIELD,
            SHIFT_WRITE,
            SHIFT_READ,
            SHIFT_WAIT,
            SHIFT_REUSE,
        )

        # Create instruction bytes with known control fields
        # Stall = 5 (bits 44:41), yield = 1 (bit 45)
        stall_value = 5
        yield_value = 1
        bits = (stall_value << SHIFT_STALL) | (yield_value << SHIFT_YIELD)
        instruction_bytes = bits.to_bytes(8, byteorder="little")

        control = extract_control_from_bytes(instruction_bytes)
        assert control.stall == stall_value
        assert control.yield_flag == yield_value

    def test_extract_control_empty_bytes(self):
        """Test control extraction with insufficient bytes."""
        control = extract_control_from_bytes(b"")
        assert control.stall == 1  # Default value

    def test_extract_control_short_bytes(self):
        """Test control extraction with too few bytes."""
        control = extract_control_from_bytes(b"\x00\x00\x00\x00")  # Only 4 bytes
        assert control.stall == 1  # Default value


class TestKernelInfoBase:
    """Test KernelInfo base class."""

    def test_kernel_info_dataclass(self):
        """Test KernelInfo dataclass."""
        info = KernelInfo(
            name="testKernel",
            vgpr_count=64,
            sgpr_count=32,
            shared_mem_size=2048,
            local_mem_size=128,
        )
        assert info.name == "testKernel"
        assert info.vgpr_count == 64
        assert info.sgpr_count == 32
        assert info.shared_mem_size == 2048
        assert info.local_mem_size == 128
        assert info.extra == {}

    def test_kernel_info_extra_fields(self):
        """Test KernelInfo with extra fields."""
        info = KernelInfo(
            name="testKernel",
            extra={"custom_field": 123, "another": "value"},
        )
        assert info.extra["custom_field"] == 123
        assert info.extra["another"] == "value"


# Fixture for finding test GPU binaries
@pytest.fixture
def gpubin_path():
    """Find a test GPU binary file."""
    # Look for gpubin files in test data
    test_dirs = [
        Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-measurements/gpubins",
        Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-measurements/gpubins-used",
    ]
    for dir_path in test_dirs:
        if dir_path.exists():
            gpubins = list(dir_path.glob("*.gpubin"))
            if gpubins:
                return gpubins[0]
    pytest.skip("No test GPU binary files found")


class TestCubinParserWithRealFile:
    """Integration tests with real GPU binary files."""

    @pytest.mark.skipif(
        not (Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-measurements/gpubins").exists(),
        reason="Test GPU binary not available",
    )
    def test_parse_real_gpubin(self, gpubin_path):
        """Test parsing real gpubin file."""
        parser = CubinParser(gpubin_path)

        # Should have at least one section
        assert len(parser.sections) >= 1
        assert any(name.startswith(".text") for name in parser.sections.keys())

        # Should have at least one function
        assert len(parser.functions) >= 1

        # Check format and vendor properties
        assert parser.format == "cubin"
        assert parser.vendor == "nvidia"

    @pytest.mark.skipif(
        not (Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-measurements/gpubins").exists(),
        reason="Test GPU binary not available",
    )
    def test_extract_control_from_real_gpubin(self, gpubin_path):
        """Test control field extraction from real gpubin."""
        parser = CubinParser(gpubin_path)
        section = list(parser.sections.values())[0]

        # Extract control for first instruction
        ctrl = parser.extract_control_at_offset(0, section.data)

        # Validate field ranges
        assert 0 <= ctrl.stall <= 15  # 4 bits
        assert 0 <= ctrl.yield_flag <= 1  # 1 bit
        assert 0 <= ctrl.write <= 7  # 3 bits
        assert 0 <= ctrl.read <= 7  # 3 bits
        assert 0 <= ctrl.wait <= 63  # 6 bits
        assert 0 <= ctrl.reuse <= 15  # 4 bits


class TestParserCoexistence:
    """Test that NVIDIA and AMD parsers can coexist."""

    def test_both_parsers_importable(self):
        """Both NVIDIA and AMD parsers can be imported together."""
        from leo.binary.parser import CubinParser, CodeObjectParser

        assert CubinParser is not None
        assert CodeObjectParser is not None

    def test_parsers_have_distinct_formats(self):
        """Each parser reports its own format."""
        # We can't instantiate without files, but we can check the registry
        from leo.binary.parser.base import _PARSERS

        assert "cubin" in _PARSERS
        assert "codeobj" in _PARSERS
        assert _PARSERS["cubin"] == CubinParser
        assert _PARSERS["codeobj"] == CodeObjectParser

    def test_factory_returns_correct_parser(self):
        """Factory returns correct parser type for each format."""
        from leo.binary.parser.base import get_parser_class

        assert get_parser_class("cubin") == CubinParser
        assert get_parser_class("codeobj") == CodeObjectParser
        assert get_parser_class("nvidia") == CubinParser
        assert get_parser_class("amd") == CodeObjectParser
