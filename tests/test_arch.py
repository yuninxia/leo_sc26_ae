"""Tests for GPU architecture abstraction layer.

Tests Milestone 1 (Architecture Abstraction) and Milestone 2 (Opcode Classification).
"""

import pytest
from leo.arch.base import GPUArchitecture
from leo.arch.nvidia import NVIDIAArchitecture, V100, A100, H100, get_architecture
from leo.arch.amd import AMDArchitecture, MI300, get_amd_architecture
from leo.arch.intel import IntelArchitecture, PonteVecchio, get_intel_architecture


# ============================================================================
# Milestone 1: Architecture Abstraction Layer Tests
# ============================================================================


class TestGPUArchitectureInterface:
    """Test that all architectures implement the abstract interface."""

    @pytest.fixture(params=[V100, A100, H100, MI300, PonteVecchio])
    def arch(self, request):
        return request.param()

    def test_is_gpu_architecture(self, arch):
        """All architecture classes should inherit from GPUArchitecture."""
        assert isinstance(arch, GPUArchitecture)

    def test_has_required_attributes(self, arch):
        """All architectures should have required properties."""
        assert hasattr(arch, "name")
        assert hasattr(arch, "vendor")
        assert hasattr(arch, "inst_size")
        assert hasattr(arch, "sms")
        assert hasattr(arch, "schedulers")
        assert hasattr(arch, "warps_per_sm")
        assert hasattr(arch, "warp_size")
        assert hasattr(arch, "frequency")

    def test_attribute_types(self, arch):
        """Properties should return correct types."""
        assert isinstance(arch.name, str)
        assert isinstance(arch.vendor, str)
        assert isinstance(arch.inst_size, int)
        assert isinstance(arch.sms, int)
        assert isinstance(arch.schedulers, int)
        assert isinstance(arch.warps_per_sm, int)
        assert isinstance(arch.warp_size, int)
        assert isinstance(arch.frequency, float)

    def test_latency_method(self, arch):
        """All architectures should implement latency() method."""
        # Test with a common opcode pattern
        lat = arch.latency("FADD")  # NVIDIA style
        assert isinstance(lat, tuple)
        assert len(lat) == 2
        assert lat[0] > 0
        assert lat[1] >= lat[0]

    def test_issue_method(self, arch):
        """All architectures should implement issue() method."""
        rate = arch.issue("FADD")
        assert isinstance(rate, int)
        assert rate > 0

    def test_classify_opcode_method(self, arch):
        """All architectures should implement classify_opcode() method."""
        result = arch.classify_opcode("FADD")
        assert isinstance(result, str)

    def test_is_memory_op_method(self, arch):
        """All architectures should implement is_memory_op() method."""
        assert isinstance(arch.is_memory_op("LDG"), bool)

    def test_get_memory_type_method(self, arch):
        """All architectures should implement get_memory_type() method."""
        result = arch.get_memory_type("LDG")
        assert isinstance(result, str)

    def test_is_sync_op_method(self, arch):
        """All architectures should implement is_sync_op() method."""
        assert isinstance(arch.is_sync_op("BAR"), bool)


class TestNVIDIAArchitectures:
    """Test NVIDIA-specific architecture properties."""

    def test_v100_properties(self):
        """V100 should have correct NVIDIA properties."""
        arch = V100()
        assert arch.name == "V100"
        assert arch.vendor == "nvidia"
        assert arch.inst_size == 16  # NVIDIA 128-bit instructions
        assert arch.warp_size == 32
        assert arch.sms == 80
        assert arch.schedulers == 4
        assert arch.warps_per_sm == 64
        assert arch.frequency == 1.38

    def test_a100_properties(self):
        """A100 should have correct NVIDIA properties."""
        arch = A100()
        assert arch.name == "A100"
        assert arch.vendor == "nvidia"
        assert arch.inst_size == 16
        assert arch.warp_size == 32
        assert arch.sms == 108
        assert arch.schedulers == 4
        assert arch.warps_per_sm == 64
        assert arch.frequency == 1.41

    def test_h100_properties(self):
        """H100 should have correct NVIDIA properties."""
        arch = H100()
        assert arch.name == "H100"
        assert arch.vendor == "nvidia"
        assert arch.inst_size == 16
        assert arch.warp_size == 32
        assert arch.sms == 132
        assert arch.schedulers == 4
        assert arch.warps_per_sm == 64
        assert arch.frequency == 1.83


class TestAMDArchitectures:
    """Test AMD-specific architecture properties."""

    def test_mi300_properties(self):
        """MI300 should have correct AMD properties."""
        arch = MI300()
        assert arch.name == "MI300"
        assert arch.vendor == "amd"
        assert arch.inst_size == 4
        assert arch.warp_size == 64
        assert arch.wave_size == 64
        assert arch.sms == 304
        assert arch.frequency == 2.10


class TestIntelArchitectures:
    """Test Intel-specific architecture properties."""

    def test_pvc_properties(self):
        """PonteVecchio should have correct Intel properties."""
        arch = PonteVecchio()
        assert arch.name == "PonteVecchio"
        assert arch.vendor == "intel"
        assert arch.inst_size == 16  # Intel 128-bit instructions
        assert arch.warp_size == 16  # Intel SIMD width
        assert arch.simd_width == 16  # Intel terminology
        assert arch.sms == 128  # Xe cores
        assert arch.schedulers == 8
        assert arch.warps_per_sm == 64
        assert arch.frequency == 1.6


class TestArchitectureRegistry:
    """Test architecture lookup functions."""

    @pytest.mark.parametrize("name,expected_class", [
        ("v100", V100),
        ("volta", V100),
        ("sm_70", V100),
        ("a100", A100),
        ("ampere", A100),
        ("sm_80", A100),
        ("h100", H100),
        ("hopper", H100),
        ("sm_90", H100),
        ("gh200", H100),
    ])
    def test_get_nvidia_architecture(self, name, expected_class):
        """get_architecture should return correct NVIDIA class."""
        arch = get_architecture(name)
        assert isinstance(arch, expected_class)

    @pytest.mark.parametrize("name,expected_class", [
        ("mi100", MI300),
        ("gfx908", MI300),
        ("cdna1", MI300),
        ("mi250", MI300),
        ("mi250x", MI300),
        ("gfx90a", MI300),
        ("cdna2", MI300),
        ("mi300", MI300),
        ("mi300a", MI300),
        ("mi300x", MI300),
        ("gfx940", MI300),
        ("cdna3", MI300),
    ])
    def test_get_amd_architecture(self, name, expected_class):
        """get_amd_architecture should return correct AMD class."""
        arch = get_amd_architecture(name)
        assert isinstance(arch, expected_class)

    @pytest.mark.parametrize("name,expected_class", [
        ("pvc", PonteVecchio),
        ("xe_hpc", PonteVecchio),
        ("max1100", PonteVecchio),
        ("pontevecchio", PonteVecchio),
    ])
    def test_get_intel_architecture(self, name, expected_class):
        """get_intel_architecture should return correct Intel class."""
        arch = get_intel_architecture(name)
        assert isinstance(arch, expected_class)

    def test_get_architecture_invalid(self):
        """get_architecture should raise for invalid names."""
        with pytest.raises(ValueError) as exc_info:
            get_architecture("invalid_gpu")
        assert "Unknown architecture" in str(exc_info.value)

    def test_get_amd_architecture_invalid(self):
        """get_amd_architecture should raise for invalid names."""
        with pytest.raises(ValueError) as exc_info:
            get_amd_architecture("invalid_gpu")
        assert "Unknown AMD architecture" in str(exc_info.value)

    def test_get_intel_architecture_invalid(self):
        """get_intel_architecture should raise for invalid names."""
        with pytest.raises(ValueError) as exc_info:
            get_intel_architecture("invalid_gpu")
        assert "Unknown Intel architecture" in str(exc_info.value)

    def test_architecture_case_insensitive(self):
        """Architecture lookup should be case insensitive."""
        assert isinstance(get_architecture("V100"), V100)
        assert isinstance(get_architecture("v100"), V100)
        assert isinstance(get_amd_architecture("MI300"), MI300)
        assert isinstance(get_amd_architecture("mi300"), MI300)
        assert isinstance(get_intel_architecture("PVC"), PonteVecchio)
        assert isinstance(get_intel_architecture("pvc"), PonteVecchio)


# ============================================================================
# Milestone 2: Opcode Classification Tests
# ============================================================================


class TestNVIDIAOpcodeClassification:
    """Test NVIDIA opcode classification."""

    @pytest.fixture
    def arch(self):
        return A100()

    def test_memory_ops(self, arch):
        """Test NVIDIA memory operation detection."""
        assert arch.is_memory_op("LDG") is True
        assert arch.is_memory_op("LDG.E.64") is True
        assert arch.is_memory_op("STG") is True
        assert arch.is_memory_op("LDS") is True
        assert arch.is_memory_op("STS") is True
        assert arch.is_memory_op("LDL") is True
        assert arch.is_memory_op("LDC") is True
        assert arch.is_memory_op("ATOM") is True
        assert arch.is_memory_op("FADD") is False
        assert arch.is_memory_op("IADD") is False

    def test_memory_types(self, arch):
        """Test NVIDIA memory type detection."""
        assert arch.get_memory_type("LDG") == "GLOBAL"
        assert arch.get_memory_type("LDG.E.64") == "GLOBAL"
        assert arch.get_memory_type("STG") == "GLOBAL"
        assert arch.get_memory_type("LDS") == "SHARED"
        assert arch.get_memory_type("STS") == "SHARED"
        assert arch.get_memory_type("LDL") == "LOCAL"
        assert arch.get_memory_type("STL") == "LOCAL"
        assert arch.get_memory_type("LDC") == "CONSTANT"

    def test_sync_ops(self, arch):
        """Test NVIDIA sync operation detection."""
        assert arch.is_sync_op("BAR") is True
        assert arch.is_sync_op("BAR.SYNC") is True
        assert arch.is_sync_op("MEMBAR") is True
        assert arch.is_sync_op("MEMBAR.GL") is True
        assert arch.is_sync_op("WARPSYNC") is True
        assert arch.is_sync_op("BSSY") is True
        assert arch.is_sync_op("BSYNC") is True
        assert arch.is_sync_op("DEPBAR") is True
        assert arch.is_sync_op("FADD") is False
        assert arch.is_sync_op("LDG") is False

    def test_classify_opcode(self, arch):
        """Test NVIDIA opcode classification."""
        assert arch.classify_opcode("FADD") == "FLOAT"
        assert arch.classify_opcode("FMUL") == "FLOAT"
        assert arch.classify_opcode("FFMA") == "FLOAT"
        assert arch.classify_opcode("DADD") == "FLOAT"  # FP64
        assert arch.classify_opcode("HADD") == "FLOAT"  # FP16
        assert arch.classify_opcode("IADD") == "INTEGER"
        assert arch.classify_opcode("IMAD") == "INTEGER"
        assert arch.classify_opcode("SHL") == "INTEGER"
        assert arch.classify_opcode("LOP3") == "INTEGER"
        assert arch.classify_opcode("LDG") == "MEMORY"
        assert arch.classify_opcode("LDG.E.128") == "MEMORY"
        assert arch.classify_opcode("STG") == "MEMORY"
        assert arch.classify_opcode("MUFU.SIN") == "MUFU"
        assert arch.classify_opcode("MUFU.RCP") == "MUFU"
        assert arch.classify_opcode("HMMA") == "TENSOR"
        assert arch.classify_opcode("IMMA") == "TENSOR"
        assert arch.classify_opcode("BRA") == "CONTROL"
        assert arch.classify_opcode("BAR") == "CONTROL"
        assert arch.classify_opcode("CALL") == "CONTROL"
        assert arch.classify_opcode("I2F") == "CONVERT"
        assert arch.classify_opcode("F2I") == "CONVERT"
        assert arch.classify_opcode("MOV") == "MISC"
        assert arch.classify_opcode("SHFL") == "MISC"

    def test_classify_hopper_opcodes(self, arch):
        """Test Hopper-specific opcode classification (WGMMA, DPX, TMA)."""
        # WGMMA (warp-group matrix multiply-accumulate)
        assert arch.classify_opcode("WGMMA") == "WGMMA"
        assert arch.classify_opcode("WGMMA.m64n256k16.f32.f16.f16") == "WGMMA"
        assert arch.classify_opcode("WGMMA.m64n128k16.f16.f16.f16") == "WGMMA"
        # DPX (dynamic programming accelerator)
        assert arch.classify_opcode("VIMNMX") == "DPX"
        assert arch.classify_opcode("VIADDMAX") == "DPX"
        assert arch.classify_opcode("VIADDMIN") == "DPX"
        assert arch.classify_opcode("VIMAX3") == "DPX"
        assert arch.classify_opcode("VIBMAX") == "DPX"
        assert arch.classify_opcode("VIADD3") == "DPX"
        # TMA (tensor memory accelerator)
        assert arch.classify_opcode("UTMALDG") == "TMA"
        assert arch.classify_opcode("USTMG") == "TMA"


class TestAMDOpcodeClassification:
    """Test AMD opcode classification."""

    @pytest.fixture
    def arch(self):
        return MI300()

    def test_memory_ops(self, arch):
        """Test AMD memory operation detection."""
        assert arch.is_memory_op("global_load_dword") is True
        assert arch.is_memory_op("global_store_dword") is True
        assert arch.is_memory_op("ds_read_b32") is True
        assert arch.is_memory_op("ds_write_b32") is True
        assert arch.is_memory_op("flat_load_dword") is True
        assert arch.is_memory_op("buffer_load_dword") is True
        assert arch.is_memory_op("scratch_load_dword") is True
        assert arch.is_memory_op("s_load_dword") is True
        assert arch.is_memory_op("v_add_f32") is False
        assert arch.is_memory_op("s_add_i32") is False

    def test_memory_types(self, arch):
        """Test AMD memory type detection."""
        assert arch.get_memory_type("global_load_dword") == "GLOBAL"
        assert arch.get_memory_type("global_store_dword") == "GLOBAL"
        assert arch.get_memory_type("buffer_load_dword") == "GLOBAL"
        assert arch.get_memory_type("ds_read_b32") == "SHARED"  # LDS = shared
        assert arch.get_memory_type("ds_write_b32") == "SHARED"
        assert arch.get_memory_type("scratch_load_dword") == "LOCAL"
        assert arch.get_memory_type("scratch_store_dword") == "LOCAL"
        assert arch.get_memory_type("s_load_dword") == "CONSTANT"  # SMEM

    def test_sync_ops(self, arch):
        """Test AMD sync operation detection."""
        assert arch.is_sync_op("s_barrier") is True
        assert arch.is_sync_op("s_waitcnt") is True
        assert arch.is_sync_op("s_waitcnt_vscnt") is True
        assert arch.is_sync_op("s_sleep") is True
        assert arch.is_sync_op("v_add_f32") is False
        assert arch.is_sync_op("global_load_dword") is False

    def test_classify_opcode(self, arch):
        """Test AMD opcode classification."""
        # Floating-point
        assert arch.classify_opcode("v_add_f32") == "FLOAT"
        assert arch.classify_opcode("v_mul_f32") == "FLOAT"
        assert arch.classify_opcode("v_fma_f32") == "FLOAT"
        assert arch.classify_opcode("v_add_f64") == "FLOAT"
        assert arch.classify_opcode("v_pk_add_f16") == "FLOAT"

        # Integer
        assert arch.classify_opcode("v_add_i32") == "INTEGER"
        assert arch.classify_opcode("v_add_u32") == "INTEGER"
        assert arch.classify_opcode("v_mul_i32") == "INTEGER"
        assert arch.classify_opcode("v_lshl_b32") == "INTEGER"
        assert arch.classify_opcode("s_add_i32") == "INTEGER"

        # Memory
        assert arch.classify_opcode("global_load_dword") == "MEMORY"
        assert arch.classify_opcode("ds_read_b32") == "MEMORY"

        # Transcendental (MUFU)
        assert arch.classify_opcode("v_rcp_f32") == "MUFU"
        assert arch.classify_opcode("v_rsq_f32") == "MUFU"
        assert arch.classify_opcode("v_sqrt_f32") == "MUFU"
        assert arch.classify_opcode("v_sin_f32") == "MUFU"
        assert arch.classify_opcode("v_cos_f32") == "MUFU"

        # Tensor/Matrix
        assert arch.classify_opcode("mfma_f32_32x32x8f16") == "TENSOR"
        assert arch.classify_opcode("v_mfma_f32_16x16x16f16") == "TENSOR"

        # Control
        assert arch.classify_opcode("s_barrier") == "CONTROL"
        assert arch.classify_opcode("s_waitcnt") == "CONTROL"
        assert arch.classify_opcode("s_branch") == "CONTROL"
        assert arch.classify_opcode("s_endpgm") == "CONTROL"

        # Conversion
        assert arch.classify_opcode("v_cvt_f32_i32") == "CONVERT"
        assert arch.classify_opcode("v_cvt_i32_f32") == "CONVERT"

        # Misc
        assert arch.classify_opcode("v_mov_b32") == "MISC"
        assert arch.classify_opcode("v_readlane_b32") == "MISC"


class TestIntelOpcodeClassification:
    """Test Intel opcode classification."""

    @pytest.fixture
    def arch(self):
        return PonteVecchio()

    def test_memory_ops(self, arch):
        """Test Intel memory operation detection (verified IGA opcodes)."""
        # Verified SEND opcodes from IGA
        assert arch.is_memory_op("send") is True
        assert arch.is_memory_op("sendc") is True
        assert arch.is_memory_op("sends") is True
        assert arch.is_memory_op("sendsc") is True
        # Non-memory ops
        assert arch.is_memory_op("add") is False
        assert arch.is_memory_op("mul") is False
        assert arch.is_memory_op("sync") is False

    def test_memory_types(self, arch):
        """Test Intel memory type detection.

        Note: Intel encodes memory type in SEND message descriptors, not opcodes.
        All SEND ops return "global" as we can't determine type from opcode alone.
        """
        # All SEND variants return "global" (conservative assumption)
        assert arch.get_memory_type("send") == "global"
        assert arch.get_memory_type("sendc") == "global"
        assert arch.get_memory_type("sends") == "global"
        assert arch.get_memory_type("sendsc") == "global"
        # Non-memory ops return "none"
        assert arch.get_memory_type("add") == "none"
        assert arch.get_memory_type("mul") == "none"

    def test_sync_ops(self, arch):
        """Test Intel sync operation detection (verified IGA opcodes)."""
        # Verified sync opcodes from IGA
        assert arch.is_sync_op("sync") is True
        assert arch.is_sync_op("wait") is True
        # Non-sync ops
        assert arch.is_sync_op("add") is False
        assert arch.is_sync_op("send") is False

    def test_classify_opcode(self, arch):
        """Test Intel opcode classification (verified IGA opcodes)."""
        # Memory (SEND-based, verified from IGA)
        assert arch.classify_opcode("send") == "MEMORY"
        assert arch.classify_opcode("sendc") == "MEMORY"
        assert arch.classify_opcode("sends") == "MEMORY"
        assert arch.classify_opcode("sendsc") == "MEMORY"

        # Sync (verified from IGA)
        assert arch.classify_opcode("sync") == "SYNC"
        assert arch.classify_opcode("wait") == "SYNC"

        # Control (verified from IGA)
        assert arch.classify_opcode("jmpi") == "CONTROL"
        assert arch.classify_opcode("call") == "CONTROL"
        assert arch.classify_opcode("calla") == "CONTROL"
        assert arch.classify_opcode("ret") == "CONTROL"
        assert arch.classify_opcode("if") == "CONTROL"
        assert arch.classify_opcode("else") == "CONTROL"
        assert arch.classify_opcode("endif") == "CONTROL"
        assert arch.classify_opcode("while") == "CONTROL"
        assert arch.classify_opcode("break") == "CONTROL"
        assert arch.classify_opcode("cont") == "CONTROL"
        assert arch.classify_opcode("goto") == "CONTROL"
        assert arch.classify_opcode("join") == "CONTROL"
        assert arch.classify_opcode("halt") == "CONTROL"

        # ALU (verified from IGA)
        assert arch.classify_opcode("add") == "ALU"
        assert arch.classify_opcode("addc") == "ALU"
        assert arch.classify_opcode("mul") == "ALU"
        assert arch.classify_opcode("mov") == "ALU"
        assert arch.classify_opcode("movi") == "ALU"
        assert arch.classify_opcode("mad") == "ALU"
        assert arch.classify_opcode("madm") == "ALU"
        assert arch.classify_opcode("and") == "ALU"
        assert arch.classify_opcode("or") == "ALU"
        assert arch.classify_opcode("xor") == "ALU"
        assert arch.classify_opcode("shl") == "ALU"
        assert arch.classify_opcode("shr") == "ALU"
        assert arch.classify_opcode("cmp") == "ALU"
        assert arch.classify_opcode("sel") == "ALU"
        assert arch.classify_opcode("math") == "ALU"
        assert arch.classify_opcode("nop") == "ALU"


class TestLatencyValues:
    """Test that latency values are reasonable."""

    @pytest.fixture(params=[V100(), A100(), H100(), MI300(), PonteVecchio()])
    def arch(self, request):
        return request.param

    def test_integer_latency(self, arch):
        """Integer ops should have low latency."""
        if arch.vendor == "nvidia":
            lat = arch.latency("IADD")
        elif arch.vendor == "amd":
            lat = arch.latency("v_add_i32")
        else:  # intel
            lat = arch.latency("add")
        assert lat[0] >= 1
        assert lat[1] <= 16  # Intel IGC FPU max is 13 (FPU+3*DELTA)

    def test_memory_latency(self, arch):
        """Memory ops should have higher latency range."""
        if arch.vendor == "nvidia":
            lat = arch.latency("LDG")
        elif arch.vendor == "amd":
            lat = arch.latency("global_load_dword")
        else:  # intel
            # Intel uses SEND for all memory ops; type is in descriptor
            lat = arch.latency("send")
        assert lat[0] >= 10
        assert lat[1] >= 100

    def test_shared_memory_faster_than_global(self, arch):
        """Shared memory should have lower latency than global.

        Note: For Intel, memory type is in SEND message descriptor, not opcode.
        We skip this test for Intel since we can't distinguish SLM from global.
        """
        if arch.vendor == "nvidia":
            shared_lat = arch.latency("LDS")
            global_lat = arch.latency("LDG")
            assert shared_lat[1] < global_lat[1]
        elif arch.vendor == "amd":
            shared_lat = arch.latency("ds_read_b32")
            global_lat = arch.latency("global_load_dword")
            assert shared_lat[1] < global_lat[1]
        # Intel: Skip - memory type is in descriptor, not opcode


class TestIssueRates:
    """Test that issue rates are reasonable."""

    @pytest.fixture(params=[V100(), A100(), H100(), MI300(), PonteVecchio()])
    def arch(self, request):
        return request.param

    def test_issue_positive(self, arch):
        """Issue rates should be positive."""
        if arch.vendor == "nvidia":
            rate = arch.issue("FADD")
        elif arch.vendor == "amd":
            rate = arch.issue("v_add_f32")
        else:  # intel
            rate = arch.issue("add")
        assert rate >= 1

    def test_fp64_slower_issue(self, arch):
        """FP64 should have slower issue rate than FP32 on older GPUs."""
        if arch.vendor == "nvidia":
            fp32_rate = arch.issue("FADD")
            fp64_rate = arch.issue("DADD")
        elif arch.vendor == "amd":
            fp32_rate = arch.issue("v_add_f32")
            fp64_rate = arch.issue("v_add_f64")
        else:  # intel - use same opcode, Intel doesn't differentiate by name
            fp32_rate = arch.issue("add")
            fp64_rate = arch.issue("add")  # Intel uses same opcode

        # FP64 should be same or slower
        assert fp64_rate >= fp32_rate


# ============================================================================
# Extended Tests: Atomic Operations, GWS, MFMA Shapes, FP8/BF8
# ============================================================================


class TestAtomicOperations:
    """Test atomic operation detection."""

    def test_nvidia_atomic_ops(self):
        """Test NVIDIA atomic operation detection."""
        arch = A100()
        # NVIDIA atomics
        assert arch.is_atomic_op("ATOM") is True
        assert arch.is_atomic_op("ATOM.ADD") is True
        assert arch.is_atomic_op("ATOM.CAS") is True
        assert arch.is_atomic_op("ATOM.EXCH") is True
        assert arch.is_atomic_op("RED") is True
        assert arch.is_atomic_op("RED.ADD") is True
        # Non-atomics
        assert arch.is_atomic_op("LDG") is False
        assert arch.is_atomic_op("STG") is False
        assert arch.is_atomic_op("FADD") is False

    def test_amd_global_atomic_ops(self):
        """Test AMD global memory atomic detection."""
        arch = MI300()
        # Global atomics
        assert arch.is_atomic_op("global_atomic_add") is True
        assert arch.is_atomic_op("global_atomic_add_f32") is True
        assert arch.is_atomic_op("global_atomic_cmpswap") is True
        assert arch.is_atomic_op("global_atomic_swap") is True
        # Flat atomics
        assert arch.is_atomic_op("flat_atomic_add") is True
        assert arch.is_atomic_op("flat_atomic_cmpswap") is True
        # Buffer atomics
        assert arch.is_atomic_op("buffer_atomic_add") is True
        # Non-atomics
        assert arch.is_atomic_op("global_load_dword") is False
        assert arch.is_atomic_op("global_store_dword") is False

    def test_amd_ds_atomic_ops(self):
        """Test AMD LDS (ds_*) atomic detection."""
        arch = MI300()
        # DS atomics
        assert arch.is_atomic_op("ds_add_u32") is True
        assert arch.is_atomic_op("ds_sub_u32") is True
        assert arch.is_atomic_op("ds_min_i32") is True
        assert arch.is_atomic_op("ds_max_u32") is True
        assert arch.is_atomic_op("ds_and_b32") is True
        assert arch.is_atomic_op("ds_or_b32") is True
        assert arch.is_atomic_op("ds_xor_b32") is True
        assert arch.is_atomic_op("ds_cmpswap_b32") is True
        # Return-value variants
        assert arch.is_atomic_op("ds_add_rtn_u32") is True
        assert arch.is_atomic_op("ds_cmpswap_rtn_b32") is True
        # Non-atomics (regular read/write)
        assert arch.is_atomic_op("ds_read_b32") is False
        assert arch.is_atomic_op("ds_write_b32") is False

    def test_atomic_latency_higher(self):
        """Atomic operations should have higher latency than regular loads."""
        arch = MI300()
        regular_lat = arch.latency("global_load_dword")
        atomic_lat = arch.latency("global_atomic_add")
        assert atomic_lat[0] >= regular_lat[0]
        assert atomic_lat[1] >= regular_lat[1]


class TestAMDGWSOperations:
    """Test AMD GWS (Global Wave Sync) operation detection."""

    def test_gws_detection(self):
        """Test GWS operation detection."""
        arch = MI300()
        # GWS operations
        assert arch.is_gws_op("ds_gws_init") is True
        assert arch.is_gws_op("ds_gws_sema_v") is True
        assert arch.is_gws_op("ds_gws_sema_br") is True
        assert arch.is_gws_op("ds_gws_sema_p") is True
        assert arch.is_gws_op("ds_gws_barrier") is True
        # Non-GWS
        assert arch.is_gws_op("ds_read_b32") is False
        assert arch.is_gws_op("ds_write_b32") is False
        assert arch.is_gws_op("s_barrier") is False

    def test_gws_is_memory_op(self):
        """GWS operations are accessed via ds_ prefix, so they're memory ops."""
        arch = MI300()
        # GWS uses ds_ prefix, so is_memory_op returns True
        assert arch.is_memory_op("ds_gws_init") is True


class TestAMDWaitcntTypes:
    """Test AMD wait counter type detection."""

    def test_waitcnt_type_detection(self):
        """Test waitcnt counter type detection."""
        from leo.arch.amd import WaitcntType

        arch = MI300()

        # Specific counter types
        assert arch.get_waitcnt_type("s_waitcnt_vmcnt") == WaitcntType.VM_CNT
        assert arch.get_waitcnt_type("s_waitcnt_lgkmcnt") == WaitcntType.LGKM_CNT
        assert arch.get_waitcnt_type("s_waitcnt_vscnt") == WaitcntType.VS_CNT
        assert arch.get_waitcnt_type("s_waitcnt_expcnt") == WaitcntType.EXP_CNT

        # Generic waitcnt waits on all
        assert arch.get_waitcnt_type("s_waitcnt") == "all"

        # Non-waitcnt returns None
        assert arch.get_waitcnt_type("s_barrier") is None
        assert arch.get_waitcnt_type("v_add_f32") is None


class TestAMDMFMAShapes:
    """Test AMD MFMA matrix shape extraction."""

    def test_mfma_shape_extraction(self):
        """Test extracting M×N×K shape from MFMA opcodes."""
        arch = MI300()

        # Various MFMA shapes
        assert arch.get_mfma_shape("v_mfma_f32_32x32x8f16") == (32, 32, 8)
        assert arch.get_mfma_shape("v_mfma_f32_16x16x16f16") == (16, 16, 16)
        assert arch.get_mfma_shape("v_mfma_f32_4x4x4f16") == (4, 4, 4)
        assert arch.get_mfma_shape("mfma_f32_32x32x1_2b_f32") == (32, 32, 1)

        # Non-MFMA returns None
        assert arch.get_mfma_shape("v_add_f32") is None
        assert arch.get_mfma_shape("global_load_dword") is None

    def test_mfma_shape_affects_latency(self):
        """Different MFMA shapes should have different latencies."""
        arch = MI300()

        # Larger matrices should have higher latency
        large_lat = arch.latency("v_mfma_f32_32x32x8f16")
        medium_lat = arch.latency("v_mfma_f32_16x16x16f16")
        small_lat = arch.latency("v_mfma_f32_4x4x4f16")

        assert large_lat[0] > medium_lat[0]
        assert medium_lat[0] > small_lat[0]

    def test_mfma_shape_affects_issue(self):
        """Different MFMA shapes should have different issue rates."""
        arch = MI300()

        large_issue = arch.issue("v_mfma_f32_32x32x8f16")
        medium_issue = arch.issue("v_mfma_f32_16x16x16f16")
        small_issue = arch.issue("v_mfma_f32_4x4x4f16")

        # Larger matrices have slower issue rate
        assert large_issue >= medium_issue
        assert medium_issue >= small_issue


class TestAMDPCControlInstructions:
    """Test AMD PC control instruction classification."""

    def test_pc_control_instructions(self):
        """Test PC control instructions are classified as CONTROL."""
        arch = MI300()

        # PC control instructions
        assert arch.classify_opcode("s_getpc_b64") == "CONTROL"
        assert arch.classify_opcode("s_setpc_b64") == "CONTROL"
        assert arch.classify_opcode("s_swappc_b64") == "CONTROL"

        # These are not sync ops
        assert arch.is_sync_op("s_getpc_b64") is False
        assert arch.is_sync_op("s_setpc_b64") is False

    def test_conditional_branch_variants(self):
        """Test conditional branch variants are classified as CONTROL."""
        arch = MI300()

        assert arch.classify_opcode("s_cbranch_scc0") == "CONTROL"
        assert arch.classify_opcode("s_cbranch_scc1") == "CONTROL"
        assert arch.classify_opcode("s_cbranch_vccz") == "CONTROL"
        assert arch.classify_opcode("s_cbranch_execz") == "CONTROL"


class TestAMDFP8BF8Operations:
    """Test AMD FP8/BF8 operation support (CDNA 3)."""

    def test_fp8_convert_classification(self):
        """Test FP8 conversion instructions are classified as CONVERT."""
        arch = MI300()

        assert arch.classify_opcode("v_cvt_pk_fp8_f32") == "CONVERT"
        assert arch.classify_opcode("v_cvt_pk_bf8_f32") == "CONVERT"
        assert arch.classify_opcode("v_cvt_f32_fp8") == "CONVERT"
        assert arch.classify_opcode("v_cvt_f32_bf8") == "CONVERT"

    def test_fp8_mfma_classification(self):
        """Test FP8 MFMA instructions are classified as TENSOR."""
        arch = MI300()

        assert arch.classify_opcode("v_mfma_f32_32x32x16_fp8_fp8") == "TENSOR"
        assert arch.classify_opcode("v_mfma_f32_16x16x32_bf8_bf8") == "TENSOR"

    def test_fp8_mfma_better_latency(self):
        """FP8 MFMA should have better latency than FP16/FP32 MFMA on MI300."""
        arch = MI300()

        fp8_lat = arch.latency("v_mfma_f32_32x32x16_fp8_fp8")
        fp16_lat = arch.latency("v_mfma_f32_32x32x8f16")

        # FP8 should have lower latency
        assert fp8_lat[0] <= fp16_lat[0]

    def test_fp8_convert_latency(self):
        """FP8 conversions should have low latency on MI300."""
        arch = MI300()

        fp8_conv_lat = arch.latency("v_cvt_pk_fp8_f32")
        regular_conv_lat = arch.latency("v_cvt_f32_i32")

        # FP8 conversions are fast on MI300
        assert fp8_conv_lat[0] <= regular_conv_lat[0]


class TestAMDLLVMLatencyValues:
    """Test MI300 latency values against LLVM SISchedule.td.

    Values sourced from llvm/lib/Target/AMDGPU/SISchedule.td:
    - SICommonWriteRes: WriteSALU=1, Write32Bit=1, WriteLDS=5, WriteSMEM=5,
      WriteVMEM=80, WriteBarrier=500, WriteTrans32=4, WriteFloatCvt=4
    - SIDPGFX942 (MI300): WriteDouble=1, WriteIntMul=1
    """

    def test_mi300_int_mul_latency(self):
        """MI300 integer multiply should have latency 1 (SIDPGFX942 WriteIntMul=1)."""
        assert MI300().latency("v_mul_i32") == (1, 1)

    def test_mi300_fp64_latency(self):
        """MI300 FP64 should have latency 1 (SIDPGFX942 WriteDouble=1)."""
        assert MI300().latency("v_add_f64") == (1, 1)

    def test_mi300_fp32_latency(self):
        """MI300 FP32 should have latency 1."""
        assert MI300().latency("v_add_f32") == (1, 1)

    def test_mi300_misc_latency(self):
        """MI300 MISC should have latency 1."""
        assert MI300().latency("v_mov_b32") == (1, 1)

    def test_mi300_lds_min_latency(self):
        """MI300 LDS min should be 5 (WriteLDS)."""
        assert MI300().latency("ds_read_b32")[0] == 5

    def test_mi300_vmem_min_latency(self):
        """MI300 VMEM min should be 80 (WriteVMEM)."""
        assert MI300().latency("global_load_dword")[0] == 80

    def test_mi300_trans_latency(self):
        """MI300 transcendental should have latency 4 (WriteTrans32)."""
        assert MI300().latency("v_rcp_f32") == (4, 4)

    def test_mi300_convert_latency(self):
        """MI300 CVT should have latency 4 (WriteFloatCvt)."""
        assert MI300().latency("v_cvt_f32_i32") == (4, 4)

    def test_mi300_smem_min_latency(self):
        """MI300 SMEM min should be 5 (WriteSMEM)."""
        assert MI300().latency("s_load_dword")[0] == 5

    def test_barrier_max_latency(self):
        """Barrier max latency should be 500 (WriteBarrier)."""
        _, max_lat = MI300().latency("s_barrier")
        assert max_lat == 500


class TestIsAtomicOpInterface:
    """Test is_atomic_op interface exists on all architectures."""

    @pytest.fixture(params=[V100, A100, H100, MI300, PonteVecchio])
    def arch(self, request):
        return request.param()

    def test_is_atomic_op_method_exists(self, arch):
        """All architectures should implement is_atomic_op() method."""
        assert hasattr(arch, "is_atomic_op")
        assert callable(arch.is_atomic_op)

    def test_is_atomic_op_returns_bool(self, arch):
        """is_atomic_op should return a boolean."""
        result = arch.is_atomic_op("some_op")
        assert isinstance(result, bool)


class TestH100LatencyValues:
    """Test H100-specific latency values based on Luo et al. IPDPS 2024."""

    def test_fp16_improved(self):
        """H100 FP16 should be faster than V100."""
        h100 = H100()
        v100 = V100()
        assert h100.latency("HFMA2")[0] < v100.latency("HFMA2")[0]

    def test_tensor_core_latency(self):
        """H100 tensor core latency reflects measured mma values."""
        h100 = H100()
        lat = h100.latency("HMMA")
        assert lat == (16, 24)

    def test_shared_memory_latency(self):
        """H100 shared memory min latency matches measurement (29 cycles)."""
        h100 = H100()
        lat = h100.latency("LDS")
        assert lat[0] == 29

    def test_global_memory_latency(self):
        """H100 global memory latency matches measurement."""
        h100 = H100()
        lat = h100.latency("LDG")
        assert lat[0] == 32   # L1 hit = 32 cycles
        assert lat[1] == 744  # L2 far miss = 744 cycles

    def test_fp64_latency(self):
        """H100 FP64 min latency reduced (doubled FP64 units)."""
        h100 = H100()
        lat = h100.latency("DFMA")
        assert lat == (4, 8)

    def test_issue_rate_int32(self):
        """H100 INT32 issue rate halved (doubled cores)."""
        h100 = H100()
        v100 = V100()
        assert h100.issue("IADD") < v100.issue("IADD")
        assert h100.issue("IADD") == 1

    def test_issue_rate_fp32(self):
        """H100 FP32 issue rate halved (doubled cores)."""
        h100 = H100()
        v100 = V100()
        assert h100.issue("FADD") < v100.issue("FADD")
        assert h100.issue("FADD") == 1

    def test_issue_rate_fp64(self):
        """H100 FP64 issue rate halved (doubled cores)."""
        h100 = H100()
        v100 = V100()
        assert h100.issue("DADD") < v100.issue("DADD")
        assert h100.issue("DADD") == 2

    def test_h100_not_delegating(self):
        """H100 should NOT delegate to A100 anymore."""
        h100 = H100()
        a100 = A100()
        # Tensor core latency differs
        assert h100.latency("HMMA") != a100.latency("HMMA")
        # FP16 latency differs
        assert h100.latency("HFMA2") != a100.latency("HFMA2")
        # Shared memory min differs
        assert h100.latency("LDS")[0] != a100.latency("LDS")[0]


class TestH100HopperInstructions:
    """Tests for Hopper-specific instruction categories (WGMMA, DPX, TMA, DSM)."""

    # --- WGMMA latency/issue ---

    def test_wgmma_latency(self):
        """H100 WGMMA latency covers all N dimensions and modes."""
        h100 = H100()
        assert h100.latency("WGMMA.m64n256k16.f32.f16.f16") == (13, 144)
        assert h100.latency("WGMMA.m64n64k16.f16.f16.f16") == (13, 144)
        assert h100.latency("WGMMA") == (13, 144)

    def test_wgmma_issue_rate(self):
        h100 = H100()
        assert h100.issue("WGMMA.m64n256k16.f32.f16.f16") == 4

    def test_wgmma_higher_latency_than_mma(self):
        """WGMMA max latency should exceed synchronous mma (HMMA)."""
        h100 = H100()
        assert h100.latency("WGMMA")[1] > h100.latency("HMMA")[1]  # 144 > 24

    # --- DPX latency/issue ---

    def test_dpx_latency_16bit(self):
        h100 = H100()
        assert h100.latency("VIADDMAX.S16X2") == (2, 4)
        assert h100.latency("VIBMAX.S16X2") == (2, 4)

    def test_dpx_latency_32bit(self):
        h100 = H100()
        assert h100.latency("VIMNMX") == (2, 10)
        assert h100.latency("VIADDMAX") == (2, 10)

    def test_dpx_issue_rate_16bit(self):
        h100 = H100()
        assert h100.issue("VIADDMAX.S16X2") == 1

    def test_dpx_issue_rate_32bit(self):
        h100 = H100()
        assert h100.issue("VIMNMX") == 4

    def test_dpx_lower_latency_than_tensor(self):
        """DPX should be much faster than tensor core ops."""
        h100 = H100()
        assert h100.latency("VIMNMX")[1] < h100.latency("HMMA")[0]  # 10 < 16

    # --- TMA latency/issue ---

    def test_tma_latency(self):
        h100 = H100()
        assert h100.latency("UTMALDG") == (369, 762)
        assert h100.latency("USTMG") == (369, 762)

    def test_tma_issue_rate(self):
        h100 = H100()
        assert h100.issue("UTMALDG") == 4

    def test_tma_higher_latency_than_global_memory(self):
        """TMA should have higher latency than regular global memory."""
        h100 = H100()
        assert h100.latency("UTMALDG")[0] > h100.latency("LDG")[0]  # 369 > 32

    # --- DSM (distributed shared memory) ---

    def test_dsm_cluster_latency(self):
        """DSM with CLUSTER modifier should have inter-SM latency range."""
        h100 = H100()
        assert h100.latency("LDS.CLUSTER") == (33, 213)

    def test_dsm_regular_shared_unchanged(self):
        """Regular shared memory (no cluster) should be unchanged."""
        h100 = H100()
        assert h100.latency("LDS") == (29, 80)

    # --- V100/A100 fallback ---

    def test_hopper_opcodes_fallback_on_older_gpus(self):
        """WGMMA/DPX/TMA opcodes return default on V100/A100."""
        v100 = V100()
        a100 = A100()
        for gpu in [v100, a100]:
            assert gpu.latency("WGMMA") == (4, 25)
            assert gpu.latency("VIMNMX") == (4, 25)
            assert gpu.latency("UTMALDG") == (4, 25)


class TestPVCIGCLatencyValues:
    """Test PVC latency values against IGC LatencyTable.h XELatencyInfo.

    Values sourced from intel-graphics-compiler/visa/LocalScheduler/LatencyTable.h.
    DPAS formula from LatencyTable.cpp: DPAS + RepeatCount - 1 (PVC).
    """

    def test_pvc_fpu_latency(self):
        """PVC FPU should have latency (6, 13) from IGC FPU_ACC/FPU+DELTA."""
        assert PonteVecchio().latency("add") == (6, 13)

    def test_pvc_math_latency(self):
        """PVC math should have latency (17, 29) from IGC MATH+DELTA_MATH."""
        assert PonteVecchio().latency("math") == (17, 29)

    def test_pvc_dpas_latency(self):
        """PVC DPAS should have latency (21, 28) from IGC DPAS formula."""
        assert PonteVecchio().latency("dpas") == (21, 28)

    def test_pvc_send_min_latency(self):
        """PVC SEND min should be 45 (IGC LSC_UNTYPED_L1)."""
        assert PonteVecchio().latency("send")[0] == 45

    def test_pvc_branch_latency(self):
        """PVC branch should have latency (16, 23) from IGC ARF/BRANCH."""
        assert PonteVecchio().latency("jmpi") == (16, 23)

    def test_pvc_sync_min_latency(self):
        """PVC sync min should be 23 (IGC SLM_FENCE)."""
        assert PonteVecchio().latency("sync")[0] == 23
