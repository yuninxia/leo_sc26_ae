"""Tests for GPU disassembler abstractions.

Tests NVIDIA (nvdisasm), AMD (llvm-objdump), and Intel (GED) disassemblers.
"""

import pytest

from leo.binary.disasm import (
    AMDDisassembler,
    Disassembler,
    DisassemblerError,
    IntelDisassembler,
    NVIDIADisassembler,
    ParsedFunction,
    ParsedInstruction,
    get_disassembler,
)
from leo.binary.disasm.amd import parse_waitcnt
from leo.binary.instruction import InstructionStat, PredicateFlag


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestDisassemblerFactory:
    """Test disassembler factory function."""

    def test_get_nvidia_disassembler(self):
        """Test getting NVIDIA disassembler."""
        disasm = get_disassembler("nvidia")
        assert isinstance(disasm, NVIDIADisassembler)
        assert disasm.vendor == "nvidia"
        assert disasm.tool_name == "nvdisasm"

    def test_get_amd_disassembler(self):
        """Test getting AMD disassembler."""
        disasm = get_disassembler("amd")
        assert isinstance(disasm, AMDDisassembler)
        assert disasm.vendor == "amd"
        assert disasm.tool_name == "llvm-objdump"

    def test_get_intel_disassembler(self):
        """Test getting Intel disassembler."""
        disasm = get_disassembler("intel")
        assert isinstance(disasm, IntelDisassembler)
        assert disasm.vendor == "intel"
        assert disasm.tool_name == "ged"

    def test_case_insensitive(self):
        """Test factory is case-insensitive."""
        assert isinstance(get_disassembler("NVIDIA"), NVIDIADisassembler)
        assert isinstance(get_disassembler("AMD"), AMDDisassembler)
        assert isinstance(get_disassembler("Intel"), IntelDisassembler)
        assert isinstance(get_disassembler("Nvidia"), NVIDIADisassembler)

    def test_invalid_vendor(self):
        """Test invalid vendor raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vendor"):
            get_disassembler("invalid")

        with pytest.raises(ValueError, match="Unknown vendor"):
            get_disassembler("arm")


# =============================================================================
# NVIDIA Disassembler Tests
# =============================================================================


class TestNVIDIADisassembler:
    """Test NVIDIA disassembler functionality."""

    @pytest.fixture
    def disasm(self):
        return NVIDIADisassembler()

    def test_is_disassembler_subclass(self, disasm):
        """Verify NVIDIADisassembler is a Disassembler."""
        assert isinstance(disasm, Disassembler)

    def test_parse_simple_instruction(self, disasm):
        """Test parsing a simple instruction line."""
        line = "        /*0050*/                   FADD R2, R0, R1 ;"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x50
        assert parsed.opcode == "FADD"
        assert parsed.operands == ["R2", "R0", "R1"]
        assert parsed.predicate is None

    def test_parse_instruction_with_predicate(self, disasm):
        """Test parsing instruction with predicate guard."""
        line = "        /*0060*/                   @P0 FADD R2, R0, R1 ;"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x60
        assert parsed.opcode == "FADD"
        assert parsed.predicate == "P0"

    def test_parse_instruction_with_negated_predicate(self, disasm):
        """Test parsing instruction with negated predicate."""
        line = "        /*0070*/                   @!P1 BRA `(.L_x_1) ;"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.predicate == "!P1"
        assert parsed.opcode == "BRA"

    def test_parse_memory_instruction(self, disasm):
        """Test parsing memory instruction with brackets."""
        line = "        /*0080*/                   LDG.E.64 R2, [R4] ;"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.opcode == "LDG.E.64"
        assert parsed.operands == ["R2", "[R4]"]

    def test_parse_instruction_with_constant(self, disasm):
        """Test parsing instruction with constant memory reference."""
        line = "        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.opcode == "IMAD.MOV.U32"
        assert "c[0x0][0x28]" in parsed.operands

    def test_skip_non_instruction_lines(self, disasm):
        """Test that non-instruction lines return None."""
        assert disasm.parse_instruction_line("") is None
        assert disasm.parse_instruction_line("  .section .text.foo,") is None
        assert disasm.parse_instruction_line("  // comment") is None
        assert disasm.parse_instruction_line(".L_x_0:") is None

    def test_skip_data_directives(self, disasm):
        """Test that data directives are skipped."""
        line = "        /*0100*/                   .byte 0x00, 0x00 ;"
        assert disasm.parse_instruction_line(line) is None

    def test_parsed_to_instruction_stat(self, disasm):
        """Test conversion to InstructionStat."""
        line = "        /*0050*/                   FADD R2, R0, R1 ;"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert isinstance(inst, InstructionStat)
        assert inst.op == "FADD"
        assert inst.pc == 0x50
        assert 2 in inst.dsts  # R2 is destination
        assert 0 in inst.srcs  # R0 is source
        assert 1 in inst.srcs  # R1 is source

    def test_parsed_to_instruction_stat_predicated(self, disasm):
        """Test conversion with predicate."""
        line = "        /*0060*/                   @P0 FADD R2, R0, R1 ;"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.predicate == 0
        assert inst.predicate_flag == PredicateFlag.PREDICATE_TRUE
        assert 0 in inst.psrcs  # P0 is a source (guard)

    def test_parsed_to_instruction_stat_negated_predicate(self, disasm):
        """Test conversion with negated predicate."""
        line = "        /*0070*/                   @!P1 BRA `(.L_x_1) ;"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.predicate == 1
        assert inst.predicate_flag == PredicateFlag.PREDICATE_FALSE

    def test_parsed_to_instruction_stat_store(self, disasm):
        """Test that store instructions treat all operands as sources."""
        line = "        /*0090*/                   STG.E [R4], R2 ;"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.op == "STG.E"
        # Both R4 (address) and R2 (data) should be sources
        assert 4 in inst.srcs
        assert 2 in inst.srcs
        assert len(inst.dsts) == 0

    def test_parsed_to_instruction_stat_branch_target(self, disasm):
        """Test branch target extraction."""
        line = "        /*0070*/                   BRA `(.L_x_1) ;"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.branch_target == ".L_x_1"

    def test_parse_function_output(self, disasm):
        """Test parsing a complete function from nvdisasm output."""
        output = """
.section .text._Z4testPf,"ax",@progbits
.L_x_0:
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   ULDC.64 UR4, c[0x0][0x118] ;
        /*0020*/                   LDG.E R0, [R1.64] ;
        /*0030*/                   FADD R2, R0, R0 ;
        /*0040*/                   STG.E [R1.64], R2 ;
        /*0050*/                   EXIT ;
.section .nv.info._Z4testPf,
"""
        func = disasm.parse_function(output)

        assert func is not None
        assert func.name == "_Z4testPf"
        assert len(func.instructions) == 6
        assert func.instructions[0].op == "IMAD.MOV.U32"
        assert func.instructions[5].op == "EXIT"
        assert ".L_x_0" in func.labels
        assert func.labels[".L_x_0"] == 0

    def test_parse_all_functions(self, disasm):
        """Test parsing multiple functions."""
        output = """
.section .text._Z5func1v,"ax",@progbits
        /*0000*/                   NOP ;
        /*0010*/                   EXIT ;
.section .text._Z5func2v,"ax",@progbits
        /*0000*/                   NOP ;
        /*0010*/                   NOP ;
        /*0020*/                   EXIT ;
.section .nv.info,
"""
        functions = disasm.parse_all_functions(output)

        assert len(functions) == 2
        assert functions[0].name == "_Z5func1v"
        assert len(functions[0].instructions) == 2
        assert functions[1].name == "_Z5func2v"
        assert len(functions[1].instructions) == 3


# =============================================================================
# AMD Disassembler Tests
# =============================================================================


class TestAMDDisassembler:
    """Test AMD disassembler functionality."""

    @pytest.fixture
    def disasm(self):
        return AMDDisassembler()

    def test_is_disassembler_subclass(self, disasm):
        """Verify AMDDisassembler is a Disassembler."""
        assert isinstance(disasm, Disassembler)

    def test_gpu_arch_configurable(self):
        """Test GPU architecture is configurable."""
        disasm_mi250 = AMDDisassembler(gpu_arch="gfx90a")
        disasm_mi300 = AMDDisassembler(gpu_arch="gfx942")
        assert disasm_mi250._gpu_arch == "gfx90a"
        assert disasm_mi300._gpu_arch == "gfx942"

    def test_parse_instruction_comment_format(self, disasm):
        """Test parsing instruction in comment format."""
        line = "s_mov_b32 m0, 0x10000 // 000000000100: BEFC00FF 00010000"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x100
        assert parsed.opcode == "s_mov_b32"
        assert "m0" in parsed.operands
        assert "0x10000" in parsed.operands

    def test_parse_instruction_addr_format(self, disasm):
        """Test parsing instruction in address format."""
        line = "       4:	7e000280        	v_mov_b32_e32 v0, 0"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 4
        assert parsed.opcode == "v_mov_b32_e32"
        assert "v0" in parsed.operands

    def test_parse_vgpr_single(self, disasm):
        """Test parsing single VGPR."""
        assert disasm._parse_vgpr("v0") == [0]
        assert disasm._parse_vgpr("v255") == [255]
        assert disasm._parse_vgpr("V10") == [10]  # Case insensitive

    def test_parse_vgpr_range(self, disasm):
        """Test parsing VGPR range."""
        assert disasm._parse_vgpr("v[0:1]") == [0, 1]
        assert disasm._parse_vgpr("v[0:3]") == [0, 1, 2, 3]
        assert disasm._parse_vgpr("v[4:5]") == [4, 5]

    def test_parse_sgpr_single(self, disasm):
        """Test parsing single SGPR."""
        assert disasm._parse_sgpr("s0") == [0]
        assert disasm._parse_sgpr("s103") == [103]
        assert disasm._parse_sgpr("S10") == [10]  # Case insensitive

    def test_parse_sgpr_range(self, disasm):
        """Test parsing SGPR range."""
        assert disasm._parse_sgpr("s[0:1]") == [0, 1]
        assert disasm._parse_sgpr("s[0:3]") == [0, 1, 2, 3]
        assert disasm._parse_sgpr("s[4:5]") == [4, 5]

    def test_parse_accvgpr(self, disasm):
        """Test parsing AccVGPR (CDNA)."""
        assert disasm._parse_accvgpr("a0") == [0]
        assert disasm._parse_accvgpr("a255") == [255]
        assert disasm._parse_accvgpr("acc0") == [0]
        assert disasm._parse_accvgpr("a[0:3]") == [0, 1, 2, 3]

    def test_is_special_register(self, disasm):
        """Test special register detection."""
        assert disasm._is_special_register("exec") is True
        assert disasm._is_special_register("exec_lo") is True
        assert disasm._is_special_register("vcc") is True
        assert disasm._is_special_register("scc") is True
        assert disasm._is_special_register("m0") is True
        assert disasm._is_special_register("v0") is False
        assert disasm._is_special_register("s0") is False

    def test_classify_memory_ops(self, disasm):
        """Test classification of memory operations."""
        assert disasm._classify_amd_opcode("global_load_dword") == "MEMORY"
        assert disasm._classify_amd_opcode("global_store_dword") == "MEMORY"
        assert disasm._classify_amd_opcode("ds_read_b32") == "MEMORY"
        assert disasm._classify_amd_opcode("ds_write_b32") == "MEMORY"
        assert disasm._classify_amd_opcode("flat_load_dword") == "MEMORY"
        assert disasm._classify_amd_opcode("buffer_load_dword") == "MEMORY"
        assert disasm._classify_amd_opcode("s_load_dword") == "MEMORY"

    def test_classify_float_ops(self, disasm):
        """Test classification of floating-point operations."""
        assert disasm._classify_amd_opcode("v_add_f32") == "FLOAT"
        assert disasm._classify_amd_opcode("v_mul_f32") == "FLOAT"
        assert disasm._classify_amd_opcode("v_fma_f32") == "FLOAT"
        assert disasm._classify_amd_opcode("v_rcp_f32") == "FLOAT"

    def test_classify_integer_ops(self, disasm):
        """Test classification of integer operations."""
        assert disasm._classify_amd_opcode("v_add_i32") == "INTEGER"
        assert disasm._classify_amd_opcode("v_add_u32") == "INTEGER"
        assert disasm._classify_amd_opcode("v_mul_i32") == "INTEGER"
        assert disasm._classify_amd_opcode("v_and_b32") == "INTEGER"

    def test_classify_control_ops(self, disasm):
        """Test classification of control flow operations."""
        assert disasm._classify_amd_opcode("s_barrier") == "CONTROL"
        assert disasm._classify_amd_opcode("s_waitcnt") == "CONTROL"
        assert disasm._classify_amd_opcode("s_branch") == "CONTROL"
        assert disasm._classify_amd_opcode("s_cbranch_execz") == "CONTROL"
        assert disasm._classify_amd_opcode("s_endpgm") == "CONTROL"

    def test_classify_tensor_ops(self, disasm):
        """Test classification of tensor/matrix operations."""
        assert disasm._classify_amd_opcode("v_mfma_f32_32x32x1f32") == "TENSOR"
        assert disasm._classify_amd_opcode("v_mfma_f32_16x16x4f16") == "TENSOR"

    def test_classify_convert_ops(self, disasm):
        """Test classification of conversion operations."""
        assert disasm._classify_amd_opcode("v_cvt_f32_i32") == "CONVERT"
        assert disasm._classify_amd_opcode("v_cvt_i32_f32") == "CONVERT"

    def test_is_store_op(self, disasm):
        """Test store operation detection."""
        assert disasm._is_store_op("global_store_dword") is True
        assert disasm._is_store_op("ds_write_b32") is True
        assert disasm._is_store_op("flat_store_dword") is True
        assert disasm._is_store_op("global_load_dword") is False
        assert disasm._is_store_op("v_add_f32") is False

    def test_is_branch_op(self, disasm):
        """Test branch operation detection."""
        assert disasm._is_branch_op("s_branch") is True
        assert disasm._is_branch_op("s_cbranch_execz") is True
        assert disasm._is_branch_op("s_call_b64") is True
        assert disasm._is_branch_op("s_return") is True
        assert disasm._is_branch_op("v_add_f32") is False

    def test_parsed_to_instruction_stat_valu(self, disasm):
        """Test conversion of VALU instruction to InstructionStat."""
        line = "v_add_f32 v0, v1, v2 // 000000000100: 02000503"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert isinstance(inst, InstructionStat)
        assert inst.op == "v_add_f32"
        assert inst.pc == 0x100
        assert 0 in inst.dsts  # v0 is destination
        assert 1 in inst.srcs  # v1 is source
        assert 2 in inst.srcs  # v2 is source

    def test_parsed_to_instruction_stat_salu(self, disasm):
        """Test conversion of SALU instruction with SGPRs."""
        line = "s_add_u32 s0, s1, s2 // 000000000100: 80000201"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.op == "s_add_u32"
        assert 0 in inst.udsts  # s0 is destination (SGPR -> uniform)
        assert 1 in inst.usrcs  # s1 is source
        assert 2 in inst.usrcs  # s2 is source

    def test_parsed_to_instruction_stat_store(self, disasm):
        """Test that store instructions treat all operands as sources."""
        line = "global_store_dword v0, v1, s[0:1] // 000000000100: DC700000"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        # All registers should be sources for stores
        assert 0 in inst.srcs  # v0 (address)
        assert 1 in inst.srcs  # v1 (data)
        assert 0 in inst.usrcs  # s0 (descriptor)
        assert 1 in inst.usrcs  # s1 (descriptor)
        assert len(inst.dsts) == 0

    def test_parsed_to_instruction_stat_no_predicate(self, disasm):
        """Test AMD instructions have no explicit predicate."""
        line = "v_add_f32 v0, v1, v2 // 000000000100: 02000503"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.predicate == -1
        assert inst.predicate_flag == PredicateFlag.PREDICATE_NONE

    def test_parse_function_output(self, disasm):
        """Test parsing a complete function from llvm-objdump output."""
        output = """
0000000000001000 <_Z10testKernelPfS_i>:
       1000:	7e000280        	v_mov_b32_e32 v0, 0
       1004:	be8c00ff 00010000	s_mov_b32 m0, 0x10000
       100c:	d81a0000 00000100	ds_write_b32 v0, v1
       1014:	bf8c007f        	s_waitcnt lgkmcnt(0)
       1018:	bf810000        	s_endpgm
"""
        func = disasm.parse_function(output)

        assert func is not None
        assert func.name == "_Z10testKernelPfS_i"
        assert len(func.instructions) == 5

    def test_parse_all_functions(self, disasm):
        """Test parsing multiple functions."""
        output = """
0000000000001000 <kernel1>:
       1000:	7e000280        	v_mov_b32_e32 v0, 0
       1004:	bf810000        	s_endpgm

0000000000002000 <kernel2>:
       2000:	7e000280        	v_mov_b32_e32 v0, 0
       2004:	7e020281        	v_mov_b32_e32 v1, 1
       2008:	bf810000        	s_endpgm
"""
        functions = disasm.parse_all_functions(output)

        assert len(functions) == 2
        assert functions[0].name == "kernel1"
        assert len(functions[0].instructions) == 2
        assert functions[1].name == "kernel2"
        assert len(functions[1].instructions) == 3


# =============================================================================
# AMD Wait Counter Parsing Tests
# =============================================================================


class TestAMDWaitcntParsing:
    """Test s_waitcnt operand parsing."""

    def test_parse_vmcnt(self):
        """Test parsing vmcnt."""
        result = parse_waitcnt("vmcnt(0)")
        assert result == {"vmcnt": 0}

        result = parse_waitcnt("vmcnt(15)")
        assert result == {"vmcnt": 15}

    def test_parse_lgkmcnt(self):
        """Test parsing lgkmcnt."""
        result = parse_waitcnt("lgkmcnt(0)")
        assert result == {"lgkmcnt": 0}

    def test_parse_combined(self):
        """Test parsing combined wait counts."""
        result = parse_waitcnt("vmcnt(0) lgkmcnt(0)")
        assert result == {"vmcnt": 0, "lgkmcnt": 0}

        result = parse_waitcnt("vmcnt(1) lgkmcnt(2) expcnt(3)")
        assert result == {"vmcnt": 1, "lgkmcnt": 2, "expcnt": 3}

    def test_parse_vscnt(self):
        """Test parsing vscnt (RDNA2+)."""
        result = parse_waitcnt("vscnt(0)")
        assert result == {"vscnt": 0}

    def test_parse_numeric(self):
        """Test parsing numeric (combined mask) format."""
        result = parse_waitcnt("0")
        assert result == {"combined": 0}

        result = parse_waitcnt("0x7f")
        assert result == {"combined": 0x7f}


# =============================================================================
# Intel Disassembler Tests (Text Round-Trip Path)
# =============================================================================


class TestIntelDisassemblerTextPath:
    """Test Intel disassembler text-path operand handling.

    The Intel disassembler encodes GRF register operands in the text
    representation (dsts=N,M srcs=N,M) so they survive the
    disassemble() -> parse_instruction_line() -> parsed_to_instruction_stat()
    round-trip.
    """

    @pytest.fixture
    def disasm(self):
        return IntelDisassembler()

    def test_parse_instruction_with_dsts_and_srcs(self, disasm):
        """Test parsing instruction line with both dsts and srcs."""
        line = "  /*0010*/ add dsts=4 srcs=2,3"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x10
        assert parsed.opcode == "add"
        assert "dsts=4" in parsed.operands
        assert "srcs=2,3" in parsed.operands

    def test_parse_instruction_with_dsts_only(self, disasm):
        """Test parsing instruction with only destination registers."""
        line = "  /*0020*/ mov dsts=10"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x20
        assert parsed.opcode == "mov"
        assert "dsts=10" in parsed.operands

    def test_parse_instruction_with_no_regs(self, disasm):
        """Test parsing instruction with no register operands."""
        line = "  /*0030*/ nop"
        parsed = disasm.parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x30
        assert parsed.opcode == "nop"
        assert parsed.operands == []

    def test_parsed_to_instruction_stat_with_regs(self, disasm):
        """Test conversion to InstructionStat with register operands."""
        line = "  /*0010*/ add dsts=4 srcs=2,3"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert isinstance(inst, InstructionStat)
        assert inst.op == "add"
        assert inst.pc == 0x10
        assert inst.dsts == [4]
        assert inst.srcs == [2, 3]

    def test_parsed_to_instruction_stat_multiple_dsts(self, disasm):
        """Test conversion with multiple destination registers."""
        line = "  /*0040*/ send dsts=10,11 srcs=5"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.dsts == [10, 11]
        assert inst.srcs == [5]

    def test_parsed_to_instruction_stat_no_regs(self, disasm):
        """Test conversion with no register operands."""
        line = "  /*0030*/ nop"
        parsed = disasm.parse_instruction_line(line)
        inst = disasm.parsed_to_instruction_stat(parsed)

        assert inst.dsts == []
        assert inst.srcs == []

    def test_parse_all_functions_preserves_regs(self, disasm):
        """Test that register operands survive the full text round-trip."""
        output = """
.section .text._Z6kernelPf
  /*0000*/ mov dsts=4 srcs=2
  /*0010*/ add dsts=5 srcs=4,3
  /*0020*/ send dsts=10,11 srcs=5
  /*0030*/ nop
"""
        functions = disasm.parse_all_functions(output)

        assert len(functions) == 1
        assert functions[0].name == "_Z6kernelPf"
        assert len(functions[0].instructions) == 4

        # Check register operands survived the round-trip
        inst0 = functions[0].instructions[0]
        assert inst0.dsts == [4]
        assert inst0.srcs == [2]

        inst1 = functions[0].instructions[1]
        assert inst1.dsts == [5]
        assert inst1.srcs == [4, 3]

        inst2 = functions[0].instructions[2]
        assert inst2.dsts == [10, 11]
        assert inst2.srcs == [5]

        inst3 = functions[0].instructions[3]
        assert inst3.dsts == []
        assert inst3.srcs == []

    def test_assign_pcs_with_intel_regs(self, disasm):
        """Test that assign_pcs works with Intel GRF register numbers."""
        from leo.binary.dependency import build_assign_pcs

        output = """
.section .text._Z6kernelPf
  /*0000*/ mov dsts=4 srcs=2
  /*0010*/ add dsts=5 srcs=4,3
  /*0020*/ send srcs=5
"""
        functions = disasm.parse_all_functions(output)
        instructions = functions[0].instructions

        # Build assign_pcs
        build_assign_pcs(instructions)

        # Instruction at PC=0x10 reads reg 4, which was written at PC=0x00
        inst_add = instructions[1]
        assert 4 in inst_add.assign_pcs
        assert inst_add.assign_pcs[4] == [0x00]

        # Instruction at PC=0x20 reads reg 5, which was written at PC=0x10
        inst_send = instructions[2]
        assert 5 in inst_send.assign_pcs
        assert inst_send.assign_pcs[5] == [0x10]


# =============================================================================
# Cross-Vendor Compatibility Tests
# =============================================================================


class TestCrossVendorCompatibility:
    """Test that both disassemblers produce compatible output."""

    def test_instruction_stat_fields_same(self):
        """Both disassemblers produce InstructionStat with same fields."""
        nvidia = NVIDIADisassembler()
        amd = AMDDisassembler()

        # Parse sample instructions
        nvidia_line = "        /*0050*/                   FADD R2, R0, R1 ;"
        amd_line = "v_add_f32 v0, v1, v2 // 000000000100: 02000503"

        nvidia_parsed = nvidia.parse_instruction_line(nvidia_line)
        amd_parsed = amd.parse_instruction_line(amd_line)

        nvidia_inst = nvidia.parsed_to_instruction_stat(nvidia_parsed)
        amd_inst = amd.parsed_to_instruction_stat(amd_parsed)

        # Both should have same field types
        assert type(nvidia_inst.op) == type(amd_inst.op) == str
        assert type(nvidia_inst.pc) == type(amd_inst.pc) == int
        assert type(nvidia_inst.dsts) == type(amd_inst.dsts) == list
        assert type(nvidia_inst.srcs) == type(amd_inst.srcs) == list

    def test_parsed_function_structure_same(self):
        """Both disassemblers produce ParsedFunction with same structure."""
        nvidia = NVIDIADisassembler()
        amd = AMDDisassembler()

        nvidia_output = """
.section .text._Z4testv,"ax",@progbits
        /*0000*/                   NOP ;
        /*0010*/                   EXIT ;
.section .nv.info,
"""
        amd_output = """
0000000000001000 <test>:
       1000:	7e000280        	v_mov_b32_e32 v0, 0
       1004:	bf810000        	s_endpgm
"""

        nvidia_func = nvidia.parse_function(nvidia_output)
        amd_func = amd.parse_function(amd_output)

        # Both should have same structure
        assert isinstance(nvidia_func, ParsedFunction)
        assert isinstance(amd_func, ParsedFunction)
        assert isinstance(nvidia_func.name, str)
        assert isinstance(amd_func.name, str)
        assert isinstance(nvidia_func.instructions, list)
        assert isinstance(amd_func.instructions, list)
        assert isinstance(nvidia_func.labels, dict)
        assert isinstance(amd_func.labels, dict)


# =============================================================================
# Tool Availability Tests (Optional - Skip if tools not installed)
# =============================================================================


class TestToolAvailability:
    """Test actual tool availability (may be skipped)."""

    def test_llvm_objdump_available(self):
        """Check if llvm-objdump is available."""
        disasm = AMDDisassembler()

        if not disasm.check_available():
            pytest.skip("llvm-objdump not available in PATH or ROCm installation")

        assert disasm.check_available() is True
        version = disasm.get_version()
        assert version is not None
        assert "." in version  # Version should have dots

    def test_nvdisasm_available(self):
        """Check if nvdisasm is available."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvdisasm", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available = False

        if not available:
            pytest.skip("nvdisasm not available in PATH")

        disasm = NVIDIADisassembler()
        assert disasm.check_available() is True
        version = disasm.get_version()
        assert version is not None
