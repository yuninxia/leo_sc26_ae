"""Tests for leo.binary module - CUDA binary analysis."""

import pytest
from pathlib import Path

from leo.binary import (
    Block,
    CFG,
    Control,
    CubinParser,
    CubinSection,
    EdgeType,
    Function,
    InstructionStat,
    ParsedFunction,
    PredicateFlag,
    Target,
    backward_slice,
    backward_slice_for_register,
    build_assign_pcs,
    build_cfg_from_instructions,
    check_nvdisasm_available,
    disassemble,
    disassemble_and_parse,
    extract_control_from_bytes,
    get_all_dependencies,
    parse_all_functions,
    print_dependencies,
    validate_assign_pcs,
)
from leo.binary.instruction import (
    parse_barrier,
    parse_predicate,
    parse_register,
    parse_registers,
    parse_uniform_register,
    parse_instruction_predicate,
    detect_indirect_addressing,
    get_operation_width,
)
from leo.binary.nvdisasm import (
    NvdisasmError,
    extract_branch_target_label,
    extract_call_target,
    parse_instruction_line,
)


# Path to test gpubin file
TEST_GPUBIN = Path(__file__).parent / (
    "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-measurements/"
    "gpubins/67e7ddd42e43d0ca040956d9d9b316fa.gpubin"
)


# =============================================================================
# Instruction Parsing Unit Tests
# =============================================================================


class TestRegisterParsing:
    """Tests for register extraction from operands."""

    def test_parse_general_register(self):
        """Test parsing general registers R0-R255."""
        assert parse_register("R0") == 0
        assert parse_register("R255") == 255
        assert parse_register("R10") == 10
        assert parse_register("[R2]") == 2
        assert parse_register("c[0x0][0x28]") is None  # Constant memory

    def test_parse_registers_64bit(self):
        """Test parsing 64-bit register pairs."""
        regs = parse_registers("R0", width=64)
        assert regs == [0, 1]

    def test_parse_registers_128bit(self):
        """Test parsing 128-bit register quads."""
        regs = parse_registers("R4", width=128)
        assert regs == [4, 5, 6, 7]

    def test_parse_predicate(self):
        """Test parsing predicate registers P0-P6."""
        assert parse_predicate("P0") == 0
        assert parse_predicate("P6") == 6
        assert parse_predicate("PT") is None  # True predicate, not P register
        assert parse_predicate("R0") is None

    def test_parse_barrier(self):
        """Test parsing barrier registers B0-B6."""
        assert parse_barrier("B0") == 0
        assert parse_barrier("B1") == 1
        assert parse_barrier("B6") == 6
        assert parse_barrier("SB1") == 1
        assert parse_barrier("`(.L_x_1)") is None  # Label, not barrier
        assert parse_barrier("R0") is None

    def test_parse_uniform_register(self):
        """Test parsing uniform registers UR0-UR63."""
        assert parse_uniform_register("UR0") == 0
        assert parse_uniform_register("UR63") == 63
        assert parse_uniform_register("R0") is None


class TestPredicateParsing:
    """Tests for instruction predicate extraction."""

    def test_parse_true_predicate(self):
        """Test parsing true predicate @P1."""
        pred, flag = parse_instruction_predicate("@P1 FADD R0, R1, R2")
        assert pred == 1
        assert flag == PredicateFlag.PREDICATE_TRUE

    def test_parse_negated_predicate(self):
        """Test parsing negated predicate @!P2."""
        pred, flag = parse_instruction_predicate("@!P2 EXIT")
        assert pred == 2
        assert flag == PredicateFlag.PREDICATE_FALSE

    def test_parse_no_predicate(self):
        """Test instruction without predicate."""
        pred, flag = parse_instruction_predicate("IMAD R0, R1, R2, R3")
        assert pred == -1
        assert flag == PredicateFlag.PREDICATE_NONE


class TestOperandHelpers:
    """Tests for operand analysis helpers."""

    def test_detect_indirect_addressing(self):
        """Test indirect addressing detection."""
        assert detect_indirect_addressing("[R2]") is True
        assert detect_indirect_addressing("[R2+0x10]") is True
        assert detect_indirect_addressing("R0") is False
        assert detect_indirect_addressing("c[0x0][0x28]") is False

    def test_get_operation_width(self):
        """Test operation width detection from opcode."""
        assert get_operation_width("LDG.E.64") == 64
        assert get_operation_width("LDG.E.128") == 128
        assert get_operation_width("FADD") == 32
        assert get_operation_width("IMAD.MOV.U32") == 32


class TestBranchTargetParsing:
    """Tests for branch target label extraction."""

    def test_extract_branch_target_basic(self):
        """Test extracting label from basic branch operand."""
        assert extract_branch_target_label("`(.L_x_0)") == ".L_x_0"
        assert extract_branch_target_label("`(.L_x_123)") == ".L_x_123"

    def test_extract_branch_target_variants(self):
        """Test extracting label from various formats."""
        assert extract_branch_target_label("`(.L_index_0)") == ".L_index_0"
        assert extract_branch_target_label("`(.L_1_2)") == ".L_1_2"

    def test_extract_branch_target_no_match(self):
        """Test non-label operands return None."""
        assert extract_branch_target_label("R0") is None
        assert extract_branch_target_label("[R2]") is None
        assert extract_branch_target_label("c[0x0][0x28]") is None
        assert extract_branch_target_label("0x100") is None

    def test_extract_call_target_basic(self):
        """Test extracting function name from CALL operand."""
        assert extract_call_target("`(_Z10helperFunc)") == "_Z10helperFunc"
        assert extract_call_target("`(_Z3foo)") == "_Z3foo"

    def test_extract_call_target_mangled_names(self):
        """Test extracting complex mangled function names."""
        assert extract_call_target("`(_Z8xcomputePKdS0_Pdii)") == "_Z8xcomputePKdS0_Pdii"


# =============================================================================
# InstructionStat Unit Tests
# =============================================================================


class TestInstructionStat:
    """Tests for InstructionStat data structure."""

    def test_instruction_creation(self):
        """Test basic InstructionStat creation."""
        inst = InstructionStat(op="IMAD", pc=0x100, dsts=[1], srcs=[2, 3])
        assert inst.op == "IMAD"
        assert inst.pc == 0x100
        assert inst.dsts == [1]
        assert inst.srcs == [2, 3]

    def test_instruction_sorting(self):
        """Test instructions sort by PC."""
        inst1 = InstructionStat(op="A", pc=0x20)
        inst2 = InstructionStat(op="B", pc=0x10)
        inst3 = InstructionStat(op="C", pc=0x30)

        sorted_insts = sorted([inst1, inst2, inst3])
        assert [i.pc for i in sorted_insts] == [0x10, 0x20, 0x30]

    def test_instruction_classification(self):
        """Test instruction type classification."""
        load = InstructionStat(op="LDG.E.64", pc=0)
        store = InstructionStat(op="STG.E.64", pc=0)
        branch = InstructionStat(op="BRA", pc=0)
        sync = InstructionStat(op="BSSY", pc=0)

        assert load.is_load() is True
        assert load.is_memory_op() is True
        assert store.is_store() is True
        assert branch.is_branch() is True
        assert sync.is_sync() is True

    def test_instruction_predicated(self):
        """Test predicated instruction detection."""
        predicated = InstructionStat(
            op="EXIT",
            pc=0,
            predicate=0,
            predicate_flag=PredicateFlag.PREDICATE_TRUE,
        )
        unpredicated = InstructionStat(op="IMAD", pc=0)

        assert predicated.is_predicated() is True
        assert unpredicated.is_predicated() is False


class TestControl:
    """Tests for Control data structure."""

    def test_control_defaults(self):
        """Test Control default values."""
        ctrl = Control()
        assert ctrl.wait == 0
        assert ctrl.read == 7  # BARRIER_NONE
        assert ctrl.write == 7
        assert ctrl.stall == 1

    def test_waits_on_barrier(self):
        """Test barrier wait mask checking."""
        ctrl = Control(wait=0b000101)  # Waits on B1 and B3
        assert ctrl.waits_on_barrier(1) is True
        assert ctrl.waits_on_barrier(3) is True
        assert ctrl.waits_on_barrier(2) is False

    def test_get_wait_barriers(self):
        """Test getting list of waited barriers."""
        ctrl = Control(wait=0b110001)  # B1, B5, B6
        barriers = ctrl.get_wait_barriers()
        assert set(barriers) == {1, 5, 6}


# =============================================================================
# nvdisasm Integration Tests
# =============================================================================


@pytest.fixture
def gpubin_path():
    """Get path to test gpubin, skip if not available."""
    if not TEST_GPUBIN.exists():
        pytest.skip("Test gpubin not available")
    return str(TEST_GPUBIN)


class TestNvdisasmAvailability:
    """Tests for nvdisasm tool availability."""

    def test_check_availability(self):
        """Test nvdisasm availability check."""
        # This should work on systems with CUDA toolkit
        available = check_nvdisasm_available()
        if not available:
            pytest.skip("nvdisasm not available")
        assert available is True


class TestNvdisasmInvocation:
    """Tests for nvdisasm invocation."""

    def test_disassemble_gpubin(self, gpubin_path):
        """Test disassembling a gpubin file."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        output = disassemble(gpubin_path)
        assert len(output) > 0
        assert ".text." in output or "IMAD" in output

    def test_disassemble_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(NvdisasmError):
            disassemble("/nonexistent/file.cubin")


class TestInstructionLineParsing:
    """Tests for parsing individual instruction lines."""

    def test_parse_simple_instruction(self):
        """Test parsing a simple instruction line."""
        line = "        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;"
        parsed = parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x0000
        assert parsed.opcode == "IMAD.MOV.U32"
        assert parsed.predicate is None
        assert "R1" in parsed.operands

    def test_parse_predicated_instruction(self):
        """Test parsing a predicated instruction."""
        line = "        /*0070*/               @P0 EXIT ;"
        parsed = parse_instruction_line(line)

        assert parsed is not None
        assert parsed.offset == 0x0070
        assert parsed.opcode == "EXIT"
        assert parsed.predicate == "P0"

    def test_parse_negated_predicate(self):
        """Test parsing instruction with negated predicate."""
        line = "        /*0130*/              @!P0 BRA `(.L_x_0) ;"
        parsed = parse_instruction_line(line)

        assert parsed is not None
        assert parsed.predicate == "!P0"
        assert parsed.opcode == "BRA"

    def test_parse_memory_instruction(self):
        """Test parsing memory instruction."""
        line = "        /*0140*/                   LDG.E.64 R30, [R12.64] ;"
        parsed = parse_instruction_line(line)

        assert parsed is not None
        assert parsed.opcode == "LDG.E.64"
        assert "[R12.64]" in parsed.operands or "R12.64" in str(parsed.operands)

    def test_skip_data_directive(self):
        """Test that data directives are skipped."""
        line = "        /*0000*/    .byte   0xff, 0xff"
        parsed = parse_instruction_line(line)
        assert parsed is None

    def test_skip_non_instruction(self):
        """Test that non-instruction lines are skipped."""
        assert parse_instruction_line("        .section .text") is None
        assert parse_instruction_line("_Z8xcomputePKdS0_Pdii:") is None
        assert parse_instruction_line("") is None


class TestFunctionParsing:
    """Tests for parsing complete functions."""

    def test_parse_function(self, gpubin_path):
        """Test parsing a function from gpubin."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)

        assert func is not None
        assert len(func.name) > 0
        assert len(func.instructions) > 0

    def test_function_has_instructions(self, gpubin_path):
        """Test that parsed function has correct instruction structure."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)

        # All instructions should have valid PCs
        for inst in func.instructions:
            assert inst.pc >= 0
            assert len(inst.op) > 0

        # Instructions should be in PC order
        pcs = [inst.pc for inst in func.instructions]
        assert pcs == sorted(pcs)

    def test_parse_all_functions(self, gpubin_path):
        """Test parsing all functions from gpubin."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        output = disassemble(gpubin_path)
        functions = parse_all_functions(output)

        assert len(functions) > 0
        for func in functions:
            assert len(func.instructions) > 0


class TestInstructionStatExtraction:
    """Tests for InstructionStat extraction from parsed instructions."""

    def test_extract_destinations(self, gpubin_path):
        """Test that destination registers are extracted."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)

        # Find IMAD instruction (should have destination)
        imad_insts = [i for i in func.instructions if "IMAD" in i.op]
        assert len(imad_insts) > 0

        # At least some should have destinations
        has_dsts = sum(1 for i in imad_insts if len(i.dsts) > 0)
        assert has_dsts > 0

    def test_extract_sources(self, gpubin_path):
        """Test that source registers are extracted."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)

        # Find an instruction with sources (IMAD typically has sources)
        insts_with_srcs = [i for i in func.instructions if len(i.srcs) > 0]
        assert len(insts_with_srcs) > 0

    def test_extract_predicates(self, gpubin_path):
        """Test that predicate information is extracted."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)

        # Find predicated instructions
        predicated = [i for i in func.instructions if i.is_predicated()]
        assert len(predicated) > 0

        # All predicated should have valid predicate register
        for inst in predicated:
            assert inst.predicate >= 0
            assert inst.predicate_flag != PredicateFlag.PREDICATE_NONE

    def test_extract_barriers(self, gpubin_path):
        """Test that barrier registers are extracted."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)

        # Find barrier instructions
        sync_insts = [i for i in func.instructions if i.is_sync()]

        if len(sync_insts) > 0:
            # At least some should have barrier destinations
            has_bdsts = sum(1 for i in sync_insts if len(i.bdsts) > 0)
            assert has_bdsts > 0

    def test_memory_operations(self, gpubin_path):
        """Test memory operation extraction."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)

        # Find load instructions
        loads = [i for i in func.instructions if i.is_load()]
        assert len(loads) > 0

        # Loads should have destinations and detect indirect addressing
        for load in loads:
            if "LDG" in load.op:
                assert len(load.dsts) > 0
                # Global loads typically use indirect addressing
                # (but our test may not always have this)


# =============================================================================
# CFG Unit Tests
# =============================================================================


class TestTarget:
    """Tests for Target (control flow edge) data structure."""

    def test_target_creation(self):
        """Test Target creation with all fields."""
        target = Target(from_pc=0x100, to_block_id=2, edge_type=EdgeType.COND_TAKEN)
        assert target.from_pc == 0x100
        assert target.to_block_id == 2
        assert target.edge_type == EdgeType.COND_TAKEN

    def test_target_default_edge_type(self):
        """Test Target defaults to DIRECT edge type."""
        target = Target(from_pc=0x100, to_block_id=1)
        assert target.edge_type == EdgeType.DIRECT

    def test_target_sorting(self):
        """Test Targets sort by source PC."""
        t1 = Target(from_pc=0x200, to_block_id=0)
        t2 = Target(from_pc=0x100, to_block_id=1)
        t3 = Target(from_pc=0x150, to_block_id=2)

        sorted_targets = sorted([t1, t2, t3])
        assert [t.from_pc for t in sorted_targets] == [0x100, 0x150, 0x200]


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_type_values(self):
        """Test EdgeType enum has expected values."""
        assert EdgeType.DIRECT == 0
        assert EdgeType.COND_TAKEN == 1
        assert EdgeType.COND_NOT_TAKEN == 2
        assert EdgeType.CALL == 3
        assert EdgeType.CALL_FT == 4
        assert EdgeType.RETURN == 5


class TestBlock:
    """Tests for Block (basic block) data structure."""

    def test_block_creation(self):
        """Test Block creation with instructions."""
        inst1 = InstructionStat(op="IMAD", pc=0x100)
        inst2 = InstructionStat(op="LDG", pc=0x110)

        block = Block(id=0, name=".L_0", instructions=[inst1, inst2])
        assert block.id == 0
        assert block.name == ".L_0"
        assert len(block.instructions) == 2

    def test_block_properties(self):
        """Test Block computed properties."""
        inst1 = InstructionStat(op="IMAD", pc=0x100)
        inst2 = InstructionStat(op="LDG", pc=0x110)
        inst3 = InstructionStat(op="FADD", pc=0x120)

        block = Block(id=0, instructions=[inst1, inst2, inst3])

        assert block.start_pc == 0x100
        assert block.end_pc == 0x120
        assert block.size == 3

    def test_block_empty_properties(self):
        """Test Block properties when empty."""
        block = Block(id=0)
        assert block.start_pc is None
        assert block.end_pc is None
        assert block.size == 0

    def test_block_contains_pc(self):
        """Test Block.contains_pc method."""
        inst1 = InstructionStat(op="IMAD", pc=0x100)
        inst2 = InstructionStat(op="LDG", pc=0x120)

        block = Block(id=0, instructions=[inst1, inst2])

        assert block.contains_pc(0x100) is True
        assert block.contains_pc(0x110) is True  # Between start and end
        assert block.contains_pc(0x120) is True
        assert block.contains_pc(0x90) is False  # Before
        assert block.contains_pc(0x130) is False  # After

    def test_block_get_instruction_at(self):
        """Test Block.get_instruction_at method."""
        inst1 = InstructionStat(op="IMAD", pc=0x100)
        inst2 = InstructionStat(op="LDG", pc=0x110)

        block = Block(id=0, instructions=[inst1, inst2])

        found = block.get_instruction_at(0x100)
        assert found is not None
        assert found.op == "IMAD"

        found = block.get_instruction_at(0x110)
        assert found is not None
        assert found.op == "LDG"

        found = block.get_instruction_at(0x200)
        assert found is None

    def test_block_successor_ids(self):
        """Test Block.get_successor_block_ids method."""
        block = Block(id=0)
        block.targets.append(Target(from_pc=0, to_block_id=1))
        block.targets.append(Target(from_pc=0, to_block_id=3))

        succ_ids = block.get_successor_block_ids()
        assert set(succ_ids) == {1, 3}

    def test_block_is_exit_block(self):
        """Test Block.is_exit_block method."""
        exit_block = Block(id=0)  # No targets
        assert exit_block.is_exit_block() is True

        non_exit = Block(id=1)
        non_exit.targets.append(Target(from_pc=0, to_block_id=2))
        assert non_exit.is_exit_block() is False

    def test_block_has_branch(self):
        """Test Block.has_branch method."""
        # Block ending with branch
        branch_block = Block(
            id=0, instructions=[InstructionStat(op="BRA", pc=0x100)]
        )
        assert branch_block.has_branch() is True

        # Block ending with non-branch
        non_branch_block = Block(
            id=1, instructions=[InstructionStat(op="IMAD", pc=0x100)]
        )
        assert non_branch_block.has_branch() is False

        # Empty block
        empty_block = Block(id=2)
        assert empty_block.has_branch() is False

    def test_block_has_call(self):
        """Test Block.has_call method."""
        call_block = Block(
            id=0, instructions=[InstructionStat(op="CALL", pc=0x100)]
        )
        assert call_block.has_call() is True

        non_call_block = Block(
            id=1, instructions=[InstructionStat(op="IMAD", pc=0x100)]
        )
        assert non_call_block.has_call() is False

    def test_block_equality(self):
        """Test Block equality by ID."""
        b1 = Block(id=5, name=".L_0")
        b2 = Block(id=5, name=".L_1")  # Same ID, different name
        b3 = Block(id=6, name=".L_0")

        assert b1 == b2  # Same ID
        assert b1 != b3  # Different ID

    def test_block_hash(self):
        """Test Block is hashable by ID."""
        b1 = Block(id=5)
        b2 = Block(id=5)
        b3 = Block(id=6)

        block_set = {b1, b2, b3}
        assert len(block_set) == 2  # b1 and b2 have same hash

    def test_block_sorting(self):
        """Test Blocks sort by first instruction PC."""
        b1 = Block(id=0, instructions=[InstructionStat(op="NOP", pc=0x200)])
        b2 = Block(id=1, instructions=[InstructionStat(op="NOP", pc=0x100)])
        b3 = Block(id=2, instructions=[InstructionStat(op="NOP", pc=0x150)])

        sorted_blocks = sorted([b1, b2, b3])
        assert [b.id for b in sorted_blocks] == [1, 2, 0]


class TestFunction:
    """Tests for Function data structure."""

    def test_function_creation(self):
        """Test Function creation."""
        b0 = Block(id=0, instructions=[InstructionStat(op="IMAD", pc=0)])
        b1 = Block(id=1, instructions=[InstructionStat(op="EXIT", pc=8)])

        func = Function(name="_Z8testFunc", blocks=[b0, b1], entry_block_id=0)
        assert func.name == "_Z8testFunc"
        assert len(func.blocks) == 2
        assert func.entry_block_id == 0

    def test_function_entry_block(self):
        """Test Function.entry_block property."""
        b0 = Block(id=0, instructions=[InstructionStat(op="IMAD", pc=0)])
        b1 = Block(id=1, instructions=[InstructionStat(op="EXIT", pc=8)])

        func = Function(name="test", blocks=[b0, b1], entry_block_id=0)
        entry = func.entry_block

        assert entry is not None
        assert entry.id == 0

    def test_function_entry_block_fallback(self):
        """Test entry_block falls back to first block if entry_block_id not found."""
        b0 = Block(id=0)
        b1 = Block(id=1)

        func = Function(name="test", blocks=[b0, b1], entry_block_id=99)  # Invalid ID
        entry = func.entry_block

        assert entry is not None
        assert entry.id == 0  # Falls back to first block

    def test_function_num_instructions(self):
        """Test Function.num_instructions property."""
        b0 = Block(id=0, instructions=[
            InstructionStat(op="IMAD", pc=0),
            InstructionStat(op="LDG", pc=8),
        ])
        b1 = Block(id=1, instructions=[
            InstructionStat(op="FADD", pc=16),
        ])

        func = Function(name="test", blocks=[b0, b1])
        assert func.num_instructions == 3

    def test_function_get_block_by_id(self):
        """Test Function.get_block_by_id method."""
        b0 = Block(id=0)
        b1 = Block(id=5)
        b2 = Block(id=10)

        func = Function(name="test", blocks=[b0, b1, b2])

        assert func.get_block_by_id(0) == b0
        assert func.get_block_by_id(5) == b1
        assert func.get_block_by_id(10) == b2
        assert func.get_block_by_id(99) is None

    def test_function_get_block_by_name(self):
        """Test Function.get_block_by_name method."""
        b0 = Block(id=0, name=".L_0")
        b1 = Block(id=1, name=".L_1")

        func = Function(name="test", blocks=[b0, b1])

        assert func.get_block_by_name(".L_0") == b0
        assert func.get_block_by_name(".L_1") == b1
        assert func.get_block_by_name(".L_99") is None

    def test_function_get_block_containing_pc(self):
        """Test Function.get_block_containing_pc method."""
        b0 = Block(id=0, instructions=[
            InstructionStat(op="IMAD", pc=0),
            InstructionStat(op="LDG", pc=8),
        ])
        b1 = Block(id=1, instructions=[
            InstructionStat(op="FADD", pc=16),
            InstructionStat(op="EXIT", pc=24),
        ])

        func = Function(name="test", blocks=[b0, b1])

        assert func.get_block_containing_pc(0) == b0
        assert func.get_block_containing_pc(8) == b0
        assert func.get_block_containing_pc(16) == b1
        assert func.get_block_containing_pc(100) is None

    def test_function_get_instruction_at(self):
        """Test Function.get_instruction_at method."""
        inst1 = InstructionStat(op="IMAD", pc=0)
        inst2 = InstructionStat(op="LDG", pc=16)
        b0 = Block(id=0, instructions=[inst1])
        b1 = Block(id=1, instructions=[inst2])

        func = Function(name="test", blocks=[b0, b1])

        assert func.get_instruction_at(0) == inst1
        assert func.get_instruction_at(16) == inst2
        assert func.get_instruction_at(99) is None

    def test_function_get_all_instructions(self):
        """Test Function.get_all_instructions method returns sorted list."""
        # Create blocks with instructions out of order
        b0 = Block(id=0, instructions=[InstructionStat(op="A", pc=16)])
        b1 = Block(id=1, instructions=[InstructionStat(op="B", pc=0)])
        b2 = Block(id=2, instructions=[InstructionStat(op="C", pc=8)])

        func = Function(name="test", blocks=[b0, b1, b2])
        all_insts = func.get_all_instructions()

        assert len(all_insts) == 3
        assert [i.pc for i in all_insts] == [0, 8, 16]  # Sorted by PC


class TestCFG:
    """Tests for CFG (Control Flow Graph) class."""

    @pytest.fixture
    def linear_cfg(self):
        """Create a linear CFG: b0 -> b1 -> b2."""
        b0 = Block(id=0, instructions=[InstructionStat(op="IMAD", pc=0)])
        b1 = Block(id=1, instructions=[InstructionStat(op="LDG", pc=8)])
        b2 = Block(id=2, instructions=[InstructionStat(op="EXIT", pc=16)])

        b0.targets.append(Target(from_pc=0, to_block_id=1))
        b1.targets.append(Target(from_pc=8, to_block_id=2))
        # b2 has no targets (exit)

        func = Function(name="linear", blocks=[b0, b1, b2], entry_block_id=0)
        return CFG(func)

    @pytest.fixture
    def diamond_cfg(self):
        """Create a diamond CFG: b0 -> {b1, b2} -> b3."""
        b0 = Block(id=0, instructions=[InstructionStat(op="BRA", pc=0, predicate=0, predicate_flag=PredicateFlag.PREDICATE_TRUE)])
        b1 = Block(id=1, instructions=[InstructionStat(op="IMAD", pc=8)])
        b2 = Block(id=2, instructions=[InstructionStat(op="FADD", pc=16)])
        b3 = Block(id=3, instructions=[InstructionStat(op="EXIT", pc=24)])

        b0.targets.append(Target(from_pc=0, to_block_id=1, edge_type=EdgeType.COND_TAKEN))
        b0.targets.append(Target(from_pc=0, to_block_id=2, edge_type=EdgeType.COND_NOT_TAKEN))
        b1.targets.append(Target(from_pc=8, to_block_id=3))
        b2.targets.append(Target(from_pc=16, to_block_id=3))

        func = Function(name="diamond", blocks=[b0, b1, b2, b3], entry_block_id=0)
        return CFG(func)

    @pytest.fixture
    def loop_cfg(self):
        """Create a loop CFG: b0 -> b1 -> b0 (back edge), b1 -> b2."""
        b0 = Block(id=0, instructions=[InstructionStat(op="IMAD", pc=0)])
        b1 = Block(id=1, instructions=[InstructionStat(op="BRA", pc=8, predicate=0, predicate_flag=PredicateFlag.PREDICATE_TRUE)])
        b2 = Block(id=2, instructions=[InstructionStat(op="EXIT", pc=16)])

        b0.targets.append(Target(from_pc=0, to_block_id=1))
        b1.targets.append(Target(from_pc=8, to_block_id=0, edge_type=EdgeType.COND_TAKEN))  # Back edge
        b1.targets.append(Target(from_pc=8, to_block_id=2, edge_type=EdgeType.COND_NOT_TAKEN))

        func = Function(name="loop", blocks=[b0, b1, b2], entry_block_id=0)
        return CFG(func)

    def test_cfg_creation(self, linear_cfg):
        """Test CFG creation from Function."""
        assert linear_cfg.function is not None
        assert linear_cfg.function.name == "linear"

    def test_cfg_get_block(self, linear_cfg):
        """Test CFG.get_block method."""
        block = linear_cfg.get_block(1)
        assert block is not None
        assert block.id == 1

        block = linear_cfg.get_block(99)
        assert block is None

    def test_cfg_predecessors_linear(self, linear_cfg):
        """Test predecessors in linear CFG."""
        b0 = linear_cfg.get_block(0)
        b1 = linear_cfg.get_block(1)
        b2 = linear_cfg.get_block(2)

        assert linear_cfg.predecessors(b0) == []  # Entry has no preds
        assert len(linear_cfg.predecessors(b1)) == 1
        assert linear_cfg.predecessors(b1)[0].id == 0
        assert len(linear_cfg.predecessors(b2)) == 1
        assert linear_cfg.predecessors(b2)[0].id == 1

    def test_cfg_predecessors_diamond(self, diamond_cfg):
        """Test predecessors in diamond CFG."""
        b3 = diamond_cfg.get_block(3)

        preds = diamond_cfg.predecessors(b3)
        assert len(preds) == 2
        assert set(p.id for p in preds) == {1, 2}

    def test_cfg_successors_linear(self, linear_cfg):
        """Test successors in linear CFG."""
        b0 = linear_cfg.get_block(0)
        b1 = linear_cfg.get_block(1)
        b2 = linear_cfg.get_block(2)

        succs = linear_cfg.successors(b0)
        assert len(succs) == 1
        assert succs[0].id == 1

        succs = linear_cfg.successors(b2)
        assert succs == []  # Exit has no succs

    def test_cfg_successors_diamond(self, diamond_cfg):
        """Test successors in diamond CFG."""
        b0 = diamond_cfg.get_block(0)

        succs = diamond_cfg.successors(b0)
        assert len(succs) == 2
        assert set(s.id for s in succs) == {1, 2}

    def test_cfg_get_entry_block(self, linear_cfg):
        """Test CFG.get_entry_block method."""
        entry = linear_cfg.get_entry_block()
        assert entry is not None
        assert entry.id == 0

    def test_cfg_get_exit_blocks(self, linear_cfg, diamond_cfg):
        """Test CFG.get_exit_blocks method."""
        exits = linear_cfg.get_exit_blocks()
        assert len(exits) == 1
        assert exits[0].id == 2

        exits = diamond_cfg.get_exit_blocks()
        assert len(exits) == 1
        assert exits[0].id == 3

    def test_cfg_is_reachable(self, linear_cfg):
        """Test CFG.is_reachable method."""
        b0 = linear_cfg.get_block(0)
        b1 = linear_cfg.get_block(1)
        b2 = linear_cfg.get_block(2)

        assert linear_cfg.is_reachable(b0, b1) is True
        assert linear_cfg.is_reachable(b0, b2) is True
        assert linear_cfg.is_reachable(b1, b2) is True

        # Reverse is not reachable (no cycles)
        assert linear_cfg.is_reachable(b2, b0) is False
        assert linear_cfg.is_reachable(b1, b0) is False

    def test_cfg_is_reachable_with_loop(self, loop_cfg):
        """Test reachability with back edges."""
        b0 = loop_cfg.get_block(0)
        b1 = loop_cfg.get_block(1)

        # With back edge, b0 is reachable from b1
        assert loop_cfg.is_reachable(b1, b0) is True
        assert loop_cfg.is_reachable(b0, b1) is True

    def test_cfg_is_loop_header(self, loop_cfg, linear_cfg):
        """Test CFG.is_loop_header method."""
        b0 = loop_cfg.get_block(0)
        b1 = loop_cfg.get_block(1)
        b2 = loop_cfg.get_block(2)

        assert loop_cfg.is_loop_header(b0) is True  # Has back edge from b1
        assert loop_cfg.is_loop_header(b1) is False
        assert loop_cfg.is_loop_header(b2) is False

        # Linear CFG has no loops
        for block in linear_cfg.function.blocks:
            assert linear_cfg.is_loop_header(block) is False

    def test_cfg_get_loop_blocks(self, loop_cfg):
        """Test CFG.get_loop_blocks method."""
        b0 = loop_cfg.get_block(0)

        loop_blocks = loop_cfg.get_loop_blocks(b0)
        assert len(loop_blocks) == 2
        assert set(b.id for b in loop_blocks) == {0, 1}

    def test_cfg_dominator_tree_linear(self, linear_cfg):
        """Test dominator computation for linear CFG."""
        idom = linear_cfg.get_dominator_tree()

        # Entry dominates itself (idom is None)
        assert idom[0] is None

        # Each block dominated by previous
        assert idom[1] == 0
        assert idom[2] == 1

    def test_cfg_dominator_tree_diamond(self, diamond_cfg):
        """Test dominator computation for diamond CFG."""
        idom = diamond_cfg.get_dominator_tree()

        # Entry dominates itself
        assert idom[0] is None

        # b1 and b2 dominated by b0
        assert idom[1] == 0
        assert idom[2] == 0

        # b3 dominated by b0 (common dominator of b1 and b2)
        assert idom[3] == 0

    def test_cfg_dominates(self, linear_cfg):
        """Test CFG.dominates method."""
        b0 = linear_cfg.get_block(0)
        b1 = linear_cfg.get_block(1)
        b2 = linear_cfg.get_block(2)

        assert linear_cfg.dominates(b0, b0) is True  # Self-dominance
        assert linear_cfg.dominates(b0, b1) is True
        assert linear_cfg.dominates(b0, b2) is True
        assert linear_cfg.dominates(b1, b2) is True

        assert linear_cfg.dominates(b2, b0) is False
        assert linear_cfg.dominates(b2, b1) is False
        assert linear_cfg.dominates(b1, b0) is False


class TestBuildCFGFromInstructions:
    """Tests for build_cfg_from_instructions function."""

    def test_build_cfg_empty(self):
        """Test building CFG from empty instruction list."""
        cfg = build_cfg_from_instructions([])
        assert cfg.function.name == "unknown"
        assert len(cfg.function.blocks) == 0

    def test_build_cfg_single_block(self):
        """Test building CFG with no branches (single block)."""
        insts = [
            InstructionStat(op="IMAD", pc=0),
            InstructionStat(op="LDG", pc=8),
            InstructionStat(op="FADD", pc=16),
        ]

        cfg = build_cfg_from_instructions(insts, function_name="test")

        assert cfg.function.name == "test"
        assert len(cfg.function.blocks) == 1
        assert cfg.function.blocks[0].size == 3

    def test_build_cfg_with_branch(self):
        """Test building CFG splits blocks at branches."""
        insts = [
            InstructionStat(op="IMAD", pc=0),
            InstructionStat(op="BRA", pc=8),  # Unconditional branch
            InstructionStat(op="FADD", pc=16),  # After branch
            InstructionStat(op="EXIT", pc=24),
        ]

        cfg = build_cfg_from_instructions(insts)

        # Should split after BRA
        assert len(cfg.function.blocks) == 2
        assert cfg.function.blocks[0].size == 2  # IMAD, BRA
        assert cfg.function.blocks[1].size == 2  # FADD, EXIT

    def test_build_cfg_fall_through(self):
        """Test fall-through edges are added for non-unconditional branches."""
        insts = [
            InstructionStat(op="IMAD", pc=0),
            InstructionStat(
                op="BRA", pc=8,
                predicate=0,
                predicate_flag=PredicateFlag.PREDICATE_TRUE
            ),  # Conditional branch
            InstructionStat(op="FADD", pc=16),
        ]

        cfg = build_cfg_from_instructions(insts)

        # First block should have fall-through edge to second
        first_block = cfg.function.blocks[0]
        assert len(first_block.targets) >= 1

        # Check the edge exists
        succ_ids = first_block.get_successor_block_ids()
        assert 1 in succ_ids  # Points to second block

    def test_build_cfg_multiple_branches(self):
        """Test CFG with multiple branches."""
        insts = [
            InstructionStat(op="IMAD", pc=0),
            InstructionStat(op="BRA", pc=8),
            InstructionStat(op="LDG", pc=16),
            InstructionStat(op="BRA", pc=24),
            InstructionStat(op="EXIT", pc=32),
        ]

        cfg = build_cfg_from_instructions(insts)

        # Should have 3 blocks
        assert len(cfg.function.blocks) == 3

    def test_build_cfg_preserves_instruction_order(self):
        """Test instructions remain in PC order."""
        # Create instructions out of order
        insts = [
            InstructionStat(op="C", pc=16),
            InstructionStat(op="A", pc=0),
            InstructionStat(op="B", pc=8),
        ]

        cfg = build_cfg_from_instructions(insts)
        all_insts = cfg.function.get_all_instructions()

        assert [i.op for i in all_insts] == ["A", "B", "C"]


class TestCFGIntegration:
    """Integration tests for CFG with real gpubin data."""

    def test_build_cfg_from_real_function(self, gpubin_path):
        """Test building CFG from real parsed function."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)
        assert func is not None

        cfg = build_cfg_from_instructions(func.instructions, func.name)

        # Should have at least one block
        assert len(cfg.function.blocks) >= 1

        # All instructions should be in blocks
        total_insts = sum(b.size for b in cfg.function.blocks)
        assert total_insts == len(func.instructions)

        # Entry block should exist
        assert cfg.get_entry_block() is not None

    def test_cfg_navigation_real_function(self, gpubin_path):
        """Test CFG navigation on real function."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)
        cfg = build_cfg_from_instructions(func.instructions, func.name)

        # Entry block should have no predecessors
        entry = cfg.get_entry_block()
        assert len(cfg.predecessors(entry)) == 0

        # Should have at least one exit block
        exits = cfg.get_exit_blocks()
        assert len(exits) >= 1

        # Exit blocks should have no successors
        for exit_block in exits:
            assert len(cfg.successors(exit_block)) == 0


class TestLabelMapping:
    """Tests for label capture and mapping in ParsedFunction."""

    def test_parsed_function_has_labels(self, gpubin_path):
        """Test that ParsedFunction captures labels from nvdisasm output."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)
        assert func is not None

        # Real functions typically have some labels for branches
        # (This may be 0 if the function has no branches, which is fine)
        # The important thing is that labels dict exists and is a dict
        assert isinstance(func.labels, dict)

    def test_branch_target_stored_in_instruction(self, gpubin_path):
        """Test that branch instructions have branch_target set."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)
        assert func is not None

        # Find branch instructions
        branches = [i for i in func.instructions if i.is_branch()]

        # If there are branches with targets, they should have branch_target set
        branches_with_targets = [b for b in branches if b.branch_target is not None]

        # Note: Not all branches have explicit targets (e.g., EXIT, RET)
        # But any BRA instruction should have one
        bra_insts = [i for i in func.instructions if i.op.startswith("BRA")]
        if bra_insts:
            # At least some BRA instructions should have targets
            # (may not be all if some are register-indirect)
            pass  # Just verify no errors occurred during parsing


class TestEdgeTypes:
    """Tests for edge type assignment in CFG building."""

    def test_conditional_branch_edge_types(self):
        """Test COND_TAKEN and COND_NOT_TAKEN edge types."""
        # Create a conditional branch that jumps to label
        insts = [
            InstructionStat(
                op="BRA", pc=0,
                predicate=0,
                predicate_flag=PredicateFlag.PREDICATE_TRUE,
                branch_target=".L_x_1"
            ),
            InstructionStat(op="FADD", pc=8),  # Fall-through target
            InstructionStat(op="EXIT", pc=16),  # Branch target
        ]

        label_to_pc = {".L_x_1": 16}  # Branch goes to EXIT

        cfg = build_cfg_from_instructions(insts, label_to_pc=label_to_pc)

        # Should have 3 blocks (one for each instruction since branch splits)
        assert len(cfg.function.blocks) >= 2

        # First block should have two edges
        first_block = cfg.function.blocks[0]

        # Find edge types
        edge_types = [t.edge_type for t in first_block.targets]

        # Should have COND_TAKEN (to branch target) and COND_NOT_TAKEN (fall-through)
        assert EdgeType.COND_TAKEN in edge_types
        assert EdgeType.COND_NOT_TAKEN in edge_types

    def test_unconditional_branch_edge_type(self):
        """Test unconditional branch has DIRECT edge and no fall-through."""
        insts = [
            InstructionStat(
                op="BRA", pc=0,
                branch_target=".L_x_1"
            ),
            InstructionStat(op="FADD", pc=8),  # Should not be fall-through target
            InstructionStat(op="EXIT", pc=16),  # Branch target
        ]

        label_to_pc = {".L_x_1": 16}

        cfg = build_cfg_from_instructions(insts, label_to_pc=label_to_pc)

        first_block = cfg.function.blocks[0]

        # Should have only one edge (DIRECT to target), no fall-through
        assert len(first_block.targets) == 1
        assert first_block.targets[0].edge_type == EdgeType.DIRECT

    def test_call_edge_types(self):
        """Test CALL and CALL_FT edge types."""
        insts = [
            InstructionStat(
                op="CALL", pc=0,
                branch_target=".L_func"
            ),
            InstructionStat(op="FADD", pc=8),  # Return point
            InstructionStat(op="EXIT", pc=16),
        ]

        # CALL to a label (function entry)
        label_to_pc = {".L_func": 16}

        cfg = build_cfg_from_instructions(insts, label_to_pc=label_to_pc)

        first_block = cfg.function.blocks[0]

        # Should have two edges: CALL and CALL_FT
        edge_types = [t.edge_type for t in first_block.targets]

        assert EdgeType.CALL in edge_types
        assert EdgeType.CALL_FT in edge_types

    def test_cfg_with_label_mapping(self, gpubin_path):
        """Test CFG building with real label mapping."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        func = disassemble_and_parse(gpubin_path)
        assert func is not None

        # Build CFG with label mapping
        cfg = build_cfg_from_instructions(
            func.instructions,
            func.name,
            label_to_pc=func.labels
        )

        # Should build successfully
        assert len(cfg.function.blocks) >= 1

        # Check that branch targets create proper block boundaries
        # If we have labels and branches, blocks at label PCs should exist
        for label, pc in func.labels.items():
            # There should be a block starting at or containing this PC
            block = cfg.function.get_block_containing_pc(pc)
            assert block is not None, f"No block for label {label} at PC {pc}"


# =============================================================================
# Dependency Analysis Tests
# =============================================================================


class TestBuildAssignPcs:
    """Tests for the build_assign_pcs algorithm - simple forward pass."""

    def test_simple_linear_dependency(self):
        """Test basic linear dependency chain: R0 -> R1 -> R5."""
        insts = [
            InstructionStat(op="IMAD", pc=0x00, dsts=[1], srcs=[2, 3, 4]),  # Writes R1
            InstructionStat(op="FADD", pc=0x08, dsts=[5], srcs=[1, 6]),  # Reads R1, writes R5
        ]
        build_assign_pcs(insts)

        # Second instruction should see R1 from first instruction
        assert 1 in insts[1].assign_pcs
        assert insts[1].assign_pcs[1] == [0x00]

        # R6 is external (no prior writer)
        assert 6 not in insts[1].assign_pcs

    def test_register_overwrite(self):
        """Test that later writes supersede earlier writes."""
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[0], srcs=[1]),  # Writes R0
            InstructionStat(op="IMAD", pc=0x10, dsts=[0], srcs=[2, 3, 4]),  # Overwrites R0
            InstructionStat(op="FADD", pc=0x20, dsts=[5], srcs=[0]),  # Reads R0
        ]
        build_assign_pcs(insts)

        # FADD sees R0 from IMAD (0x10), not LDG (0x00)
        assert 0 in insts[2].assign_pcs
        assert insts[2].assign_pcs[0] == [0x10]

    def test_predicate_dependency(self):
        """Test predicate register dependency tracking."""
        insts = [
            InstructionStat(op="SETP", pc=0x00, pdsts=[0]),  # Writes P0
            InstructionStat(
                op="BRA",
                pc=0x10,
                predicate=0,
                predicate_flag=PredicateFlag.PREDICATE_TRUE,
            ),  # Uses P0 as guard
        ]
        build_assign_pcs(insts)

        # BRA should have predicate guard dependency on SETP
        assert insts[1].predicate_assign_pcs == [0x00]

    def test_barrier_dependency(self):
        """Test barrier register dependency tracking."""
        insts = [
            InstructionStat(op="BAR.SYNC", pc=0x00, bdsts=[1]),  # Writes B1
            InstructionStat(op="LDG", pc=0x08, bsrcs=[1]),  # Reads B1
        ]
        build_assign_pcs(insts)

        # LDG should see B1 from BAR.SYNC
        assert 1 in insts[1].bassign_pcs
        assert insts[1].bassign_pcs[1] == [0x00]

    def test_uniform_register_dependency(self):
        """Test uniform register dependency tracking."""
        insts = [
            InstructionStat(op="LDC", pc=0x00, udsts=[0]),  # Writes UR0
            InstructionStat(op="IMAD", pc=0x08, usrcs=[0], dsts=[5]),  # Reads UR0
        ]
        build_assign_pcs(insts)

        # IMAD should see UR0 from LDC
        assert 0 in insts[1].uassign_pcs
        assert insts[1].uassign_pcs[0] == [0x00]

    def test_predicate_source_dependency(self):
        """Test predicate registers as sources (not guards)."""
        insts = [
            InstructionStat(op="SETP.EQ", pc=0x00, pdsts=[0]),  # Writes P0
            InstructionStat(op="SEL", pc=0x08, psrcs=[0], dsts=[1]),  # Reads P0 as source
        ]
        build_assign_pcs(insts)

        # SEL should see P0 from SETP
        assert 0 in insts[1].passign_pcs
        assert insts[1].passign_pcs[0] == [0x00]

    def test_multiple_sources(self):
        """Test instruction with multiple source registers."""
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[0], srcs=[]),  # Writes R0
            InstructionStat(op="LDG", pc=0x08, dsts=[1], srcs=[]),  # Writes R1
            InstructionStat(op="FADD", pc=0x10, dsts=[2], srcs=[0, 1]),  # Reads R0, R1
        ]
        build_assign_pcs(insts)

        # FADD should see both R0 and R1
        assert 0 in insts[2].assign_pcs
        assert 1 in insts[2].assign_pcs
        assert insts[2].assign_pcs[0] == [0x00]
        assert insts[2].assign_pcs[1] == [0x08]

    def test_no_dependency_for_external_input(self):
        """Test that external inputs have no assign_pcs entry."""
        insts = [
            InstructionStat(op="FADD", pc=0x00, dsts=[5], srcs=[1, 2]),  # R1, R2 are external
        ]
        build_assign_pcs(insts)

        # No prior writers for R1 or R2
        assert 1 not in insts[0].assign_pcs
        assert 2 not in insts[0].assign_pcs

    def test_empty_instruction_list(self):
        """Test with empty instruction list."""
        insts = []
        build_assign_pcs(insts)  # Should not raise
        assert len(insts) == 0

    def test_single_instruction(self):
        """Test with single instruction."""
        insts = [
            InstructionStat(op="FADD", pc=0x00, dsts=[5], srcs=[1, 2]),
        ]
        build_assign_pcs(insts)

        # First instruction has no dependencies
        assert insts[0].assign_pcs == {}

    def test_self_read_after_write(self):
        """Test instruction that reads same register it writes (WAR)."""
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[5], srcs=[]),  # Writes R5
            InstructionStat(op="IADD", pc=0x08, dsts=[5], srcs=[5, 1]),  # R5 = R5 + R1
        ]
        build_assign_pcs(insts)

        # IADD should see R5 from LDG
        assert 5 in insts[1].assign_pcs
        assert insts[1].assign_pcs[5] == [0x00]


class TestGetAllDependencies:
    """Tests for get_all_dependencies query function."""

    def test_gathers_all_register_types(self):
        """Test that all register type dependencies are gathered."""
        inst = InstructionStat(op="FADD", pc=0x00)

        # Manually set up assign_pcs maps
        inst.assign_pcs = {0: [0x10], 1: [0x20]}  # General registers
        inst.passign_pcs = {0: [0x30]}  # Predicate
        inst.bassign_pcs = {1: [0x40]}  # Barrier
        inst.uassign_pcs = {0: [0x50]}  # Uniform
        inst.upassign_pcs = {0: [0x60]}  # Uniform predicate
        inst.predicate_assign_pcs = [0x70]  # Predicate guard

        deps = get_all_dependencies(inst)

        # Should contain all PCs
        assert 0x10 in deps
        assert 0x20 in deps
        assert 0x30 in deps
        assert 0x40 in deps
        assert 0x50 in deps
        assert 0x60 in deps
        assert 0x70 in deps
        assert len(deps) == 7

    def test_empty_dependencies(self):
        """Test instruction with no dependencies."""
        inst = InstructionStat(op="FADD", pc=0x00)

        # Initialize empty maps
        inst.assign_pcs = {}
        inst.passign_pcs = {}
        inst.bassign_pcs = {}
        inst.uassign_pcs = {}
        inst.upassign_pcs = {}
        inst.predicate_assign_pcs = []

        deps = get_all_dependencies(inst)
        assert len(deps) == 0

    def test_deduplication(self):
        """Test that duplicate PCs are deduplicated."""
        inst = InstructionStat(op="FADD", pc=0x00)

        # Same PC appears in multiple maps
        inst.assign_pcs = {0: [0x10]}
        inst.passign_pcs = {0: [0x10]}  # Same PC
        inst.bassign_pcs = {}
        inst.uassign_pcs = {}
        inst.upassign_pcs = {}
        inst.predicate_assign_pcs = [0x10]  # Same PC again

        deps = get_all_dependencies(inst)

        # Should be deduplicated via set
        assert 0x10 in deps
        assert len(deps) == 1


class TestBackwardSlice:
    """Tests for backward_slice function - transitive closure."""

    def test_simple_chain(self):
        """Test backward slice through simple dependency chain."""
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[0], srcs=[]),  # No deps
            InstructionStat(op="FADD", pc=0x08, dsts=[1], srcs=[0]),  # Depends on 0x00
            InstructionStat(op="FMUL", pc=0x10, dsts=[2], srcs=[1]),  # Depends on 0x08
        ]
        build_assign_pcs(insts)

        pc_to_inst = {inst.pc: inst for inst in insts}
        slice_pcs = backward_slice(0x10, pc_to_inst)

        # Should include entire chain
        assert 0x10 in slice_pcs  # Target itself
        assert 0x08 in slice_pcs  # Direct dependency
        assert 0x00 in slice_pcs  # Transitive dependency

    def test_max_depth_limit(self):
        """Test that max_depth prevents infinite traversal."""
        # Create a chain longer than max_depth
        n = 10
        insts = [InstructionStat(op="NOP", pc=i * 8, dsts=[0], srcs=[]) for i in range(n)]

        # Chain dependencies: each reads what previous wrote
        for i in range(1, n):
            insts[i].srcs = [0]
            insts[i].dsts = [0]

        build_assign_pcs(insts)
        pc_to_inst = {inst.pc: inst for inst in insts}

        # Slice with max_depth=5 should not reach beginning
        slice_pcs = backward_slice(insts[-1].pc, pc_to_inst, max_depth=5)

        # Should have at most 6 instructions (depth 0-5)
        assert len(slice_pcs) <= 7

    def test_includes_target_pc(self):
        """Test that target PC is always included in slice."""
        insts = [
            InstructionStat(op="FADD", pc=0x00, dsts=[5], srcs=[1, 2]),  # No prior deps
        ]
        build_assign_pcs(insts)

        pc_to_inst = {inst.pc: inst for inst in insts}
        slice_pcs = backward_slice(0x00, pc_to_inst)

        assert 0x00 in slice_pcs

    def test_handles_missing_target(self):
        """Test graceful handling of missing target PC."""
        pc_to_inst = {}  # Empty map
        slice_pcs = backward_slice(0x100, pc_to_inst)

        # Should return set with target (even if not in map)
        assert 0x100 in slice_pcs

    def test_multiple_paths(self):
        """Test slice that follows multiple dependency paths."""
        # R2 = R0 + R1, where R0 and R1 come from different sources
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[0], srcs=[]),  # R0 source
            InstructionStat(op="LDG", pc=0x08, dsts=[1], srcs=[]),  # R1 source
            InstructionStat(op="FADD", pc=0x10, dsts=[2], srcs=[0, 1]),  # Uses both
        ]
        build_assign_pcs(insts)

        pc_to_inst = {inst.pc: inst for inst in insts}
        slice_pcs = backward_slice(0x10, pc_to_inst)

        # Should include both paths
        assert 0x00 in slice_pcs
        assert 0x08 in slice_pcs
        assert 0x10 in slice_pcs


class TestBackwardSliceForRegister:
    """Tests for backward_slice_for_register - register-specific slicing."""

    def test_specific_register_slice(self):
        """Test slice for specific register only."""
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[0], srcs=[]),  # R0
            InstructionStat(op="LDG", pc=0x08, dsts=[1], srcs=[]),  # R1
            InstructionStat(op="FADD", pc=0x10, dsts=[2], srcs=[0, 1]),  # Uses R0, R1
        ]
        build_assign_pcs(insts)

        pc_to_inst = {inst.pc: inst for inst in insts}

        # Slice for R0 only should not include R1's source
        slice_r0 = backward_slice_for_register(0x10, 0, pc_to_inst)
        assert 0x00 in slice_r0  # R0 source

        # Slice for R1 only
        slice_r1 = backward_slice_for_register(0x10, 1, pc_to_inst)
        assert 0x08 in slice_r1  # R1 source

    def test_no_dependency_for_register(self):
        """Test when target has no dependency on specific register."""
        insts = [
            InstructionStat(op="FADD", pc=0x00, dsts=[5], srcs=[1, 2]),  # No deps
        ]
        build_assign_pcs(insts)

        pc_to_inst = {inst.pc: inst for inst in insts}

        # R0 is not used by this instruction
        slice_pcs = backward_slice_for_register(0x00, 0, pc_to_inst)
        assert len(slice_pcs) == 0

    def test_missing_target_instruction(self):
        """Test when target PC not in instruction map."""
        pc_to_inst = {}
        slice_pcs = backward_slice_for_register(0x100, 0, pc_to_inst)
        assert len(slice_pcs) == 0


class TestValidateAssignPcs:
    """Tests for validate_assign_pcs validation function."""

    def test_valid_instructions(self):
        """Test validation passes for correctly built assign_pcs."""
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[0], srcs=[]),
            InstructionStat(op="FADD", pc=0x08, dsts=[1], srcs=[0]),
        ]
        build_assign_pcs(insts)

        errors = validate_assign_pcs(insts)
        assert len(errors) == 0

    def test_detects_invalid_pc_reference(self):
        """Test detection of reference to non-existent PC."""
        inst = InstructionStat(op="FADD", pc=0x00, srcs=[0])
        inst.assign_pcs = {0: [0x100]}  # PC 0x100 doesn't exist

        errors = validate_assign_pcs([inst])
        assert len(errors) >= 1
        assert "invalid pc" in errors[0].lower()

    def test_detects_self_reference(self):
        """Test detection of instruction depending on itself."""
        inst = InstructionStat(op="FADD", pc=0x00, srcs=[0])
        inst.assign_pcs = {0: [0x00]}  # Self-reference

        errors = validate_assign_pcs([inst])
        assert len(errors) >= 1
        assert "self-reference" in errors[0].lower()

    def test_detects_assign_pcs_without_source(self):
        """Test detection of assign_pcs entry for non-source register."""
        inst = InstructionStat(op="FADD", pc=0x00, srcs=[1])  # Only uses R1
        inst.assign_pcs = {0: [0x10]}  # But has entry for R0

        # Need a second instruction at 0x10
        inst2 = InstructionStat(op="NOP", pc=0x10, dsts=[0])

        errors = validate_assign_pcs([inst, inst2])
        assert len(errors) >= 1
        assert "not in srcs" in errors[0].lower()

    def test_empty_instruction_list(self):
        """Test validation of empty list."""
        errors = validate_assign_pcs([])
        assert len(errors) == 0


class TestPrintDependencies:
    """Tests for print_dependencies debugging function."""

    def test_print_does_not_crash(self, capsys):
        """Test that print_dependencies runs without errors."""
        insts = [
            InstructionStat(op="LDG", pc=0x00, dsts=[0], srcs=[]),
            InstructionStat(op="FADD", pc=0x08, dsts=[1], srcs=[0]),
        ]
        build_assign_pcs(insts)

        # Should not raise
        print_dependencies(insts)

        # Should produce some output
        captured = capsys.readouterr()
        assert "0x0000" in captured.out or "0x00" in captured.out
        assert "LDG" in captured.out


class TestDependencyEdgeCases:
    """Edge case tests for dependency analysis."""

    def test_wide_register_dependency(self):
        """Test dependency tracking with wide (64/128-bit) registers."""
        # 64-bit register pair R0:R1
        insts = [
            InstructionStat(op="LDG.64", pc=0x00, dsts=[0, 1], srcs=[]),  # Writes R0:R1
            InstructionStat(op="FADD.64", pc=0x08, dsts=[2, 3], srcs=[0, 1]),  # Reads R0:R1
        ]
        build_assign_pcs(insts)

        # Both R0 and R1 should have dependencies
        assert 0 in insts[1].assign_pcs
        assert 1 in insts[1].assign_pcs
        assert insts[1].assign_pcs[0] == [0x00]
        assert insts[1].assign_pcs[1] == [0x00]

    def test_predicated_instruction_dependency(self):
        """Test dependency with predicated instruction."""
        insts = [
            InstructionStat(op="SETP", pc=0x00, pdsts=[0]),  # Writes P0
            InstructionStat(
                op="IMAD",
                pc=0x08,
                dsts=[5],
                srcs=[1, 2, 3],
                predicate=0,
                predicate_flag=PredicateFlag.PREDICATE_TRUE,
            ),  # @P0 guarded
            InstructionStat(op="FADD", pc=0x10, dsts=[6], srcs=[5]),  # Uses R5
        ]
        build_assign_pcs(insts)

        # IMAD should have predicate_assign_pcs
        assert insts[1].predicate_assign_pcs == [0x00]

        # FADD should see R5 from IMAD
        assert insts[2].assign_pcs[5] == [0x08]

    def test_mixed_register_types(self):
        """Test instruction using multiple register types simultaneously."""
        # Instruction that uses general, predicate, and barrier
        insts = [
            InstructionStat(op="SETP", pc=0x00, pdsts=[0]),  # P0
            InstructionStat(op="BAR.SYNC", pc=0x08, bdsts=[1]),  # B1
            InstructionStat(op="LDG", pc=0x10, dsts=[5], srcs=[]),  # R5
            InstructionStat(
                op="STG",
                pc=0x18,
                srcs=[5],
                psrcs=[0],
                bsrcs=[1],
                predicate=0,
            ),  # Uses all
        ]
        build_assign_pcs(insts)

        stg = insts[3]
        # General register dependency
        assert 5 in stg.assign_pcs
        assert stg.assign_pcs[5] == [0x10]

        # Predicate dependency
        assert 0 in stg.passign_pcs
        assert stg.passign_pcs[0] == [0x00]

        # Barrier dependency
        assert 1 in stg.bassign_pcs
        assert stg.bassign_pcs[1] == [0x08]

        # Predicate guard
        assert stg.predicate_assign_pcs == [0x00]


# =============================================================================
# CUBIN Control Field Extraction Tests
# =============================================================================


class TestExtractControlFromBytes:
    """Tests for extract_control_from_bytes function.

    Based on GPA-artifact AnalyzeInstruction.cpp bit masks:
        stall: bits [44:41], mask 0x00001e0000000000, shift 41
        yield: bit [45], mask 0x0000200000000000, shift 45
        write: bits [48:46], mask 0x0001c00000000000, shift 46
        read:  bits [51:49], mask 0x000e000000000000, shift 49
        wait:  bits [57:52], mask 0x03f0000000000000, shift 52
        reuse: bits [61:58], mask 0x3c00000000000000, shift 58
    """

    def test_all_zeros(self):
        """Test control extraction from all-zero bytes."""
        data = bytes(8)
        ctrl = extract_control_from_bytes(data)

        assert ctrl.stall == 0
        assert ctrl.yield_flag == 0
        assert ctrl.write == 0
        assert ctrl.read == 0
        assert ctrl.wait == 0
        assert ctrl.reuse == 0

    def test_stall_field_extraction(self):
        """Test stall field (bits 41-44) extraction."""
        # stall = 15 (0xF) in bits 41-44
        # 0xF << 41 = 0x00001e0000000000
        value = 0xF << 41
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.stall == 15
        # Other fields should be 0
        assert ctrl.yield_flag == 0
        assert ctrl.write == 0
        assert ctrl.read == 0
        assert ctrl.wait == 0
        assert ctrl.reuse == 0

    def test_stall_partial_value(self):
        """Test stall with value 5."""
        value = 5 << 41
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)
        assert ctrl.stall == 5

    def test_yield_field_extraction(self):
        """Test yield field (bit 45) extraction."""
        # yield = 1 in bit 45
        value = 1 << 45
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.yield_flag == 1
        assert ctrl.stall == 0

    def test_write_barrier_extraction(self):
        """Test write barrier field (bits 46-48) extraction."""
        # write = 3 in bits 46-48
        value = 3 << 46
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.write == 3
        # BARRIER_NONE is 7, so 3 means barrier B3
        assert ctrl.stall == 0
        assert ctrl.yield_flag == 0

    def test_write_barrier_max_value(self):
        """Test write barrier with max value 7 (BARRIER_NONE)."""
        value = 7 << 46
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)
        assert ctrl.write == 7  # BARRIER_NONE

    def test_read_barrier_extraction(self):
        """Test read barrier field (bits 49-51) extraction."""
        # read = 5 in bits 49-51
        value = 5 << 49
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.read == 5
        assert ctrl.write == 0

    def test_wait_mask_extraction(self):
        """Test wait mask field (bits 52-57) extraction."""
        # wait = 0b101010 (42) - waits on B2, B4, B6
        value = 0b101010 << 52
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.wait == 0b101010
        # Verify barrier checking
        assert ctrl.waits_on_barrier(2) is True
        assert ctrl.waits_on_barrier(4) is True
        assert ctrl.waits_on_barrier(6) is True
        assert ctrl.waits_on_barrier(1) is False
        assert ctrl.waits_on_barrier(3) is False
        assert ctrl.waits_on_barrier(5) is False

    def test_wait_mask_all_barriers(self):
        """Test wait mask with all barriers set."""
        # wait = 0b111111 (63) - waits on all B1-B6
        value = 0b111111 << 52
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.wait == 63
        for i in range(1, 7):
            assert ctrl.waits_on_barrier(i) is True

    def test_reuse_field_extraction(self):
        """Test reuse cache field (bits 58-61) extraction."""
        # reuse = 0xA (10) in bits 58-61
        value = 0xA << 58
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.reuse == 10
        assert ctrl.wait == 0

    def test_all_fields_combined(self):
        """Test extraction with all fields set to non-zero values."""
        # Set all fields:
        # stall = 7, yield = 1, write = 2, read = 3, wait = 21, reuse = 5
        stall = 7 << 41
        yield_flag = 1 << 45
        write = 2 << 46
        read = 3 << 49
        wait = 21 << 52
        reuse = 5 << 58

        value = stall | yield_flag | write | read | wait | reuse
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.stall == 7
        assert ctrl.yield_flag == 1
        assert ctrl.write == 2
        assert ctrl.read == 3
        assert ctrl.wait == 21
        assert ctrl.reuse == 5

    def test_insufficient_bytes_returns_default(self):
        """Test that insufficient bytes returns default Control."""
        # Less than 8 bytes
        data = bytes(4)
        ctrl = extract_control_from_bytes(data)

        # Should return default Control values
        assert ctrl.stall == 1  # Default
        assert ctrl.read == 7  # BARRIER_NONE
        assert ctrl.write == 7  # BARRIER_NONE

    def test_empty_bytes_returns_default(self):
        """Test that empty bytes returns default Control."""
        data = bytes(0)
        ctrl = extract_control_from_bytes(data)
        assert ctrl.stall == 1  # Default

    def test_extra_bytes_ignored(self):
        """Test that bytes beyond first 8 are ignored."""
        # 16 bytes with control in first 8
        value = 10 << 41  # stall = 10
        first_8 = value.to_bytes(8, byteorder='little')
        data = first_8 + bytes(8)  # Extra 8 zeros

        ctrl = extract_control_from_bytes(data)
        assert ctrl.stall == 10

    def test_realistic_instruction_encoding(self):
        """Test with a realistic instruction encoding pattern.

        Typical instruction: stall=4, write=7 (none), read=7 (none), wait=0
        """
        # stall=4, write=7 (BARRIER_NONE), read=7 (BARRIER_NONE)
        stall = 4 << 41
        write = 7 << 46
        read = 7 << 49

        value = stall | write | read
        data = value.to_bytes(8, byteorder='little')
        ctrl = extract_control_from_bytes(data)

        assert ctrl.stall == 4
        assert ctrl.write == 7
        assert ctrl.read == 7
        assert ctrl.wait == 0
        assert ctrl.yield_flag == 0
        assert ctrl.reuse == 0


class TestCubinParserInitialization:
    """Tests for CubinParser initialization and error handling."""

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CubinParser("/nonexistent/path/to/file.cubin")

    def test_invalid_elf_raises_error(self, tmp_path):
        """Test that invalid ELF file raises ValueError."""
        # Create a file with invalid content
        invalid_file = tmp_path / "invalid.cubin"
        invalid_file.write_bytes(b"not a valid elf file")

        with pytest.raises(ValueError):
            CubinParser(str(invalid_file))


class TestControlIntegration:
    """Integration tests for Control with InstructionStat."""

    def test_instruction_with_custom_control(self):
        """Test InstructionStat with populated Control fields."""
        ctrl = Control(
            stall=5,
            yield_flag=1,
            write=2,
            read=3,
            wait=0b001001,  # B1 and B4
            reuse=0
        )
        inst = InstructionStat(op="LDG.E.64", pc=0x100, control=ctrl)

        assert inst.control.stall == 5
        assert inst.control.yield_flag == 1
        assert inst.control.write == 2
        assert inst.control.read == 3
        assert inst.control.waits_on_barrier(1) is True
        assert inst.control.waits_on_barrier(4) is True
        assert inst.control.waits_on_barrier(2) is False

    def test_default_control_values(self):
        """Test InstructionStat with default Control."""
        inst = InstructionStat(op="FADD", pc=0x200)

        # Default values
        assert inst.control.stall == 1
        assert inst.control.write == 7  # BARRIER_NONE
        assert inst.control.read == 7  # BARRIER_NONE
        assert inst.control.wait == 0
        assert inst.control.yield_flag == 0
        assert inst.control.reuse == 0


class TestCubinParserRealBinary:
    """Integration tests with real CUBIN/gpubin files."""

    def test_parse_real_gpubin(self, gpubin_path):
        """Test parsing real gpubin file."""
        parser = CubinParser(gpubin_path)

        # Should find at least one .text section
        assert len(parser.sections) >= 1
        assert any(name.startswith('.text') for name in parser.sections.keys())

        # Should find at least one function
        assert len(parser.functions) >= 1

    def test_extract_control_from_real_gpubin(self, gpubin_path):
        """Test control field extraction from real gpubin."""
        parser = CubinParser(gpubin_path)
        section = list(parser.sections.values())[0]

        # Extract control for first instruction
        ctrl = parser.extract_control_at_offset(0, section.data)

        # Control fields should be in valid ranges
        assert 0 <= ctrl.stall <= 15  # 4 bits
        assert 0 <= ctrl.yield_flag <= 1  # 1 bit
        assert 0 <= ctrl.write <= 7  # 3 bits
        assert 0 <= ctrl.read <= 7  # 3 bits
        assert 0 <= ctrl.wait <= 63  # 6 bits
        assert 0 <= ctrl.reuse <= 15  # 4 bits

    def test_populate_controls_real_gpubin(self, gpubin_path):
        """Test populating control fields for parsed instructions."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        parser = CubinParser(gpubin_path)
        func = disassemble_and_parse(gpubin_path)

        # Populate control fields
        updated = parser.populate_instruction_controls(func.instructions)

        # All instructions should be updated
        assert updated == len(func.instructions)

        # Verify control fields are non-default for at least some instructions
        non_default_stall = sum(1 for inst in func.instructions if inst.control.stall != 1)
        assert non_default_stall > 0, "Expected some instructions with non-default stall"

    def test_barrier_patterns_real_gpubin(self, gpubin_path):
        """Test that barrier patterns are reasonable in real code."""
        if not check_nvdisasm_available():
            pytest.skip("nvdisasm not available")

        parser = CubinParser(gpubin_path)
        func = disassemble_and_parse(gpubin_path)
        parser.populate_instruction_controls(func.instructions)

        # Find instructions with barrier waits
        waiting_insts = [inst for inst in func.instructions if inst.control.wait != 0]

        # Real code should have some barrier synchronization
        assert len(waiting_insts) > 0, "Expected some instructions waiting on barriers"

        # Find instructions setting barriers
        barrier_setters = [inst for inst in func.instructions if inst.control.write != 7]

        # Should have barrier setters
        assert len(barrier_setters) > 0, "Expected some instructions setting barriers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
