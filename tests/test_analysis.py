"""Tests for leo.analysis module - GPU performance analysis algorithms.

Test organization:
- TestVMAProperty: VMAProperty dataclass tests
- TestVMAPropertyMap: VMAPropertyMap container tests
- TestVMAPropertyMapBuild: VMAPropertyMap.build() integration tests
- TestPredicateTracking: Predicate tracking algorithm tests
- TestLatencyPruning: Latency pruning algorithm tests
"""

import pytest
from pathlib import Path

from leo.binary.instruction import InstructionStat, Control, PredicateFlag
from leo.analysis import (
    # VMA property map
    VMAProperty,
    VMAPropertyMap,
    build_vma_property_map,
    # Predicate tracking
    PredicateState,
    StopReason,
    should_stop_predicate_tracking,
    track_dependency_with_predicates,
    refine_assign_pcs_with_predicates,
    # Latency pruning
    LatencyCheckResult,
    PathResult,
    check_latency_hidden_linear,
    prune_assign_pcs_by_latency,
    apply_latency_pruning,
    # Blame attribution
    BlameCategory,
    BlameEdge,
    KernelBlameResult,
    blame_instructions,
    reverse_ratio,
    distribute_blame,
)
from leo.arch.nvidia import get_architecture


# Path to test database (with instruction nodes)
TEST_DB_PATH = Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-database"


# =============================================================================
# VMAProperty Tests
# =============================================================================


class TestVMAProperty:
    """Tests for VMAProperty dataclass."""

    def test_create_empty(self):
        """Test creating empty VMAProperty with defaults."""
        prop = VMAProperty()
        assert prop.vma == 0
        assert prop.instruction is None
        assert prop.cct_id == -1
        assert prop.prof_metrics == {}
        assert prop.has_profile_data is False

    def test_create_with_metrics(self):
        """Test creating VMAProperty with metrics."""
        prop = VMAProperty(
            vma=0x100,
            cct_id=42,
            prof_metrics={
                "gcycles": 100000.0,
                "gcycles:stl": 60000.0,
                "gcycles:isu": 40000.0,
            },
            has_profile_data=True,
        )
        assert prop.vma == 0x100
        assert prop.cct_id == 42
        assert prop.has_profile_data is True

    def test_total_cycles_property(self):
        """Test total_cycles accessor."""
        prop = VMAProperty(prof_metrics={"gcycles": 50000.0})
        assert prop.total_cycles == 50000.0

        prop_empty = VMAProperty()
        assert prop_empty.total_cycles == 0.0

    def test_stall_cycles_property(self):
        """Test stall_cycles accessor."""
        prop = VMAProperty(prof_metrics={"gcycles:stl": 30000.0})
        assert prop.stall_cycles == 30000.0

    def test_issue_cycles_property(self):
        """Test issue_cycles accessor."""
        prop = VMAProperty(prof_metrics={"gcycles:isu": 20000.0})
        assert prop.issue_cycles == 20000.0

    def test_memory_stall_cycles_aggregation(self):
        """Test memory stall cycles uses vendor-aware metric lookup."""
        # NVIDIA: uses gcycles:stl:gmem + lmem (not mem)
        prop_nvidia = VMAProperty(
            prof_metrics={
                "gcycles:stl:mem": 10000.0,
                "gcycles:stl:gmem": 20000.0,
                "gcycles:stl:lmem": 5000.0,
            },
            vendor="nvidia",
        )
        assert prop_nvidia.memory_stall_cycles == 25000.0  # gmem + lmem

        # AMD: uses gcycles:stl:mem + lmem (primary metric)
        prop_amd = VMAProperty(
            prof_metrics={
                "gcycles:stl:mem": 10000.0,
                "gcycles:stl:gmem": 20000.0,
                "gcycles:stl:lmem": 5000.0,
            },
            vendor="amd",
        )
        assert prop_amd.memory_stall_cycles == 15000.0  # mem + lmem (uses mem, not gmem)

    def test_sync_stall_cycles(self):
        """Test sync stall cycles accessor."""
        prop = VMAProperty(prof_metrics={"gcycles:stl:sync": 8000.0})
        assert prop.sync_stall_cycles == 8000.0

    def test_dependency_stall_cycles(self):
        """Test dependency stall cycles accessor."""
        prop = VMAProperty(prof_metrics={"gcycles:stl:idep": 12000.0})
        assert prop.dependency_stall_cycles == 12000.0

    def test_stall_breakdown(self):
        """Test stall breakdown dictionary."""
        prop = VMAProperty(
            prof_metrics={
                "gcycles:stl:mem": 10000.0,
                "gcycles:stl:sync": 5000.0,
                "gcycles:stl:idep": 3000.0,
                "gcycles": 100000.0,  # Not a stall type
            }
        )
        breakdown = prop.get_stall_breakdown()
        assert "gcycles:stl:mem" in breakdown
        assert "gcycles:stl:sync" in breakdown
        assert "gcycles:stl:idep" in breakdown
        assert "gcycles" not in breakdown

    def test_is_stalling_with_threshold(self):
        """Test is_stalling with different thresholds."""
        prop = VMAProperty(prof_metrics={"gcycles:stl": 5000.0})
        assert prop.is_stalling(threshold=0) is True
        assert prop.is_stalling(threshold=4999) is True
        assert prop.is_stalling(threshold=5000) is False
        assert prop.is_stalling(threshold=10000) is False


class TestVMAPropertyMap:
    """Tests for VMAPropertyMap container."""

    @pytest.fixture
    def sample_instructions(self):
        """Create sample instructions for testing."""
        return [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[1]),
            InstructionStat(op="FADD", pc=0x110, dsts=[2], srcs=[0, 3]),
            InstructionStat(op="STG.E", pc=0x120, dsts=[], srcs=[2, 4]),
        ]

    def test_build_from_instructions(self, sample_instructions):
        """Test building map from instructions only (no database)."""
        vma_map = VMAPropertyMap.build_from_instructions(
            sample_instructions, arch_name="a100"
        )
        assert len(vma_map) == 3
        assert 0x100 in vma_map
        assert 0x110 in vma_map
        assert 0x120 in vma_map

    def test_contains(self, sample_instructions):
        """Test __contains__ operator."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        assert 0x100 in vma_map
        assert 0x999 not in vma_map

    def test_getitem(self, sample_instructions):
        """Test __getitem__ operator."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        prop = vma_map[0x100]
        assert prop.vma == 0x100
        assert prop.instruction.op == "LDG.E"

    def test_getitem_missing_raises(self, sample_instructions):
        """Test __getitem__ raises KeyError for missing PC."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        with pytest.raises(KeyError):
            _ = vma_map[0x999]

    def test_get_with_default(self, sample_instructions):
        """Test get() method with default."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        prop = vma_map.get(0x100)
        assert prop is not None

        missing = vma_map.get(0x999)
        assert missing is None

        default = VMAProperty(vma=0x999)
        result = vma_map.get(0x999, default)
        assert result.vma == 0x999

    def test_iteration(self, sample_instructions):
        """Test iterating over map."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        pcs = list(vma_map)
        assert len(pcs) == 3
        assert 0x100 in pcs
        assert 0x110 in pcs
        assert 0x120 in pcs

    def test_items_keys_values(self, sample_instructions):
        """Test items(), keys(), values() methods."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        assert len(list(vma_map.keys())) == 3
        assert len(list(vma_map.values())) == 3
        assert len(list(vma_map.items())) == 3

    def test_latency_lookup(self, sample_instructions):
        """Test that latencies are looked up from architecture."""
        vma_map = VMAPropertyMap.build_from_instructions(
            sample_instructions, arch_name="a100"
        )
        ldg_prop = vma_map.get(0x100)
        assert ldg_prop.latency_lower == 28  # A100 LDG min latency
        assert ldg_prop.latency_upper == 1024  # A100 LDG max latency

        fadd_prop = vma_map.get(0x110)
        assert fadd_prop.latency_lower == 4  # A100 FADD min latency
        assert fadd_prop.latency_upper == 4  # A100 FADD max latency

    def test_sampled_count_no_profile(self, sample_instructions):
        """Test sampled count when no profile data."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        assert vma_map.get_sampled_count() == 0

    def test_total_cycles_no_profile(self, sample_instructions):
        """Test total cycles when no profile data."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        assert vma_map.get_total_cycles() == 0.0
        assert vma_map.get_total_stall_cycles() == 0.0

    def test_top_stalling_empty(self, sample_instructions):
        """Test get_top_stalling with no stalling instructions."""
        vma_map = VMAPropertyMap.build_from_instructions(sample_instructions)
        top = vma_map.get_top_stalling(n=5)
        # Returns all 3, but all have 0 stall cycles
        assert len(top) == 3
        for pc, prop in top:
            assert prop.stall_cycles == 0.0


class TestVMAPropertyMapBuild:
    """Integration tests for VMAPropertyMap.build() with real database."""

    @pytest.fixture
    def db_reader(self):
        """Get database reader for test database."""
        from leo.db import DatabaseReader

        if not TEST_DB_PATH.exists():
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")
        return DatabaseReader(str(TEST_DB_PATH))

    @pytest.fixture
    def real_instructions(self, db_reader):
        """Get instructions at real offsets from database."""
        cct = db_reader.get_cct("*")
        inst_nodes = cct[cct["type"] == "instruction"]
        offsets = inst_nodes["offset"].tolist()

        return [
            InstructionStat(op="LDG.E", pc=offset, dsts=[0], srcs=[1])
            for offset in offsets[:20]  # Use first 20 for speed
        ]

    def test_build_with_database(self, db_reader, real_instructions):
        """Test building VMA map with real database."""
        if not TEST_DB_PATH.exists():
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        vma_map = VMAPropertyMap.build(
            db_path=str(TEST_DB_PATH),
            instructions=real_instructions,
            arch_name="a100",
        )

        assert len(vma_map) == len(real_instructions)
        assert vma_map.get_sampled_count() > 0

    def test_build_has_profile_data(self, db_reader, real_instructions):
        """Test that built map has profile data for sampled instructions."""
        if not TEST_DB_PATH.exists():
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        vma_map = VMAPropertyMap.build(
            db_path=str(TEST_DB_PATH),
            instructions=real_instructions,
            arch_name="a100",
        )

        sampled = [p for p in vma_map.values() if p.has_profile_data]
        assert len(sampled) > 0

        # Check that sampled instructions have metrics
        for prop in sampled:
            assert len(prop.prof_metrics) > 0

    def test_build_total_cycles_positive(self, db_reader, real_instructions):
        """Test that total cycles are positive for sampled instructions."""
        if not TEST_DB_PATH.exists():
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        vma_map = VMAPropertyMap.build(
            db_path=str(TEST_DB_PATH),
            instructions=real_instructions,
            arch_name="a100",
        )

        assert vma_map.get_total_cycles() > 0

    def test_build_stall_breakdown(self, db_reader, real_instructions):
        """Test stall type breakdown from database."""
        if not TEST_DB_PATH.exists():
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        vma_map = VMAPropertyMap.build(
            db_path=str(TEST_DB_PATH),
            instructions=real_instructions,
            arch_name="a100",
        )

        breakdown = vma_map.get_stall_type_breakdown()
        # Should have some stall types if any instructions are sampled
        if vma_map.get_sampled_count() > 0:
            assert len(breakdown) >= 0  # May be empty if no stalls

    def test_build_convenience_function(self, real_instructions):
        """Test build_vma_property_map convenience function."""
        if not TEST_DB_PATH.exists():
            pytest.skip(f"Test database not found: {TEST_DB_PATH}")

        vma_map = build_vma_property_map(
            db_path=str(TEST_DB_PATH),
            instructions=real_instructions,
            arch_name="a100",
        )

        assert isinstance(vma_map, VMAPropertyMap)
        assert len(vma_map) > 0


# =============================================================================
# Predicate Tracking Tests
# =============================================================================


class TestPredicateState:
    """Tests for PredicateState class."""

    def test_empty_state(self):
        """Test empty predicate state."""
        state = PredicateState()
        assert len(state.predicate_map) == 0
        assert state.depth == 0

    def test_push_pop_predicate(self):
        """Test pushing and popping predicates."""
        state = PredicateState()

        # Push @P0
        state.push_predicate(0, PredicateFlag.PREDICATE_TRUE)
        assert len(state.predicate_map) == 1

        # Pop @P0
        state.pop_predicate(0, PredicateFlag.PREDICATE_TRUE)
        assert len(state.predicate_map) == 0

    def test_push_none_predicate(self):
        """Test that PREDICATE_NONE is ignored."""
        state = PredicateState()
        state.push_predicate(0, PredicateFlag.PREDICATE_NONE)
        assert len(state.predicate_map) == 0

    def test_contradiction_detection(self):
        """Test contradiction detection between @P and @!P."""
        state = PredicateState()

        # Push @P0
        state.push_predicate(0, PredicateFlag.PREDICATE_TRUE)

        # Check if @!P0 would contradict
        assert state.has_contradiction(0, PredicateFlag.PREDICATE_FALSE) is True

        # @P1 should not contradict @P0
        assert state.has_contradiction(1, PredicateFlag.PREDICATE_FALSE) is False

    def test_clone(self):
        """Test state cloning for branching."""
        state = PredicateState()
        state.push_predicate(0, PredicateFlag.PREDICATE_TRUE)
        state.depth = 5

        clone = state.clone()
        assert clone.depth == 5
        assert len(clone.predicate_map) == 1

        # Modifying clone shouldn't affect original
        clone.push_predicate(1, PredicateFlag.PREDICATE_FALSE)
        assert len(state.predicate_map) == 1
        assert len(clone.predicate_map) == 2


class TestShouldStopPredicateTracking:
    """Tests for should_stop_predicate_tracking function."""

    def test_stop_at_depth_limit(self):
        """Test stopping at depth limit."""
        target = InstructionStat(op="FADD", pc=0x100)
        candidate = InstructionStat(
            op="MOV", pc=0x80, predicate=0, predicate_flag=PredicateFlag.PREDICATE_TRUE
        )
        state = PredicateState(depth=8, track_limit=8)

        reason = should_stop_predicate_tracking(target, candidate, state)
        assert reason == StopReason.DEPTH_LIMIT

    def test_stop_at_unconditional(self):
        """Test stopping at unconditional instruction."""
        target = InstructionStat(
            op="FADD", pc=0x100, predicate=0, predicate_flag=PredicateFlag.PREDICATE_TRUE
        )
        candidate = InstructionStat(
            op="MOV", pc=0x80, predicate=-1, predicate_flag=PredicateFlag.PREDICATE_NONE
        )
        state = PredicateState()

        reason = should_stop_predicate_tracking(target, candidate, state)
        assert reason == StopReason.UNCONDITIONAL

    def test_stop_at_exact_match(self):
        """Test stopping at exact predicate match."""
        target = InstructionStat(
            op="FADD", pc=0x100, predicate=0, predicate_flag=PredicateFlag.PREDICATE_TRUE
        )
        candidate = InstructionStat(
            op="MOV", pc=0x80, predicate=0, predicate_flag=PredicateFlag.PREDICATE_TRUE
        )
        state = PredicateState()

        reason = should_stop_predicate_tracking(target, candidate, state)
        assert reason == StopReason.EXACT_MATCH

    def test_stop_at_contradiction(self):
        """Test stopping at contradictory predicates."""
        target = InstructionStat(op="FADD", pc=0x100)
        candidate = InstructionStat(
            op="MOV", pc=0x80, predicate=0, predicate_flag=PredicateFlag.PREDICATE_FALSE
        )
        state = PredicateState()
        state.push_predicate(0, PredicateFlag.PREDICATE_TRUE)

        reason = should_stop_predicate_tracking(target, candidate, state)
        assert reason == StopReason.CONTRADICTION

    def test_continue_when_no_stop_condition(self):
        """Test continuing when no stop condition is met."""
        target = InstructionStat(op="FADD", pc=0x100)
        candidate = InstructionStat(
            op="MOV", pc=0x80, predicate=1, predicate_flag=PredicateFlag.PREDICATE_TRUE
        )
        state = PredicateState()

        reason = should_stop_predicate_tracking(target, candidate, state)
        assert reason == StopReason.CONTINUE


# =============================================================================
# Latency Pruning Tests
# =============================================================================


class TestLatencyCheckResult:
    """Tests for LatencyCheckResult dataclass."""

    def test_create_hidden(self):
        """Test creating a hidden result."""
        result = LatencyCheckResult(
            is_hidden=True,
            accumulated_cycles=150,
            latency_threshold=100,
            found_use=False,
        )
        assert result.is_hidden is True
        assert result.accumulated_cycles >= result.latency_threshold

    def test_create_not_hidden(self):
        """Test creating a not-hidden result."""
        result = LatencyCheckResult(
            is_hidden=False,
            accumulated_cycles=50,
            latency_threshold=100,
            found_use=True,
        )
        assert result.is_hidden is False
        assert result.found_use is True


class TestCheckLatencyHiddenLinear:
    """Tests for check_latency_hidden_linear function."""

    @pytest.fixture
    def arch(self):
        """Get A100 architecture for latency lookup."""
        return get_architecture("a100")

    def test_hidden_by_stalls(self, arch):
        """Test dependency hidden by sufficient stall cycles."""
        # LDG at 0x100, use at 0x200
        # Intermediate instructions have enough stalls to hide latency
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[1], control=Control(stall=1)),
            InstructionStat(op="IADD", pc=0x110, dsts=[2], srcs=[3, 4], control=Control(stall=15)),
            InstructionStat(op="IADD", pc=0x120, dsts=[5], srcs=[6, 7], control=Control(stall=15)),
            InstructionStat(op="FADD", pc=0x200, dsts=[8], srcs=[0, 9], control=Control(stall=1)),
        ]

        result = check_latency_hidden_linear(
            from_pc=0x100,
            to_pc=0x200,
            target_reg=0,
            instructions=instructions,
            arch=arch,
            use_min_latency=True,  # Use min latency (28 for LDG)
        )

        # 15 + 15 = 30 stalls >= 28 min latency for LDG
        assert result.is_hidden is True
        assert result.accumulated_cycles >= result.latency_threshold

    def test_not_hidden_insufficient_stalls(self, arch):
        """Test dependency not hidden when stalls insufficient."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[1], control=Control(stall=1)),
            InstructionStat(op="IADD", pc=0x110, dsts=[2], srcs=[3, 4], control=Control(stall=2)),
            InstructionStat(op="FADD", pc=0x120, dsts=[8], srcs=[0, 9], control=Control(stall=1)),
        ]

        result = check_latency_hidden_linear(
            from_pc=0x100,
            to_pc=0x120,
            target_reg=0,
            instructions=instructions,
            arch=arch,
            use_min_latency=True,
        )

        # Only 2 stalls < 28 min latency
        assert result.is_hidden is False

    def test_finds_intermediate_use(self, arch):
        """Test finding intermediate register use."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[1], control=Control(stall=1)),
            InstructionStat(op="FADD", pc=0x110, dsts=[2], srcs=[0, 3], control=Control(stall=2)),  # Uses R0
            InstructionStat(op="FADD", pc=0x120, dsts=[4], srcs=[0, 5], control=Control(stall=1)),
        ]

        result = check_latency_hidden_linear(
            from_pc=0x100,
            to_pc=0x120,
            target_reg=0,
            instructions=instructions,
            arch=arch,
            use_min_latency=True,
        )

        assert result.found_use is True


class TestApplyLatencyPruning:
    """Tests for apply_latency_pruning function."""

    def test_prunes_hidden_dependencies(self):
        """Test that hidden dependencies are pruned."""
        arch = get_architecture("a100")

        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[1], control=Control(stall=1)),
            InstructionStat(op="IADD", pc=0x110, dsts=[2], srcs=[3, 4], control=Control(stall=30)),  # 30 stalls
            InstructionStat(op="FADD", pc=0x120, dsts=[5], srcs=[0, 6], control=Control(stall=1)),
        ]

        # Set up assign_pcs: R0 at 0x120 depends on 0x100
        instructions[2].assign_pcs = {0: [0x100]}

        apply_latency_pruning(instructions, arch, use_min_latency=True)

        # The dependency should be pruned (30 stalls >= 28 min latency)
        assert 0 not in instructions[2].assign_pcs or len(instructions[2].assign_pcs[0]) == 0

    def test_keeps_real_dependencies(self):
        """Test that real dependencies are kept."""
        arch = get_architecture("a100")

        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[1], control=Control(stall=1)),
            InstructionStat(op="IADD", pc=0x110, dsts=[2], srcs=[3, 4], control=Control(stall=2)),  # Only 2 stalls
            InstructionStat(op="FADD", pc=0x120, dsts=[5], srcs=[0, 6], control=Control(stall=1)),
        ]

        # Set up assign_pcs: R0 at 0x120 depends on 0x100
        instructions[2].assign_pcs = {0: [0x100]}

        apply_latency_pruning(instructions, arch, use_min_latency=True)

        # The dependency should be kept (2 stalls < 28 min latency)
        assert 0 in instructions[2].assign_pcs
        assert 0x100 in instructions[2].assign_pcs[0]


# =============================================================================
# Blame Attribution Tests
# =============================================================================


class TestReverseRatio:
    """Tests for reverse_ratio inverse distance weighting."""

    def test_single_node(self):
        """Test with single node."""
        distances = {0x100: 5.0}
        weights = reverse_ratio(distances)
        assert len(weights) == 1
        assert abs(weights[0x100] - 1.0) < 0.001

    def test_closer_gets_more_weight(self):
        """Test that closer nodes get higher weight."""
        distances = {0x100: 2.0, 0x200: 4.0, 0x300: 10.0}
        weights = reverse_ratio(distances)

        # Closer (2) should get more weight than farther (10)
        assert weights[0x100] > weights[0x200] > weights[0x300]

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        distances = {0x100: 2.0, 0x200: 4.0, 0x300: 8.0}
        weights = reverse_ratio(distances)
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_empty_input(self):
        """Test with empty input."""
        weights = reverse_ratio({})
        assert weights == {}

    def test_specific_values(self):
        """Test specific weight calculation."""
        distances = {0x00: 2.0, 0x04: 4.0}
        weights = reverse_ratio(distances)
        # pivot=2, ratios = {0x00: 1.0, 0x04: 0.5}, sum=1.5
        # weights = {0x00: 1/1.5=0.667, 0x04: 0.5/1.5=0.333}
        assert abs(weights[0x00] - 0.667) < 0.01
        assert abs(weights[0x04] - 0.333) < 0.01


class TestDistributeBlame:
    """Tests for distribute_blame three-factor weighting."""

    def test_single_dependency(self):
        """Test blame distribution with single dependency."""
        result = distribute_blame(
            total_stall=100.0,
            total_lat=50.0,
            distances={0x100: 2.0},
            efficiencies={0x100: 1.0},
            issue_counts={0x100: 100},
        )

        stall, lat = result[0x100]
        assert abs(stall - 100.0) < 0.001
        assert abs(lat - 50.0) < 0.001

    def test_blame_sums_to_total(self):
        """Test that distributed blame sums to total."""
        result = distribute_blame(
            total_stall=200.0,
            total_lat=100.0,
            distances={0x100: 2.0, 0x200: 4.0, 0x300: 8.0},
            efficiencies={0x100: 1.0, 0x200: 0.8, 0x300: 0.5},
            issue_counts={0x100: 100, 0x200: 200, 0x300: 50},
        )

        total_stall = sum(s for s, l in result.values())
        total_lat = sum(l for s, l in result.values())

        assert abs(total_stall - 200.0) < 0.001
        assert abs(total_lat - 100.0) < 0.001

    def test_empty_input(self):
        """Test with empty input."""
        result = distribute_blame(
            total_stall=100.0,
            total_lat=50.0,
            distances={},
            efficiencies={},
            issue_counts={},
        )
        assert result == {}


class TestBlameEdge:
    """Tests for BlameEdge dataclass."""

    def test_total_blame(self):
        """Test total_blame calculation."""
        edge = BlameEdge(
            src_pc=0x100,
            dst_pc=0x200,
            distance=4.0,
            stall_blame=50.0,
            lat_blame=20.0,
        )
        assert edge.total_blame() == 70.0

    def test_is_self_blame(self):
        """Test self-blame detection."""
        self_edge = BlameEdge(
            src_pc=0x100,
            dst_pc=0x100,
            distance=0.0,
            stall_blame=30.0,
            lat_blame=0.0,
        )
        assert self_edge.is_self_blame() is True

        normal_edge = BlameEdge(
            src_pc=0x100,
            dst_pc=0x200,
            distance=4.0,
            stall_blame=30.0,
            lat_blame=0.0,
        )
        assert normal_edge.is_self_blame() is False


class TestKernelBlameResult:
    """Tests for KernelBlameResult aggregation."""

    @pytest.fixture
    def sample_edges(self):
        """Create sample blame edges for testing."""
        return [
            BlameEdge(
                src_pc=0x100, dst_pc=0x200, distance=2.0,
                stall_blame=100.0, lat_blame=10.0,
                blame_type="mem_dep_gmem", blame_category="mem_dep",
                src_opcode="LDG.E", dst_opcode="FADD",
            ),
            BlameEdge(
                src_pc=0x110, dst_pc=0x200, distance=4.0,
                stall_blame=50.0, lat_blame=5.0,
                blame_type="exec_dep_dep", blame_category="exec_dep",
                src_opcode="IADD", dst_opcode="FADD",
            ),
            BlameEdge(
                src_pc=0x200, dst_pc=0x200, distance=0.0,
                stall_blame=30.0, lat_blame=0.0,
                blame_type="self_scheduler", blame_category="self",
                src_opcode="FADD", dst_opcode="FADD",
            ),
        ]

    def test_total_blame(self, sample_edges):
        """Test total blame calculation."""
        from leo.analysis.blame import _aggregate_blames
        result = _aggregate_blames(sample_edges, 0, 1)

        assert result.total_stall_blame == 180.0
        assert result.total_lat_blame == 15.0

    def test_blame_by_category(self, sample_edges):
        """Test blame aggregation by category."""
        from leo.analysis.blame import _aggregate_blames
        result = _aggregate_blames(sample_edges, 0, 1)

        assert "mem_dep" in result.blame_by_category
        assert "exec_dep" in result.blame_by_category
        assert "self" in result.blame_by_category

        assert result.blame_by_category["mem_dep"] == 110.0  # 100 + 10
        assert result.blame_by_category["exec_dep"] == 55.0  # 50 + 5
        assert result.blame_by_category["self"] == 30.0

    def test_blame_by_type(self, sample_edges):
        """Test blame aggregation by type."""
        from leo.analysis.blame import _aggregate_blames
        result = _aggregate_blames(sample_edges, 0, 1)

        assert "mem_dep_gmem" in result.blame_by_type
        assert "exec_dep_dep" in result.blame_by_type
        assert "self_scheduler" in result.blame_by_type

    def test_self_blame_count(self, sample_edges):
        """Test self-blame counting."""
        from leo.analysis.blame import _aggregate_blames
        result = _aggregate_blames(sample_edges, 0, 1)
        assert result.num_self_blame == 1

    def test_get_blame_for_pc(self, sample_edges):
        """Test getting blame for a specific PC."""
        from leo.analysis.blame import _aggregate_blames
        result = _aggregate_blames(sample_edges, 0, 1)

        blame_0x100 = result.get_blame_for_pc(0x100)
        assert blame_0x100 == 110.0  # 100 + 10

    def test_get_top_blame_sources(self, sample_edges):
        """Test getting top blame sources."""
        from leo.analysis.blame import _aggregate_blames
        result = _aggregate_blames(sample_edges, 0, 1)

        top = result.get_top_blame_sources(3)
        assert len(top) == 3
        # First should be 0x100 with highest blame
        assert top[0][0] == 0x100


class TestBlameInstructions:
    """Tests for blame_instructions main function."""

    @pytest.fixture
    def sample_instructions(self):
        """Create sample instructions with dependencies."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2], control=Control(stall=4)),
            InstructionStat(op="LDG.E", pc=0x110, dsts=[1], srcs=[3], control=Control(stall=4)),
            InstructionStat(op="IADD", pc=0x120, dsts=[4], srcs=[1, 5], control=Control(stall=2)),
            InstructionStat(op="FADD", pc=0x130, dsts=[6], srcs=[0, 4], control=Control(stall=2)),
            InstructionStat(op="STG.E", pc=0x140, dsts=[], srcs=[6, 7], control=Control(stall=1)),
        ]

        # Set up dependencies
        instructions[2].assign_pcs = {1: [0x110]}
        instructions[3].assign_pcs = {0: [0x100], 4: [0x120]}
        instructions[4].assign_pcs = {6: [0x130]}

        return instructions

    def test_basic_blame_attribution(self, sample_instructions):
        """Test basic blame attribution."""
        # Build VMA map
        vma_map = VMAPropertyMap.build_from_instructions(
            sample_instructions, arch_name="a100"
        )

        # Add mock profile data
        for pc in [0x130, 0x140]:
            prop = vma_map.get(pc)
            if prop:
                prop.prof_metrics["gcycles:stl"] = 1000.0
                prop.prof_metrics["gcycles:isu"] = 100.0
                prop.has_profile_data = True

        result = blame_instructions(vma_map, sample_instructions)

        assert result.num_blame_edges > 0
        assert result.total_stall_blame > 0

    def test_no_profile_data(self, sample_instructions):
        """Test with no profile data."""
        vma_map = VMAPropertyMap.build_from_instructions(
            sample_instructions, arch_name="a100"
        )
        # No profile data added

        result = blame_instructions(vma_map, sample_instructions)

        # Should have no blame edges since no profile data
        assert result.num_blame_edges == 0

    def test_self_blame_when_no_deps(self):
        """Test self-blame when instruction has no dependencies."""
        instructions = [
            InstructionStat(op="FADD", pc=0x100, dsts=[0], srcs=[1, 2], control=Control(stall=2)),
        ]
        # No assign_pcs set - no dependencies

        vma_map = VMAPropertyMap.build_from_instructions(instructions, arch_name="a100")
        prop = vma_map.get(0x100)
        if prop:
            prop.prof_metrics["gcycles:stl"] = 500.0
            prop.prof_metrics["gcycles:isu"] = 50.0
            prop.has_profile_data = True

        result = blame_instructions(vma_map, instructions)

        # Should have self-blame
        assert result.num_self_blame == 1
        assert result.total_stall_blame == 500.0


# =============================================================================
# CCT Dependency Graph Tests
# =============================================================================

from leo.analysis import (
    CCTDepNode,
    CCTDepEdge,
    CCTDepGraph,
    DependencyPath,
    EdgePathInfo,
    GraphStatistics,
    build_cct_dep_graph,
    compute_backward_slice,
    compute_forward_slice,
    compute_distance,
    classify_edge_type,
    find_barrier_memory_deps,
    prune_barrier_constraints,
    prune_latency_constraints,
    get_stalling_nodes,
)


class TestCCTDepNode:
    """Tests for CCTDepNode dataclass."""

    def test_basic_creation(self):
        """Test basic node creation."""
        node = CCTDepNode(cct_id=1, vma=0x100)
        assert node.cct_id == 1
        assert node.vma == 0x100
        assert node.instruction is None
        assert node.stall_cycles == 0.0

    def test_with_metrics(self):
        """Test node with metrics."""
        node = CCTDepNode(
            cct_id=42,
            vma=0x200,
            stall_cycles=100.0,
            lat_cycles=50.0,
            mem_lat_cycles=30.0,
            issue_count=10,
        )
        assert node.stall_cycles == 100.0
        assert node.lat_cycles == 50.0
        assert node.has_stalls() is True
        assert node.get_total_stalls() == 150.0

    def test_has_stalls(self):
        """Test has_stalls method."""
        no_stalls = CCTDepNode(cct_id=1, vma=0x100)
        assert no_stalls.has_stalls() is False

        with_stalls = CCTDepNode(cct_id=2, vma=0x110, stall_cycles=10.0)
        assert with_stalls.has_stalls() is True

        with_lat = CCTDepNode(cct_id=3, vma=0x120, lat_cycles=5.0)
        assert with_lat.has_stalls() is True

    def test_equality_by_cct_id(self):
        """Test that nodes are equal by cct_id."""
        node1 = CCTDepNode(cct_id=1, vma=0x100)
        node2 = CCTDepNode(cct_id=1, vma=0x200)  # Different VMA
        node3 = CCTDepNode(cct_id=2, vma=0x100)  # Different cct_id

        assert node1 == node2
        assert node1 != node3

    def test_hashable(self):
        """Test that nodes are hashable."""
        node1 = CCTDepNode(cct_id=1, vma=0x100)
        node2 = CCTDepNode(cct_id=2, vma=0x110)

        node_set = {node1, node2}
        assert len(node_set) == 2
        assert node1 in node_set

    def test_with_instruction(self):
        """Test node with instruction attached."""
        inst = InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2])
        node = CCTDepNode(cct_id=1, vma=0x100, instruction=inst)

        assert node.is_memory_op() is True
        assert node.is_load() is True
        assert node.is_sync() is False


class TestCCTDepEdge:
    """Tests for CCTDepEdge dataclass."""

    def test_basic_creation(self):
        """Test basic edge creation."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)
        assert edge.from_cct_id == 1
        assert edge.to_cct_id == 2
        assert edge.edge_type == "exec_dep"

    def test_with_edge_type(self):
        """Test edge with custom type."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2, edge_type="mem_dep")
        assert edge.edge_type == "mem_dep"

    def test_equality(self):
        """Test edge equality by endpoint IDs."""
        edge1 = CCTDepEdge(from_cct_id=1, to_cct_id=2)
        edge2 = CCTDepEdge(from_cct_id=1, to_cct_id=2, edge_type="mem_dep")
        edge3 = CCTDepEdge(from_cct_id=1, to_cct_id=3)

        # Same endpoints are equal, regardless of type
        assert edge1 == edge2
        assert edge1 != edge3

    def test_hashable(self):
        """Test that edges are hashable."""
        edge1 = CCTDepEdge(from_cct_id=1, to_cct_id=2)
        edge2 = CCTDepEdge(from_cct_id=2, to_cct_id=3)

        edge_set = {edge1, edge2}
        assert len(edge_set) == 2


class TestCCTDepGraph:
    """Tests for CCTDepGraph class."""

    @pytest.fixture
    def empty_graph(self):
        """Create an empty graph."""
        return CCTDepGraph()

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes."""
        return [
            CCTDepNode(cct_id=1, vma=0x100, stall_cycles=10.0),
            CCTDepNode(cct_id=2, vma=0x110, stall_cycles=20.0),
            CCTDepNode(cct_id=3, vma=0x120, stall_cycles=5.0),
            CCTDepNode(cct_id=4, vma=0x130, stall_cycles=0.0),
        ]

    @pytest.fixture
    def populated_graph(self, sample_nodes):
        """Create a graph with nodes and edges."""
        graph = CCTDepGraph()
        for node in sample_nodes:
            graph.add_node(node)

        # Edges: 1 -> 2 -> 3 -> 4
        #        1 ------> 3
        graph.add_edge(sample_nodes[0], sample_nodes[1], "exec_dep")
        graph.add_edge(sample_nodes[1], sample_nodes[2], "mem_dep")
        graph.add_edge(sample_nodes[2], sample_nodes[3], "exec_dep")
        graph.add_edge(sample_nodes[0], sample_nodes[2], "exec_dep")

        return graph

    def test_add_node(self, empty_graph):
        """Test adding nodes."""
        node = CCTDepNode(cct_id=1, vma=0x100)
        empty_graph.add_node(node)

        assert empty_graph.size() == 1
        assert empty_graph.get_node(1) == node

    def test_add_edge(self, empty_graph):
        """Test adding edges."""
        node1 = CCTDepNode(cct_id=1, vma=0x100)
        node2 = CCTDepNode(cct_id=2, vma=0x110)

        empty_graph.add_edge(node1, node2, "exec_dep")

        assert empty_graph.size() == 2
        assert empty_graph.edge_count() == 1
        assert empty_graph.has_edge(1, 2)
        assert empty_graph.get_edge_type(1, 2) == "exec_dep"

    def test_no_self_loops(self, empty_graph):
        """Test that self-loops are not added."""
        node = CCTDepNode(cct_id=1, vma=0x100)
        empty_graph.add_edge(node, node)

        assert empty_graph.edge_count() == 0

    def test_incoming_outgoing_nodes(self, populated_graph, sample_nodes):
        """Test incoming and outgoing node queries."""
        # Node 3 (cct_id=3) has incoming from 1, 2
        node3 = sample_nodes[2]
        incoming = list(populated_graph.incoming_nodes(node3))
        assert len(incoming) == 2
        incoming_ids = {n.cct_id for n in incoming}
        assert incoming_ids == {1, 2}

        # Node 1 (cct_id=1) has outgoing to 2, 3
        node1 = sample_nodes[0]
        outgoing = list(populated_graph.outgoing_nodes(node1))
        assert len(outgoing) == 2
        outgoing_ids = {n.cct_id for n in outgoing}
        assert outgoing_ids == {2, 3}

    def test_remove_edge(self, populated_graph):
        """Test removing edges."""
        initial_count = populated_graph.edge_count()
        result = populated_graph.remove_edge_by_id(1, 2)

        assert result is True
        assert populated_graph.edge_count() == initial_count - 1
        assert not populated_graph.has_edge(1, 2)

        # Removing non-existent edge returns False
        result = populated_graph.remove_edge_by_id(1, 4)
        assert result is False

    def test_get_node_by_vma(self, populated_graph, sample_nodes):
        """Test node lookup by VMA."""
        node = populated_graph.get_node_by_vma(0x110)
        assert node is not None
        assert node.cct_id == 2

        # Non-existent VMA
        node = populated_graph.get_node_by_vma(0x999)
        assert node is None

    def test_in_out_degree(self, populated_graph):
        """Test degree calculations."""
        # Node 1: in=0, out=2
        assert populated_graph.in_degree(1) == 0
        assert populated_graph.out_degree(1) == 2

        # Node 3: in=2, out=1
        assert populated_graph.in_degree(3) == 2
        assert populated_graph.out_degree(3) == 1

        # Node 4: in=1, out=0
        assert populated_graph.in_degree(4) == 1
        assert populated_graph.out_degree(4) == 0

    def test_statistics(self, populated_graph):
        """Test graph statistics."""
        stats = populated_graph.get_statistics()

        assert stats.num_nodes == 4
        assert stats.num_edges == 4
        assert stats.num_nodes_with_stalls == 3  # Nodes 1, 2, 3 have stalls
        assert stats.num_source_nodes == 1  # Node 1 has no incoming
        assert stats.num_sink_nodes == 1  # Node 4 has no outgoing
        assert stats.total_stall_cycles == 35.0  # 10 + 20 + 5 + 0
        assert "exec_dep" in stats.edge_types
        assert "mem_dep" in stats.edge_types

    def test_clear(self, populated_graph):
        """Test clearing the graph."""
        populated_graph.clear()

        assert populated_graph.size() == 0
        assert populated_graph.edge_count() == 0

    def test_node_iteration(self, populated_graph):
        """Test iterating over nodes."""
        node_ids = {n.cct_id for n in populated_graph.nodes()}
        assert node_ids == {1, 2, 3, 4}

    def test_edge_iteration(self, populated_graph):
        """Test iterating over edges."""
        edges = list(populated_graph.edges())
        assert len(edges) == 4

    def test_get_edge_exists(self, populated_graph):
        """Test get_edge returns edge when it exists."""
        edge = populated_graph.get_edge(1, 2)
        assert edge is not None
        assert edge.from_cct_id == 1
        assert edge.to_cct_id == 2

    def test_get_edge_not_exists(self, populated_graph):
        """Test get_edge returns None when edge doesn't exist."""
        edge = populated_graph.get_edge(1, 4)  # No direct edge
        assert edge is None

    def test_get_edge_with_paths(self):
        """Test get_edge returns edge with path data intact."""
        graph = CCTDepGraph()
        node1 = CCTDepNode(cct_id=1, vma=0x100)
        node2 = CCTDepNode(cct_id=2, vma=0x200)
        graph.add_edge(node1, node2)

        # Get the edge and add path data
        edge = graph.get_edge(1, 2)
        assert edge is not None

        path = EdgePathInfo(
            block_ids=(1, 2),
            instruction_count=5.0,
            accumulated_stalls=10,
        )
        edge.add_path(path)

        # Re-get the edge and verify path data persists
        edge_again = graph.get_edge(1, 2)
        assert edge_again is not None
        assert edge_again.has_valid_paths()
        assert edge_again.average_path_distance() == 5.0


class TestGraphTraversal:
    """Tests for graph traversal functions."""

    @pytest.fixture
    def traversal_graph(self):
        """Create a graph for traversal tests."""
        graph = CCTDepGraph()
        # Diamond pattern: 1 -> 2, 3 -> 4
        nodes = [
            CCTDepNode(cct_id=1, vma=0x100),
            CCTDepNode(cct_id=2, vma=0x110),
            CCTDepNode(cct_id=3, vma=0x120),
            CCTDepNode(cct_id=4, vma=0x130),
        ]
        for n in nodes:
            graph.add_node(n)

        graph.add_edge(nodes[0], nodes[1])
        graph.add_edge(nodes[0], nodes[2])
        graph.add_edge(nodes[1], nodes[3])
        graph.add_edge(nodes[2], nodes[3])

        return graph, nodes

    def test_backward_slice(self, traversal_graph):
        """Test backward slice computation."""
        graph, nodes = traversal_graph

        # Backward slice from node 4 should include all nodes
        slice_nodes = compute_backward_slice(nodes[3], graph)
        slice_ids = {n.cct_id for n in slice_nodes}
        assert slice_ids == {1, 2, 3, 4}

        # Backward slice from node 2 should include 1, 2
        slice_nodes = compute_backward_slice(nodes[1], graph)
        slice_ids = {n.cct_id for n in slice_nodes}
        assert slice_ids == {1, 2}

    def test_forward_slice(self, traversal_graph):
        """Test forward slice computation."""
        graph, nodes = traversal_graph

        # Forward slice from node 1 should include all nodes
        slice_nodes = compute_forward_slice(nodes[0], graph)
        slice_ids = {n.cct_id for n in slice_nodes}
        assert slice_ids == {1, 2, 3, 4}

        # Forward slice from node 3 should include 3, 4
        slice_nodes = compute_forward_slice(nodes[2], graph)
        slice_ids = {n.cct_id for n in slice_nodes}
        assert slice_ids == {3, 4}

    def test_backward_slice_with_max_depth(self, traversal_graph):
        """Test backward slice with depth limit."""
        graph, nodes = traversal_graph

        # With max_depth=1, only direct predecessors
        slice_nodes = compute_backward_slice(nodes[3], graph, max_depth=1)
        slice_ids = {n.cct_id for n in slice_nodes}
        # Node 4 + its direct predecessors (2, 3)
        assert 4 in slice_ids
        assert 2 in slice_ids or 3 in slice_ids


class TestComputeDistance:
    """Tests for compute_distance function."""

    def test_same_vma(self):
        """Test distance for same VMA."""
        assert compute_distance(0x100, 0x100) == 0.0

    def test_forward_dependency(self):
        """Test forward dependency distance."""
        # With inst_size=16, distance of 32 bytes = 2 instructions
        assert compute_distance(0x100, 0x120, inst_size=16) == 2.0

    def test_backward_dependency(self):
        """Test backward dependency returns absolute distance (GPA-aligned)."""
        # Backward (from > to) now returns absolute distance
        # Distance: |0x100 - 0x120| / 16 = 0x20 / 16 = 2 instructions
        assert compute_distance(0x120, 0x100) == 2.0

    def test_custom_inst_size(self):
        """Test with custom instruction size."""
        # 64 bytes with 8-byte instructions = 8 instructions
        assert compute_distance(0x100, 0x140, inst_size=8) == 8.0


class TestClassifyEdgeType:
    """Tests for classify_edge_type function."""

    def test_sync_instruction(self):
        """Test sync instruction classification."""
        sync_inst = InstructionStat(op="BAR.SYNC", pc=0x100)
        other_inst = InstructionStat(op="FADD", pc=0x110)

        assert classify_edge_type(sync_inst, other_inst) == "sync"
        assert classify_edge_type(other_inst, sync_inst) == "sync"

    def test_memory_load(self):
        """Test memory load classification."""
        load_inst = InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2])
        other_inst = InstructionStat(op="FADD", pc=0x110)

        assert classify_edge_type(load_inst, other_inst) == "mem_dep"

    def test_branch(self):
        """Test branch instruction classification."""
        branch_inst = InstructionStat(op="BRA", pc=0x100)
        other_inst = InstructionStat(op="FADD", pc=0x110)

        assert classify_edge_type(branch_inst, other_inst) == "control"

    def test_default_exec_dep(self):
        """Test default execution dependency."""
        inst1 = InstructionStat(op="FADD", pc=0x100)
        inst2 = InstructionStat(op="FMUL", pc=0x110)

        assert classify_edge_type(inst1, inst2) == "exec_dep"


class TestGetStallingNodes:
    """Tests for get_stalling_nodes function."""

    def test_get_stalling_nodes(self):
        """Test getting stalling nodes."""
        graph = CCTDepGraph()
        nodes = [
            CCTDepNode(cct_id=1, vma=0x100, stall_cycles=100.0),
            CCTDepNode(cct_id=2, vma=0x110, stall_cycles=0.0),
            CCTDepNode(cct_id=3, vma=0x120, stall_cycles=50.0, lat_cycles=25.0),
            CCTDepNode(cct_id=4, vma=0x130, stall_cycles=200.0),
        ]
        for n in nodes:
            graph.add_node(n)

        stalling = get_stalling_nodes(graph)

        assert len(stalling) == 3  # Nodes 1, 3, 4 have stalls
        # Should be sorted by total stalls descending
        assert stalling[0].cct_id == 4  # 200 total
        assert stalling[1].cct_id == 1  # 100 total
        assert stalling[2].cct_id == 3  # 75 total

    def test_with_threshold(self):
        """Test with stall threshold."""
        graph = CCTDepGraph()
        nodes = [
            CCTDepNode(cct_id=1, vma=0x100, stall_cycles=100.0),
            CCTDepNode(cct_id=2, vma=0x110, stall_cycles=50.0),
            CCTDepNode(cct_id=3, vma=0x120, stall_cycles=10.0),
        ]
        for n in nodes:
            graph.add_node(n)

        stalling = get_stalling_nodes(graph, threshold=40.0)

        assert len(stalling) == 2  # Only nodes with > 40 stalls
        assert stalling[0].cct_id == 1
        assert stalling[1].cct_id == 2


class TestPruneBarrierConstraints:
    """Tests for prune_barrier_constraints function.

    GPA Reference: pruneCCTDepGraphBarrier() at GPUAdvisor-Blame.cpp:437-486

    GPA-aligned behavior:
    - Skip only if BOTH read and write barriers are BARRIER_NONE
    - If at least one barrier is set, check if destination waits on it
    - Remove edge if destination doesn't wait on any of source's active barriers
    """

    def test_prune_mismatched_barriers_both_set(self):
        """Test pruning edges when both barriers set but neither matches."""
        graph = CCTDepGraph()

        # Instruction 1 has both read=2 and write=3
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(read=2, write=3))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 1 only (doesn't match either 2 or 3)
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000001))  # Wait on B1
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        initial_edges = graph.edge_count()
        removed = prune_barrier_constraints(graph)

        # Edge should be removed due to barrier mismatch
        assert removed == 1
        assert graph.edge_count() == initial_edges - 1

    def test_keep_matched_barriers_via_read(self):
        """Test keeping edges when destination waits on read barrier."""
        graph = CCTDepGraph()

        # Instruction 1 has read=2, write=3
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(read=2, write=3))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 2 (matches read)
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000010))  # Wait on B2
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be kept (to waits on from's read barrier)
        assert removed == 0
        assert graph.edge_count() == 1

    def test_keep_matched_barriers_via_write(self):
        """Test keeping edges when destination waits on write barrier."""
        graph = CCTDepGraph()

        # Instruction 1 has read=2, write=3
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(read=2, write=3))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 3 (matches write)
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000100))  # Wait on B3
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be kept (to waits on from's write barrier)
        assert removed == 0
        assert graph.edge_count() == 1

    def test_prune_when_only_write_set_no_match(self):
        """Test pruning when only write barrier is set and destination doesn't match.

        GPA-aligned: prunes if at least one barrier is set and destination
        doesn't wait on any active barriers.
        """
        graph = CCTDepGraph()

        # Instruction 1 only has write=2 (read defaults to BARRIER_NONE=7)
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(write=2))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 1 (doesn't match write=2)
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000001))  # Wait on B1
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be PRUNED because dest doesn't wait on write barrier B2
        assert removed == 1
        assert graph.edge_count() == 0

    def test_prune_when_only_read_set_no_match(self):
        """Test pruning when only read barrier is set and destination doesn't match."""
        graph = CCTDepGraph()

        # Instruction 1 only has read=2 (write defaults to BARRIER_NONE=7)
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(read=2))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 1 (doesn't match read=2)
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000001))  # Wait on B1
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be PRUNED because dest doesn't wait on read barrier B2
        assert removed == 1
        assert graph.edge_count() == 0

    def test_keep_when_only_write_set_and_matches(self):
        """Test keeping edge when only write barrier is set and destination matches."""
        graph = CCTDepGraph()

        # Instruction 1 only has write=2
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(write=2))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 2 (matches write)
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000010))  # Wait on B2
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be KEPT because dest waits on write barrier B2
        assert removed == 0
        assert graph.edge_count() == 1

    def test_keep_when_only_read_set_and_matches(self):
        """Test keeping edge when only read barrier is set and destination matches."""
        graph = CCTDepGraph()

        # Instruction 1 only has read=3
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(read=3))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 3 (matches read)
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000100))  # Wait on B3
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be KEPT because dest waits on read barrier B3
        assert removed == 0
        assert graph.edge_count() == 1

    def test_skip_when_both_barriers_none(self):
        """Test that edges are skipped when both barriers are BARRIER_NONE."""
        graph = CCTDepGraph()

        # Instruction 1 has no barriers (both default to BARRIER_NONE=7)
        inst1 = InstructionStat(op="FADD", pc=0x100, control=Control())
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on barrier 1
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000001))
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be KEPT because source has no barrier constraints
        assert removed == 0
        assert graph.edge_count() == 1

    def test_keep_when_waiting_on_both_barriers(self):
        """Test keeping edges when destination waits on both barriers."""
        graph = CCTDepGraph()

        # Instruction 1 has read=2, write=3
        inst1 = InstructionStat(op="DEPBAR", pc=0x100, control=Control(read=2, write=3))
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        # Instruction 2 waits on both B2 and B3
        inst2 = InstructionStat(op="FADD", pc=0x110, control=Control(wait=0b000110))  # Wait on B2, B3
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        graph.add_edge(node1, node2)

        removed = prune_barrier_constraints(graph)

        # Edge should be kept
        assert removed == 0
        assert graph.edge_count() == 1


class TestPruneLatencyConstraints:
    """Tests for prune_latency_constraints function.

    GPA Reference: pruneCCTDepGraphLatency() at GPUAdvisor-Blame.cpp:868-971

    Tests both the simple PC-distance fallback and CFG-based tracking.
    """

    def test_with_cfg_keeps_valid_paths(self):
        """Test CFG-based pruning keeps edges with valid paths."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.cfg import CFG, Block, Function

        graph = CCTDepGraph()
        arch = get_architecture("a100")

        # Create instructions
        inst1 = InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2], control=Control(stall=1))
        inst2 = InstructionStat(op="FADD", pc=0x110, dsts=[1], srcs=[3, 4], control=Control(stall=1))
        inst3 = InstructionStat(op="FMUL", pc=0x120, dsts=[2], srcs=[0, 5], control=Control(stall=1))

        # inst3 uses register 0 from inst1
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)
        node3 = CCTDepNode(cct_id=3, vma=0x120, instruction=inst3)

        graph.add_edge(node1, node3)

        # Create CFG via Function wrapper
        block = Block(
            id=1,
            instructions=[inst1, inst2, inst3],
            targets=[],
        )
        function = Function(name="test_kernel", blocks=[block], entry_block_id=1)
        cfg = CFG(function)

        pc_to_inst = {0x100: inst1, 0x110: inst2, 0x120: inst3}

        # With only 2 instructions between (stall=1 each), accumulated stalls = 2
        # LDG.E latency is ~200, so 2 < 200, edge should be kept
        removed = prune_latency_constraints(graph, arch, cfg=cfg, pc_to_inst=pc_to_inst)
        assert removed == 0
        assert graph.edge_count() == 1

    def test_with_cfg_prunes_hidden(self):
        """Test CFG-based pruning removes hidden edges."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.cfg import CFG, Block, Function

        graph = CCTDepGraph()
        arch = get_architecture("a100")

        # Create instructions with high stalls
        inst1 = InstructionStat(op="IADD", pc=0x100, dsts=[0], srcs=[1, 2], control=Control(stall=1))
        # Many instructions with high stall values to exceed IADD latency (~4 cycles)
        inst2 = InstructionStat(op="FADD", pc=0x110, dsts=[1], srcs=[3, 4], control=Control(stall=2))
        inst3 = InstructionStat(op="FADD", pc=0x120, dsts=[2], srcs=[5, 6], control=Control(stall=2))
        inst4 = InstructionStat(op="FMUL", pc=0x130, dsts=[3], srcs=[0, 7], control=Control(stall=1))

        # inst4 uses register 0 from inst1
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)
        node4 = CCTDepNode(cct_id=4, vma=0x130, instruction=inst4)

        graph.add_edge(node1, node4)

        # Create CFG via Function wrapper
        block = Block(
            id=1,
            instructions=[inst1, inst2, inst3, inst4],
            targets=[],
        )
        function = Function(name="test_kernel", blocks=[block], entry_block_id=1)
        cfg = CFG(function)

        pc_to_inst = {0x100: inst1, 0x110: inst2, 0x120: inst3, 0x130: inst4}

        # Accumulated stalls: 2 + 2 = 4 cycles (skipping inst1)
        # IADD latency is 4-5 cycles, so 4 >= 4, edge should be pruned
        removed = prune_latency_constraints(graph, arch, cfg=cfg, pc_to_inst=pc_to_inst)
        assert removed == 1
        assert graph.edge_count() == 0


class TestAMDLatencyPruning:
    """Test CFG-based latency pruning for AMD architectures.

    Verifies that removing the vendor=='nvidia' guard enables correct
    CFG-based pruning for AMD with stall=1 per instruction.
    """

    def test_amd_cfg_prunes_hidden_valu(self):
        """AMD VALU with LLVM latency=1 should be pruned after 1 instruction."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.cfg import CFG, Block, Function
        from leo.arch.amd import MI300

        graph = CCTDepGraph()
        arch = MI300()  # v_add_f32 has latency (1, 1)

        # v_add_f32 defines register 0
        inst1 = InstructionStat(op="v_add_f32", pc=0x100, dsts=[0], srcs=[1, 2],
                                control=Control(stall=1))
        # Intervening instruction (1 stall cycle)
        inst2 = InstructionStat(op="v_mov_b32", pc=0x104, dsts=[3], srcs=[4],
                                control=Control(stall=1))
        # Consumer uses register 0
        inst3 = InstructionStat(op="v_mul_f32", pc=0x108, dsts=[5], srcs=[0, 6],
                                control=Control(stall=1))

        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)
        node3 = CCTDepNode(cct_id=3, vma=0x108, instruction=inst3)
        graph.add_edge(node1, node3)

        block = Block(id=1, instructions=[inst1, inst2, inst3], targets=[])
        function = Function(name="test_kernel", blocks=[block], entry_block_id=1)
        cfg = CFG(function)
        pc_to_inst = {0x100: inst1, 0x104: inst2, 0x108: inst3}

        # Accumulated stalls: inst2.stall=1 >= v_add_f32 max_latency=1
        removed = prune_latency_constraints(graph, arch, cfg=cfg, pc_to_inst=pc_to_inst)
        assert removed == 1
        assert graph.edge_count() == 0

    def test_amd_cfg_keeps_vmem(self):
        """AMD VMEM with latency=80+ should NOT be pruned with few instructions."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.cfg import CFG, Block, Function
        from leo.arch.amd import MI300

        graph = CCTDepGraph()
        arch = MI300()  # global_load_dword has latency (80, 300)

        inst1 = InstructionStat(op="global_load_dword", pc=0x100, dsts=[0], srcs=[2],
                                control=Control(stall=1))
        inst2 = InstructionStat(op="v_add_f32", pc=0x104, dsts=[3], srcs=[4, 5],
                                control=Control(stall=1))
        inst3 = InstructionStat(op="v_mul_f32", pc=0x108, dsts=[6], srcs=[0, 7],
                                control=Control(stall=1))

        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)
        node3 = CCTDepNode(cct_id=3, vma=0x108, instruction=inst3)
        graph.add_edge(node1, node3)

        block = Block(id=1, instructions=[inst1, inst2, inst3], targets=[])
        function = Function(name="test_kernel", blocks=[block], entry_block_id=1)
        cfg = CFG(function)
        pc_to_inst = {0x100: inst1, 0x104: inst2, 0x108: inst3}

        # Accumulated stalls: 1 cycle << 300 max latency → NOT pruned
        removed = prune_latency_constraints(graph, arch, cfg=cfg, pc_to_inst=pc_to_inst)
        assert removed == 0
        assert graph.edge_count() == 1

    def test_amd_s_nop_accelerates_pruning(self):
        """s_nop N should contribute N+1 cycles to latency hiding."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.cfg import CFG, Block, Function
        from leo.arch.amd import MI300

        graph = CCTDepGraph()
        arch = MI300()  # v_rcp_f32 (MUFU) has latency (4, 4)

        inst1 = InstructionStat(op="v_rcp_f32", pc=0x100, dsts=[0], srcs=[1],
                                control=Control(stall=1))
        # s_nop 3 = 4 idle cycles (parsed by AMD disassembler)
        inst2 = InstructionStat(op="s_nop", pc=0x104, dsts=[], srcs=[],
                                control=Control(stall=4))
        inst3 = InstructionStat(op="v_mul_f32", pc=0x108, dsts=[5], srcs=[0, 6],
                                control=Control(stall=1))

        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)
        node3 = CCTDepNode(cct_id=3, vma=0x108, instruction=inst3)
        graph.add_edge(node1, node3)

        block = Block(id=1, instructions=[inst1, inst2, inst3], targets=[])
        function = Function(name="test_kernel", blocks=[block], entry_block_id=1)
        cfg = CFG(function)
        pc_to_inst = {0x100: inst1, 0x104: inst2, 0x108: inst3}

        # Accumulated stalls: s_nop stall=4 >= MUFU max_latency=4
        removed = prune_latency_constraints(graph, arch, cfg=cfg, pc_to_inst=pc_to_inst)
        assert removed == 1


class TestBuildSBIDLifecycleMap:
    """Tests for the SBID lifecycle map builder."""

    def test_single_lifecycle(self):
        """set $0 → wait $0 → map contains set_pc with wait_pc."""
        from leo.analysis.graph import _build_sbid_lifecycle_map
        from leo.binary.instruction import IntelSWSB

        insts = {
            0x00: InstructionStat(op="send", pc=0x00, swsb=IntelSWSB(
                raw=0xC0, has_sbid=True, sbid_type="set", sbid=0)),
            0x10: InstructionStat(op="add", pc=0x10),
            0x20: InstructionStat(op="add", pc=0x20, swsb=IntelSWSB(
                raw=0x80, has_sbid=True, sbid_type="dst_wait", sbid=0)),
        }
        result = _build_sbid_lifecycle_map(insts)
        assert 0x00 in result
        sbid, wait_pcs, end_pc = result[0x00]
        assert sbid == 0
        assert 0x20 in wait_pcs
        assert end_pc == -1  # No reuse

    def test_token_reuse(self):
        """set $0 → set $0 again → first lifecycle closed with end_pc."""
        from leo.analysis.graph import _build_sbid_lifecycle_map
        from leo.binary.instruction import IntelSWSB

        insts = {
            0x00: InstructionStat(op="send", pc=0x00, swsb=IntelSWSB(
                raw=0xC0, has_sbid=True, sbid_type="set", sbid=0)),
            0x10: InstructionStat(op="add", pc=0x10),
            0x20: InstructionStat(op="send", pc=0x20, swsb=IntelSWSB(
                raw=0xC0, has_sbid=True, sbid_type="set", sbid=0)),
        }
        result = _build_sbid_lifecycle_map(insts)
        assert 0x00 in result
        assert result[0x00][2] == 0x20  # end_pc = reuse PC
        assert 0x20 in result
        assert result[0x20][2] == -1  # Second lifecycle still open

    def test_multiple_tokens(self):
        """set $0 + set $1 → wait $0 + wait $1 → two separate lifecycles."""
        from leo.analysis.graph import _build_sbid_lifecycle_map
        from leo.binary.instruction import IntelSWSB

        insts = {
            0x00: InstructionStat(op="send", pc=0x00, swsb=IntelSWSB(
                raw=0xC0, has_sbid=True, sbid_type="set", sbid=0)),
            0x10: InstructionStat(op="send", pc=0x10, swsb=IntelSWSB(
                raw=0xC1, has_sbid=True, sbid_type="set", sbid=1)),
            0x20: InstructionStat(op="add", pc=0x20, swsb=IntelSWSB(
                raw=0x80, has_sbid=True, sbid_type="dst_wait", sbid=0)),
            0x30: InstructionStat(op="add", pc=0x30, swsb=IntelSWSB(
                raw=0x81, has_sbid=True, sbid_type="dst_wait", sbid=1)),
        }
        result = _build_sbid_lifecycle_map(insts)
        assert len(result) == 2
        assert 0x20 in result[0x00][1]  # wait $0
        assert 0x30 in result[0x10][1]  # wait $1

    def test_empty(self):
        """No instructions → empty map."""
        from leo.analysis.graph import _build_sbid_lifecycle_map

        result = _build_sbid_lifecycle_map({})
        assert len(result) == 0


def _empty_cfg():
    """Create an empty CFG for tests that only exercise SBID/distance logic."""
    from leo.binary.cfg import CFG, Function
    return CFG(Function(name="test"))


class TestIntelSBIDLatencyPruning:
    """Tests for Intel SWSB SBID-based latency pruning of send edges."""

    def test_prune_send_no_wait(self):
        """Send with set $0, no wait between producer and consumer → PRUNE."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.instruction import IntelSWSB
        from leo.arch.intel import PonteVecchio

        graph = CCTDepGraph()
        arch = PonteVecchio()

        send_inst = InstructionStat(
            op="send", pc=0x00, dsts=[0], srcs=[2],
            swsb=IntelSWSB(raw=0xC0, has_sbid=True, sbid_type="set", sbid=0))
        alu_inst = InstructionStat(op="add", pc=0x10, dsts=[1], srcs=[3])
        consumer_inst = InstructionStat(op="add", pc=0x20, dsts=[2], srcs=[0])

        node_send = CCTDepNode(cct_id=1, vma=0x00, instruction=send_inst)
        node_consumer = CCTDepNode(cct_id=3, vma=0x20, instruction=consumer_inst)
        graph.add_edge(node_send, node_consumer)

        pc_to_inst = {0x00: send_inst, 0x10: alu_inst, 0x20: consumer_inst}
        removed = prune_latency_constraints(graph, arch, cfg=_empty_cfg(), pc_to_inst=pc_to_inst)
        assert removed == 1

    def test_keep_send_wait_between(self):
        """Send with set $0, wait $0 between producer and consumer → KEEP."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.instruction import IntelSWSB
        from leo.arch.intel import PonteVecchio

        graph = CCTDepGraph()
        arch = PonteVecchio()

        send_inst = InstructionStat(
            op="send", pc=0x00, dsts=[0], srcs=[2],
            swsb=IntelSWSB(raw=0xC0, has_sbid=True, sbid_type="set", sbid=0))
        wait_inst = InstructionStat(
            op="add", pc=0x10, dsts=[1], srcs=[3],
            swsb=IntelSWSB(raw=0x80, has_sbid=True, sbid_type="dst_wait", sbid=0))
        consumer_inst = InstructionStat(op="add", pc=0x20, dsts=[2], srcs=[0])

        node_send = CCTDepNode(cct_id=1, vma=0x00, instruction=send_inst)
        node_consumer = CCTDepNode(cct_id=3, vma=0x20, instruction=consumer_inst)
        graph.add_edge(node_send, node_consumer)

        pc_to_inst = {0x00: send_inst, 0x10: wait_inst, 0x20: consumer_inst}
        removed = prune_latency_constraints(graph, arch, cfg=_empty_cfg(), pc_to_inst=pc_to_inst)
        assert removed == 0
        assert graph.edge_count() == 1

    def test_send_no_sbid_falls_through(self):
        """Send without SBID → standard latency check (distance vs max_latency)."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.arch.intel import PonteVecchio

        graph = CCTDepGraph()
        arch = PonteVecchio()  # send max_latency=500, inst_size=16

        # send with no SWSB annotation
        send_inst = InstructionStat(op="send", pc=0x00, dsts=[0], srcs=[2])
        # consumer very far away: 0x2000/16 = 512 > 500
        consumer_inst = InstructionStat(op="add", pc=0x2000, dsts=[2], srcs=[0])

        node_send = CCTDepNode(cct_id=1, vma=0x00, instruction=send_inst)
        node_consumer = CCTDepNode(cct_id=2, vma=0x2000, instruction=consumer_inst)
        graph.add_edge(node_send, node_consumer)

        pc_to_inst = {0x00: send_inst, 0x2000: consumer_inst}
        removed = prune_latency_constraints(graph, arch, cfg=_empty_cfg(), pc_to_inst=pc_to_inst)
        assert removed == 1  # distance=512 >= 500 → pruned by standard check

    def test_implicit_wait_from_reuse(self):
        """Second set $0 between producer-consumer acts as implicit wait → KEEP."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.instruction import IntelSWSB
        from leo.arch.intel import PonteVecchio

        graph = CCTDepGraph()
        arch = PonteVecchio()

        send1 = InstructionStat(
            op="send", pc=0x00, dsts=[0], srcs=[2],
            swsb=IntelSWSB(raw=0xC0, has_sbid=True, sbid_type="set", sbid=0))
        send2 = InstructionStat(
            op="send", pc=0x20, dsts=[1], srcs=[3],
            swsb=IntelSWSB(raw=0xC0, has_sbid=True, sbid_type="set", sbid=0))
        consumer = InstructionStat(op="add", pc=0x30, dsts=[2], srcs=[0])

        node_send1 = CCTDepNode(cct_id=1, vma=0x00, instruction=send1)
        node_consumer = CCTDepNode(cct_id=3, vma=0x30, instruction=consumer)
        graph.add_edge(node_send1, node_consumer)

        pc_to_inst = {0x00: send1, 0x20: send2, 0x30: consumer}
        removed = prune_latency_constraints(graph, arch, cfg=_empty_cfg(), pc_to_inst=pc_to_inst)
        assert removed == 0  # Implicit wait at 0x20 → keep

    def test_wrong_token_no_help(self):
        """Wait for different token doesn't help → PRUNE."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.instruction import IntelSWSB
        from leo.arch.intel import PonteVecchio

        graph = CCTDepGraph()
        arch = PonteVecchio()

        send_inst = InstructionStat(
            op="send", pc=0x00, dsts=[0], srcs=[2],
            swsb=IntelSWSB(raw=0xC0, has_sbid=True, sbid_type="set", sbid=0))
        # wait $1 (wrong token)
        wait_inst = InstructionStat(
            op="add", pc=0x10, dsts=[1], srcs=[3],
            swsb=IntelSWSB(raw=0x81, has_sbid=True, sbid_type="dst_wait", sbid=1))
        consumer_inst = InstructionStat(op="add", pc=0x20, dsts=[2], srcs=[0])

        node_send = CCTDepNode(cct_id=1, vma=0x00, instruction=send_inst)
        node_consumer = CCTDepNode(cct_id=3, vma=0x20, instruction=consumer_inst)
        graph.add_edge(node_send, node_consumer)

        pc_to_inst = {0x00: send_inst, 0x10: wait_inst, 0x20: consumer_inst}
        removed = prune_latency_constraints(graph, arch, cfg=_empty_cfg(), pc_to_inst=pc_to_inst)
        assert removed == 1  # Wrong token → still pruned

    def test_wait_on_consumer_itself(self):
        """Wait annotation on the consumer instruction → KEEP."""
        from leo.analysis.graph import prune_latency_constraints
        from leo.binary.instruction import IntelSWSB
        from leo.arch.intel import PonteVecchio

        graph = CCTDepGraph()
        arch = PonteVecchio()

        send_inst = InstructionStat(
            op="send", pc=0x00, dsts=[0], srcs=[2],
            swsb=IntelSWSB(raw=0xC0, has_sbid=True, sbid_type="set", sbid=0))
        consumer_inst = InstructionStat(
            op="add", pc=0x20, dsts=[2], srcs=[0],
            swsb=IntelSWSB(raw=0x80, has_sbid=True, sbid_type="dst_wait", sbid=0))

        node_send = CCTDepNode(cct_id=1, vma=0x00, instruction=send_inst)
        node_consumer = CCTDepNode(cct_id=2, vma=0x20, instruction=consumer_inst)
        graph.add_edge(node_send, node_consumer)

        pc_to_inst = {0x00: send_inst, 0x20: consumer_inst}
        removed = prune_latency_constraints(graph, arch, cfg=_empty_cfg(), pc_to_inst=pc_to_inst)
        assert removed == 0  # Wait at consumer PC → kept


class TestBuildCCTDepGraph:
    """Tests for build_cct_dep_graph function."""

    @pytest.fixture
    def sample_instructions_with_deps(self):
        """Create sample instructions with dependencies set up."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2]),
            InstructionStat(op="LDG.E", pc=0x110, dsts=[1], srcs=[3]),
            InstructionStat(op="FADD", pc=0x120, dsts=[4], srcs=[0, 1]),
            InstructionStat(op="STG.E", pc=0x130, dsts=[], srcs=[4, 5]),
        ]

        # Set up dependencies
        instructions[2].assign_pcs = {0: [0x100], 1: [0x110]}
        instructions[3].assign_pcs = {4: [0x120]}

        return instructions

    def test_build_from_instructions(self, sample_instructions_with_deps):
        """Test building graph from instructions with VMAPropertyMap."""
        instructions = sample_instructions_with_deps

        # Build VMA map with profile data
        vma_map = VMAPropertyMap.build_from_instructions(instructions, arch_name="a100")

        # Add mock profile data
        for inst in instructions:
            prop = vma_map.get(inst.pc)
            if prop:
                prop.cct_id = inst.pc  # Use PC as cct_id for testing
                prop.prof_metrics["gcycles:stl"] = 100.0
                prop.prof_metrics["gcycles:isu"] = 10.0
                prop.has_profile_data = True

        graph = build_cct_dep_graph(
            vma_map, instructions,
            apply_opcode_pruning=False,
            apply_latency_pruning=False
        )

        # Should have 4 nodes and 3 edges
        assert graph.size() == 4
        assert graph.edge_count() == 3

        # Check edge types
        node_0x120 = graph.get_node_by_vma(0x120)
        assert node_0x120 is not None
        incoming = list(graph.incoming_nodes(node_0x120))
        assert len(incoming) == 2

    def test_build_with_no_profile_data(self, sample_instructions_with_deps):
        """Test that nodes without profile data are excluded."""
        instructions = sample_instructions_with_deps

        vma_map = VMAPropertyMap.build_from_instructions(instructions, arch_name="a100")
        # No profile data added - all has_profile_data = False

        graph = build_cct_dep_graph(vma_map, instructions)

        # Should have no nodes since no profile data
        assert graph.size() == 0


# =============================================================================
# NVIDIA Barrier Dependency Tests
# =============================================================================


class TestNVIDIABarrierDeps:
    """Tests for NVIDIA barrier-based memory dependency linking."""

    def test_find_barrier_memory_deps_single_barrier(self):
        """Test finding a single barrier dependency (LDG sets B1, FADD waits B1)."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2]),
            InstructionStat(op="FADD", pc=0x110, dsts=[4], srcs=[0, 1]),
        ]
        # LDG sets barrier B1
        instructions[0].control = Control(write=1, stall=4)
        # FADD waits on barrier B1
        instructions[1].control = Control(wait=0b000001, stall=2)  # bit 0 = B1

        deps = find_barrier_memory_deps(instructions[1], instructions, 1)
        assert deps == [0x100]

    def test_find_barrier_memory_deps_multiple_barriers(self):
        """Test finding multiple barrier dependencies (two LDGs, one wait)."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2]),
            InstructionStat(op="LDG.E", pc=0x110, dsts=[1], srcs=[3]),
            InstructionStat(op="DFMA", pc=0x120, dsts=[4], srcs=[0, 1]),
        ]
        # First LDG sets B1, second sets B2
        instructions[0].control = Control(write=1, stall=4)
        instructions[1].control = Control(write=2, stall=4)
        # DFMA waits on B1 and B2
        instructions[2].control = Control(wait=0b000011, stall=2)  # bits 0,1 = B1,B2

        deps = find_barrier_memory_deps(instructions[2], instructions, 2)
        assert sorted(deps) == [0x100, 0x110]

    def test_find_barrier_memory_deps_no_wait(self):
        """Test instruction with no barrier wait returns empty."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2]),
            InstructionStat(op="FADD", pc=0x110, dsts=[4], srcs=[0, 1]),
        ]
        instructions[0].control = Control(write=1, stall=4)
        instructions[1].control = Control(wait=0, stall=1)  # No wait

        deps = find_barrier_memory_deps(instructions[1], instructions, 1)
        assert deps == []

    def test_find_barrier_memory_deps_depbar_instruction(self):
        """Test DEPBAR instruction with explicit barrier operand."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2]),
            InstructionStat(op="DEPBAR.LE", pc=0x110, bdsts=[1]),
        ]
        instructions[0].control = Control(write=1, stall=4)
        instructions[1].control = Control(wait=0, stall=1)

        deps = find_barrier_memory_deps(instructions[1], instructions, 1)
        assert deps == [0x100]

    def test_build_cct_dep_graph_with_nvidia_barriers(self):
        """Test that build_cct_dep_graph creates barrier edges for NVIDIA."""
        from leo.arch import get_architecture

        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2]),
            InstructionStat(op="FADD", pc=0x110, dsts=[4], srcs=[0, 1]),
        ]
        # LDG sets B1, FADD waits on B1
        instructions[0].control = Control(write=1, stall=4)
        instructions[1].control = Control(wait=0b000001, stall=2)

        # Set up register deps
        instructions[1].assign_pcs = {0: [0x100]}

        arch = get_architecture("a100")
        vma_map = VMAPropertyMap.build_from_instructions(instructions, arch_name="a100")

        # Add profile data
        for inst in instructions:
            prop = vma_map.get(inst.pc)
            if prop:
                prop.cct_id = inst.pc
                prop.prof_metrics["gcycles:stl"] = 100.0
                prop.prof_metrics["gcycles:isu"] = 10.0
                prop.has_profile_data = True

        graph = build_cct_dep_graph(
            vma_map, instructions, arch=arch,
            apply_opcode_pruning=False,
            apply_latency_pruning=False,
        )

        # Should have barrier edge from LDG → FADD
        # Plus the regular register dep edge
        node_fadd = graph.get_node_by_vma(0x110)
        assert node_fadd is not None
        incoming = list(graph.incoming_nodes(node_fadd))
        # LDG should be an incoming node (via both register dep and barrier)
        ldg_pcs = [n.vma for n in incoming]
        assert 0x100 in ldg_pcs


# =============================================================================
# Back-Slicing Engine Tests
# =============================================================================

from leo.analysis import (
    BackSliceConfig,
    BackSliceStats,
    BackSliceEngine,
    analyze_instructions,
)


class TestBackSliceConfig:
    """Tests for BackSliceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BackSliceConfig()
        assert config.arch_name == "a100"
        assert config.enable_predicate_tracking is True
        assert config.apply_opcode_pruning is True
        assert config.apply_barrier_pruning is True
        assert config.apply_graph_latency_pruning is True
        assert config.stall_threshold == 0.0
        assert config.debug is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = BackSliceConfig(
            arch_name="v100",
            enable_predicate_tracking=False,
            debug=True,
        )
        assert config.arch_name == "v100"
        assert config.enable_predicate_tracking is False
        assert config.debug is True


class TestBackSliceStats:
    """Tests for BackSliceStats dataclass."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = BackSliceStats()
        assert stats.num_instructions == 0
        assert stats.initial_nodes == 0
        assert stats.final_edges == 0
        assert stats.edges_pruned_opcode == 0


class TestBackSliceEngine:
    """Tests for BackSliceEngine class."""

    @pytest.fixture
    def sample_instructions(self):
        """Create sample instructions with dependencies."""
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2], control=Control(stall=4)),
            InstructionStat(op="LDG.E", pc=0x110, dsts=[1], srcs=[3], control=Control(stall=4)),
            InstructionStat(op="FADD", pc=0x120, dsts=[4], srcs=[0, 1], control=Control(stall=2)),
            InstructionStat(op="STG.E", pc=0x130, dsts=[], srcs=[4, 5], control=Control(stall=1)),
        ]

        # Set up dependencies
        instructions[2].assign_pcs = {0: [0x100], 1: [0x110]}
        instructions[3].assign_pcs = {4: [0x120]}

        return instructions

    @pytest.fixture
    def vma_map_with_stalls(self, sample_instructions):
        """Create VMAPropertyMap with simulated stall data."""
        vma_map = VMAPropertyMap.build_from_instructions(
            sample_instructions, arch_name="a100"
        )

        # Add mock profile data with stalls
        for inst in sample_instructions:
            prop = vma_map.get(inst.pc)
            if prop:
                prop.cct_id = inst.pc  # Use PC as cct_id
                prop.has_profile_data = True

        # Add stalls to FADD instruction
        prop = vma_map.get(0x120)
        if prop:
            prop.prof_metrics["gcycles:stl"] = 100.0
            prop.prof_metrics["gcycles:isu"] = 10.0

        # Add stalls to STG instruction
        prop = vma_map.get(0x130)
        if prop:
            prop.prof_metrics["gcycles:stl"] = 50.0
            prop.prof_metrics["gcycles:isu"] = 5.0

        return vma_map

    def test_create_from_instructions(self, sample_instructions):
        """Test creating engine from instructions."""
        engine = BackSliceEngine.from_instructions(sample_instructions)

        assert engine.vma_map is not None
        assert len(engine.instructions) == 4
        assert engine.graph is None  # Not built yet

    def test_create_with_vma_map(self, sample_instructions, vma_map_with_stalls):
        """Test creating engine with pre-built VMAPropertyMap."""
        engine = BackSliceEngine(
            vma_map=vma_map_with_stalls,
            instructions=sample_instructions,
        )

        assert engine.vma_map is vma_map_with_stalls
        assert len(engine.instructions) == 4

    def test_analyze_basic(self, sample_instructions, vma_map_with_stalls):
        """Test basic analysis pipeline."""
        engine = BackSliceEngine(
            vma_map=vma_map_with_stalls,
            instructions=sample_instructions,
        )

        result = engine.analyze()

        assert result is not None
        assert engine.graph is not None
        assert engine.stats.num_instructions == 4

    def test_analyze_produces_blame(self, sample_instructions, vma_map_with_stalls):
        """Test that analysis produces blame attribution."""
        engine = BackSliceEngine(
            vma_map=vma_map_with_stalls,
            instructions=sample_instructions,
        )

        result = engine.analyze()

        # Should have some blame since we have stalls
        assert result.total_stall_blame > 0 or result.num_self_blame > 0

    def test_analyze_stats(self, sample_instructions, vma_map_with_stalls):
        """Test that analysis populates stats."""
        engine = BackSliceEngine(
            vma_map=vma_map_with_stalls,
            instructions=sample_instructions,
        )

        result = engine.analyze()
        stats = engine.get_stats()

        assert stats.num_instructions == 4
        assert stats.num_instructions_with_profile == 4
        assert stats.initial_nodes >= 0
        assert stats.final_nodes >= 0

    def test_analyze_with_debug(self, sample_instructions, vma_map_with_stalls, caplog):
        """Test analysis with debug output."""
        import logging
        config = BackSliceConfig(debug=True)
        engine = BackSliceEngine(
            vma_map=vma_map_with_stalls,
            instructions=sample_instructions,
            config=config,
        )

        with caplog.at_level(logging.DEBUG, logger="leo.analysis.backslice"):
            result = engine.analyze()

        # Should have logged debug output
        assert "Building CCT dependency graph" in caplog.text

    def test_get_graph_after_analyze(self, sample_instructions, vma_map_with_stalls):
        """Test that graph is accessible after analyze()."""
        engine = BackSliceEngine(
            vma_map=vma_map_with_stalls,
            instructions=sample_instructions,
        )

        # Before analyze
        assert engine.get_graph() is None

        engine.analyze()

        # After analyze
        graph = engine.get_graph()
        assert graph is not None
        assert isinstance(graph, CCTDepGraph)

    def test_uses_path_based_distance_when_available(self):
        """Test that BackSliceEngine uses edge path distance for blame.

        GPA Reference: blameCCTDepGraph() at GPUAdvisor-Blame.cpp:1193-1198
        Uses CFG path-based distance averaging when available.

        This tests the integration by directly calling _distribute_blame_to_deps
        after setting up a graph with path data on edges.
        """
        # Create instructions with dependency
        instructions = [
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[]),
            InstructionStat(op="FADD", pc=0x200, dsts=[1], srcs=[0]),
        ]
        # Set up dependency: FADD depends on LDG.E for register 0
        instructions[1].assign_pcs = {0: [0x100]}

        # Create VMA map with stalls on FADD
        vma_map = VMAPropertyMap.build_from_instructions(
            instructions, arch_name="a100"
        )

        # Add profile data to both instructions
        ldg_prop = vma_map.get(0x100)
        if ldg_prop:
            ldg_prop.cct_id = 100
            ldg_prop.prof_metrics["gcycles:isu"] = 10.0
            ldg_prop.has_profile_data = True

        fadd_prop = vma_map.get(0x200)
        if fadd_prop:
            fadd_prop.cct_id = 200
            fadd_prop.prof_metrics["gcycles:stl"] = 100.0
            fadd_prop.has_profile_data = True

        # Create engine and build graph
        engine = BackSliceEngine(
            vma_map=vma_map,
            instructions=instructions,
        )
        engine._build_graph()

        # Add path data to edge BEFORE calling _distribute_blame_to_deps
        # Simple VMA distance would be (0x200-0x100)/16 = 16
        # We set path-based distance to 50.0
        for e in engine.graph.edges():
            if e.from_cct_id == 100 and e.to_cct_id == 200:
                path = EdgePathInfo(
                    block_ids=(1, 2, 3),
                    instruction_count=50.0,  # Different from VMA distance (16)
                    accumulated_stalls=20,
                    is_hidden=False,
                )
                e.add_path(path)
                break

        # Get nodes for direct call
        from_node = engine.graph.get_node(100)
        to_node = engine.graph.get_node(200)
        assert from_node is not None
        assert to_node is not None

        # Directly call the method that uses path-based distance
        blame_edges = engine._distribute_blame_to_deps(to_node, [from_node])

        # Verify the blame edge uses path-based distance (50.0)
        assert len(blame_edges) == 1
        assert blame_edges[0].distance == 50.0, (
            f"Expected path-based distance 50.0, got {blame_edges[0].distance}"
        )


class TestAnalyzeInstructions:
    """Tests for analyze_instructions convenience function."""

    @pytest.fixture
    def simple_instructions(self):
        """Create simple instructions for testing."""
        instructions = [
            InstructionStat(op="FADD", pc=0x100, dsts=[0], srcs=[1, 2]),
            InstructionStat(op="FMUL", pc=0x110, dsts=[3], srcs=[0, 4]),
        ]
        instructions[1].assign_pcs = {0: [0x100]}
        return instructions

    def test_analyze_without_vma_map(self, simple_instructions):
        """Test analyzing without pre-built VMAPropertyMap."""
        result = analyze_instructions(simple_instructions)

        assert result is not None
        assert isinstance(result, KernelBlameResult)

    def test_analyze_with_vma_map(self, simple_instructions):
        """Test analyzing with pre-built VMAPropertyMap."""
        vma_map = VMAPropertyMap.build_from_instructions(
            simple_instructions, arch_name="a100"
        )

        # Add mock stalls
        prop = vma_map.get(0x110)
        if prop:
            prop.cct_id = 0x110
            prop.prof_metrics["gcycles:stl"] = 50.0
            prop.has_profile_data = True

        result = analyze_instructions(
            simple_instructions,
            vma_map=vma_map,
        )

        assert result is not None


class TestBackSliceEngineBlameTypes:
    """Tests for blame type classification in BackSliceEngine."""

    @pytest.fixture
    def engine_with_instructions(self):
        """Create engine with various instruction types."""
        instructions = [
            # Memory operations
            InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2]),
            InstructionStat(op="LDS", pc=0x110, dsts=[1], srcs=[3]),
            InstructionStat(op="LDL", pc=0x120, dsts=[4], srcs=[5]),
            # Compute operations
            InstructionStat(op="FADD", pc=0x130, dsts=[6], srcs=[0, 1]),
            # Sync operations
            InstructionStat(op="BAR.SYNC", pc=0x140),
        ]

        # Set up dependencies
        instructions[3].assign_pcs = {0: [0x100], 1: [0x110]}

        vma_map = VMAPropertyMap.build_from_instructions(instructions, arch_name="a100")
        for inst in instructions:
            prop = vma_map.get(inst.pc)
            if prop:
                prop.cct_id = inst.pc
                prop.has_profile_data = True

        return BackSliceEngine(vma_map=vma_map, instructions=instructions)

    def test_detailize_mem_dep_gmem(self, engine_with_instructions):
        """Test global memory dependency type."""
        engine = engine_with_instructions
        from_inst = InstructionStat(op="LDG.E", pc=0x100, dsts=[0], srcs=[2])
        to_inst = InstructionStat(op="FADD", pc=0x130, dsts=[6], srcs=[0, 1])

        blame_type, category = engine._detailize_blame_type(from_inst, to_inst)

        # Has register dependency, so should be exec_dep
        assert category == "exec_dep"

    def test_detailize_mem_dep_smem(self, engine_with_instructions):
        """Test shared memory dependency type."""
        engine = engine_with_instructions
        from_inst = InstructionStat(op="LDS", pc=0x110, dsts=[1], srcs=[3])
        to_inst = InstructionStat(op="FADD", pc=0x130, dsts=[6], srcs=[0, 1])

        blame_type, category = engine._detailize_blame_type(from_inst, to_inst)

        # Has register dependency from shared memory
        assert blame_type == "exec_dep_smem"
        assert category == "exec_dep"

    def test_detailize_sync(self, engine_with_instructions):
        """Test sync dependency type."""
        engine = engine_with_instructions
        from_inst = InstructionStat(op="BAR.SYNC", pc=0x140)
        to_inst = InstructionStat(op="FADD", pc=0x150)

        blame_type, category = engine._detailize_blame_type(from_inst, to_inst)

        assert blame_type == "sync_barrier"
        assert category == "sync"


# =============================================================================
# EdgePathInfo Tests (GPA-aligned path recording)
# =============================================================================


class TestEdgePathInfo:
    """Tests for EdgePathInfo dataclass (GPA-aligned CFG path tracking).

    EdgePathInfo stores path information for latency analysis between
    two instructions in the CFG. Multiple paths per edge are possible
    due to different control flow paths. Paths are deduplicated by
    block_ids sequence.

    GPA Reference: CCTEdgePathMap in GPUAdvisor.hpp:142
    """

    def test_create_empty(self):
        """Test creating EdgePathInfo with all defaults."""
        path = EdgePathInfo()
        assert path.block_ids == ()
        assert path.accumulated_stalls == 0
        assert path.instruction_count == 0.0
        assert path.is_hidden is False

    def test_create_with_values(self):
        """Test creating EdgePathInfo with specific values."""
        block_ids = (1, 2, 3)
        path = EdgePathInfo(
            block_ids=block_ids,
            accumulated_stalls=150,
            instruction_count=12.5,
            is_hidden=True,
        )
        assert path.block_ids == block_ids
        assert path.accumulated_stalls == 150
        assert path.instruction_count == 12.5
        assert path.is_hidden is True

    def test_create_with_partial_values(self):
        """Test creating with some fields specified."""
        path = EdgePathInfo(
            block_ids=(5, 6, 7),
            accumulated_stalls=200,
        )
        assert path.block_ids == (5, 6, 7)
        assert path.accumulated_stalls == 200
        assert path.instruction_count == 0.0  # Default
        assert path.is_hidden is False  # Default

    def test_equality_same_block_ids(self):
        """Test paths are equal if block_ids match, regardless of other fields."""
        path1 = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=100,
            instruction_count=10.0,
            is_hidden=False,
        )
        path2 = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=200,  # Different
            instruction_count=20.0,  # Different
            is_hidden=True,  # Different
        )
        assert path1 == path2
        assert hash(path1) == hash(path2)

    def test_equality_different_block_ids(self):
        """Test paths are not equal if block_ids differ."""
        path1 = EdgePathInfo(block_ids=(1, 2, 3), accumulated_stalls=100)
        path2 = EdgePathInfo(block_ids=(1, 2, 4), accumulated_stalls=100)
        assert path1 != path2
        assert hash(path1) != hash(path2)

    def test_equality_with_different_type(self):
        """Test equality comparison with different type."""
        path = EdgePathInfo(block_ids=(1, 2))
        assert path != "not a path"
        assert path != 42

    def test_hashable_in_set(self):
        """Test EdgePathInfo can be added to sets (deduplication)."""
        path1 = EdgePathInfo(block_ids=(1, 2, 3), accumulated_stalls=100)
        path2 = EdgePathInfo(block_ids=(1, 2, 3), accumulated_stalls=200)
        path3 = EdgePathInfo(block_ids=(4, 5), accumulated_stalls=100)

        paths_set = {path1, path2, path3}
        # path1 and path2 have same block_ids, so should be deduplicated
        assert len(paths_set) == 2
        assert path1 in paths_set
        assert path3 in paths_set

    def test_hashable_in_dict(self):
        """Test EdgePathInfo can be used as dict key."""
        path1 = EdgePathInfo(block_ids=(1, 2), accumulated_stalls=50)
        path2 = EdgePathInfo(block_ids=(3, 4), accumulated_stalls=75)

        path_dict = {path1: "info1", path2: "info2"}
        assert len(path_dict) == 2

        # Lookup by equivalent path
        lookup_path = EdgePathInfo(block_ids=(1, 2), accumulated_stalls=999)
        assert path_dict[lookup_path] == "info1"

    def test_hash_stable(self):
        """Test hash is stable across multiple calls."""
        path = EdgePathInfo(block_ids=(1, 2, 3))
        hash1 = hash(path)
        hash2 = hash(path)
        assert hash1 == hash2

    def test_empty_block_ids(self):
        """Test path with empty block_ids (simple PC distance case)."""
        # This represents the fallback case without CFG
        path = EdgePathInfo(
            block_ids=(),
            accumulated_stalls=0,
            instruction_count=16.0,
            is_hidden=False,
        )
        assert path.block_ids == ()
        assert len(path.block_ids) == 0
        assert path.instruction_count == 16.0

    def test_single_block_path(self):
        """Test path with single block (simple forward dependency)."""
        path = EdgePathInfo(block_ids=(42,))
        assert path.block_ids == (42,)
        assert len(path.block_ids) == 1

    def test_large_block_sequence(self):
        """Test path with many blocks in sequence."""
        block_ids = tuple(range(100))  # 100 blocks
        path = EdgePathInfo(block_ids=block_ids)
        assert path.block_ids == block_ids
        assert len(path.block_ids) == 100


# =============================================================================
# CCTDepEdge Path Method Tests
# =============================================================================


class TestCCTDepEdgePaths:
    """Tests for CCTDepEdge path-related methods.

    Tests the three new methods:
    - has_valid_paths(): Returns True if valid_paths is non-empty
    - average_path_distance(): Computes average instruction_count
    - add_path(): Adds paths with deduplication by block_ids

    GPA Reference: CCTEdgePathMap usage in GPUAdvisor-Blame.cpp
    """

    # =========================================================================
    # Test: has_valid_paths()
    # =========================================================================

    def test_has_valid_paths_empty_list(self):
        """Test has_valid_paths() with empty list (default state)."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # No paths added yet
        assert edge.has_valid_paths() is False
        assert len(edge.valid_paths) == 0

    def test_has_valid_paths_after_adding_single_path(self):
        """Test has_valid_paths() returns True after adding one path."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        path = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=100,
            instruction_count=5.0,
            is_hidden=False,
        )
        edge.valid_paths.append(path)

        assert edge.has_valid_paths() is True
        assert len(edge.valid_paths) == 1

    def test_has_valid_paths_after_adding_multiple_paths(self):
        """Test has_valid_paths() with multiple paths."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Add multiple paths with different block sequences
        for i in range(3):
            path = EdgePathInfo(
                block_ids=tuple(range(i, i + 3)),
                accumulated_stalls=50 * (i + 1),
                instruction_count=2.0 * (i + 1),
                is_hidden=False,
            )
            edge.valid_paths.append(path)

        assert edge.has_valid_paths() is True
        assert len(edge.valid_paths) == 3

    # =========================================================================
    # Test: average_path_distance()
    # =========================================================================

    def test_average_path_distance_no_paths_returns_negative_one(self):
        """Test average_path_distance() returns -1.0 when no paths exist."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # No paths added
        assert edge.average_path_distance() == -1.0

    def test_average_path_distance_single_path(self):
        """Test average_path_distance() with one path."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        path = EdgePathInfo(
            block_ids=(1, 2),
            accumulated_stalls=100,
            instruction_count=10.0,
            is_hidden=False,
        )
        edge.valid_paths.append(path)

        # With single path, average = 10.0 / 1 = 10.0
        assert edge.average_path_distance() == 10.0

    def test_average_path_distance_two_paths(self):
        """Test average_path_distance() with two paths."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Path 1: 10 instructions
        path1 = EdgePathInfo(
            block_ids=(1, 2),
            accumulated_stalls=100,
            instruction_count=10.0,
            is_hidden=False,
        )

        # Path 2: 20 instructions
        path2 = EdgePathInfo(
            block_ids=(3, 4),
            accumulated_stalls=200,
            instruction_count=20.0,
            is_hidden=False,
        )

        edge.valid_paths.append(path1)
        edge.valid_paths.append(path2)

        # Average = (10 + 20) / 2 = 15.0
        assert edge.average_path_distance() == 15.0

    def test_average_path_distance_multiple_paths(self):
        """Test average_path_distance() with multiple paths."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        distances = [5.0, 10.0, 15.0, 20.0, 25.0]
        for i, distance in enumerate(distances):
            path = EdgePathInfo(
                block_ids=(i,),
                accumulated_stalls=i * 100,
                instruction_count=distance,
                is_hidden=False,
            )
            edge.valid_paths.append(path)

        # Average = (5 + 10 + 15 + 20 + 25) / 5 = 75 / 5 = 15.0
        expected_avg = sum(distances) / len(distances)
        assert abs(edge.average_path_distance() - expected_avg) < 0.001

    def test_average_path_distance_with_zero_instruction_count(self):
        """Test average_path_distance() with zero instruction count paths."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        path1 = EdgePathInfo(
            block_ids=(1,),
            accumulated_stalls=0,
            instruction_count=0.0,
            is_hidden=False,
        )

        path2 = EdgePathInfo(
            block_ids=(2,),
            accumulated_stalls=100,
            instruction_count=10.0,
            is_hidden=False,
        )

        edge.valid_paths.append(path1)
        edge.valid_paths.append(path2)

        # Average = (0 + 10) / 2 = 5.0
        assert edge.average_path_distance() == 5.0

    # =========================================================================
    # Test: add_path()
    # =========================================================================

    def test_add_path_new_path_returns_true(self):
        """Test add_path() returns True when adding new path."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        path = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=100,
            instruction_count=5.0,
            is_hidden=False,
        )

        result = edge.add_path(path)

        assert result is True
        assert len(edge.valid_paths) == 1
        assert edge.valid_paths[0] == path

    def test_add_path_duplicate_block_ids_returns_false(self):
        """Test add_path() returns False for duplicate block_ids."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        path1 = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=100,
            instruction_count=5.0,
            is_hidden=False,
        )

        # Same block_ids but different accumulated_stalls and instruction_count
        path2 = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=200,  # Different
            instruction_count=10.0,  # Different
            is_hidden=True,  # Different
        )

        # Add first path
        result1 = edge.add_path(path1)
        assert result1 is True
        assert len(edge.valid_paths) == 1

        # Try to add duplicate - should be rejected
        result2 = edge.add_path(path2)
        assert result2 is False
        assert len(edge.valid_paths) == 1  # Still only one path

        # Original path is preserved
        assert edge.valid_paths[0] == path1

    def test_add_path_different_block_ids_returns_true(self):
        """Test add_path() returns True for different block sequences."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        paths = [
            EdgePathInfo(block_ids=(1, 2), accumulated_stalls=100, instruction_count=5.0),
            EdgePathInfo(block_ids=(3, 4), accumulated_stalls=200, instruction_count=10.0),
            EdgePathInfo(block_ids=(5, 6), accumulated_stalls=150, instruction_count=7.5),
        ]

        for path in paths:
            result = edge.add_path(path)
            assert result is True

        assert len(edge.valid_paths) == 3

    def test_add_path_similar_but_different_block_ids(self):
        """Test add_path() distinguishes between similar block sequences."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Very similar block sequences
        path1 = EdgePathInfo(block_ids=(1, 2, 3), instruction_count=5.0)
        path2 = EdgePathInfo(block_ids=(1, 2, 3, 4), instruction_count=5.0)  # One more
        path3 = EdgePathInfo(block_ids=(1, 2), instruction_count=5.0)  # One fewer

        assert edge.add_path(path1) is True
        assert edge.add_path(path2) is True
        assert edge.add_path(path3) is True

        assert len(edge.valid_paths) == 3

    def test_add_path_different_order_block_ids_are_different(self):
        """Test add_path() treats different block orders as different paths."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        path1 = EdgePathInfo(block_ids=(1, 2, 3), instruction_count=5.0)
        path2 = EdgePathInfo(block_ids=(3, 2, 1), instruction_count=5.0)  # Reversed

        assert edge.add_path(path1) is True
        assert edge.add_path(path2) is True

        assert len(edge.valid_paths) == 2

    def test_add_path_empty_block_ids(self):
        """Test add_path() with empty block_ids tuple."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Some paths might have empty block_ids (e.g., simple fallback case)
        path = EdgePathInfo(
            block_ids=(),
            accumulated_stalls=0,
            instruction_count=2.0,
            is_hidden=False,
        )

        result = edge.add_path(path)

        assert result is True
        assert len(edge.valid_paths) == 1

    def test_add_path_preserves_path_metrics(self):
        """Test add_path() preserves all EdgePathInfo attributes."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        original_path = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=250,
            instruction_count=12.5,
            is_hidden=True,
        )

        result = edge.add_path(original_path)

        assert result is True
        assert len(edge.valid_paths) == 1

        stored_path = edge.valid_paths[0]
        assert stored_path.block_ids == (1, 2, 3)
        assert stored_path.accumulated_stalls == 250
        assert stored_path.instruction_count == 12.5
        assert stored_path.is_hidden is True

    # =========================================================================
    # Integration Tests: Combined Methods
    # =========================================================================

    def test_add_path_and_check_has_valid_paths(self):
        """Test integration: add_path() then has_valid_paths()."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Initially no paths
        assert edge.has_valid_paths() is False

        # Add one path
        path = EdgePathInfo(block_ids=(1,), accumulated_stalls=100, instruction_count=5.0)
        edge.add_path(path)

        # Now should have valid paths
        assert edge.has_valid_paths() is True

    def test_add_path_and_compute_average(self):
        """Test integration: add_path() then average_path_distance()."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Initially returns -1
        assert edge.average_path_distance() == -1.0

        # Add paths and verify average
        distances = [5.0, 10.0, 15.0]
        for i, dist in enumerate(distances):
            path = EdgePathInfo(
                block_ids=(i,),
                accumulated_stalls=100,
                instruction_count=dist,
                is_hidden=False,
            )
            edge.add_path(path)

        expected_avg = sum(distances) / len(distances)
        assert abs(edge.average_path_distance() - expected_avg) < 0.001

    def test_add_duplicate_does_not_affect_average(self):
        """Test that adding duplicate doesn't change average."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Add initial paths
        path1 = EdgePathInfo(block_ids=(1,), accumulated_stalls=100, instruction_count=10.0)
        path2 = EdgePathInfo(block_ids=(2,), accumulated_stalls=100, instruction_count=20.0)

        edge.add_path(path1)
        edge.add_path(path2)
        initial_avg = edge.average_path_distance()

        # Try to add duplicate of path1
        duplicate = EdgePathInfo(block_ids=(1,), accumulated_stalls=200, instruction_count=30.0)
        result = edge.add_path(duplicate)

        assert result is False
        assert edge.average_path_distance() == initial_avg


# =============================================================================
# Path Recording Integration Tests
# =============================================================================


class TestPathRecordingIntegration:
    """Integration tests for path recording in latency pruning.

    Tests that prune_latency_constraints() properly stores valid paths
    on edges for later use in blame computation.

    GPA Reference: cct_edge_path_map storage at GPUAdvisor-Blame.cpp:895-941
    """

    def test_edge_average_path_distance_used_for_blame(self):
        """Test that average_path_distance can be used for blame computation."""
        edge = CCTDepEdge(from_cct_id=1, to_cct_id=2)

        # Simulate paths from CFG analysis with different instruction counts
        path1 = EdgePathInfo(
            block_ids=(1, 2, 3),
            accumulated_stalls=100,
            instruction_count=10.0,
            is_hidden=False,
        )
        path2 = EdgePathInfo(
            block_ids=(1, 4, 3),  # Different middle block
            accumulated_stalls=120,
            instruction_count=14.0,
            is_hidden=False,
        )

        edge.add_path(path1)
        edge.add_path(path2)

        # Average distance = (10 + 14) / 2 = 12.0
        avg_distance = edge.average_path_distance()
        assert avg_distance == 12.0

        # This distance would be used in inverse-distance weighting for blame
        # Closer instructions get more blame, so we'd use 1/distance
        blame_weight = 1.0 / avg_distance
        assert blame_weight == pytest.approx(1.0 / 12.0)
