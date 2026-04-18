"""Tests for execution constraints pruning (GPA's Stage 4).

Tests verify that edges are pruned when source instructions have zero execution count,
and kept when they have non-zero execution count.
"""

import pytest
from leo.binary.instruction import InstructionStat, Control
from leo.analysis.graph import (
    CCTDepGraph,
    CCTDepNode,
    prune_execution_constraints,
)
from leo.analysis.vma_property import VMAProperty, VMAPropertyMap
from leo.constants.metrics import METRIC_GINS_EXE


class TestExecutionPruning:
    """Tests for prune_execution_constraints()."""

    def test_prune_zero_execution_count(self):
        """Edges from instructions with zero execution count should be pruned."""
        # Create a graph with two nodes
        graph = CCTDepGraph()

        # Source instruction (dependency provider)
        from_inst = InstructionStat(op="FADD", pc=0x100, dsts=[0], srcs=[1, 2])
        from_node = CCTDepNode(
            cct_id=1,
            vma=0x100,
            instruction=from_inst,
            stall_cycles=0,
        )

        # Destination instruction (stalled)
        to_inst = InstructionStat(op="FMUL", pc=0x110, dsts=[3], srcs=[0, 4])
        to_node = CCTDepNode(
            cct_id=2,
            vma=0x110,
            instruction=to_inst,
            stall_cycles=100.0,
        )

        graph.add_edge(from_node, to_node, "exec_dep")
        assert graph.edge_count() == 1

        # Create VMAPropertyMap with zero execution count for source
        vma_map = VMAPropertyMap()
        vma_map._map[0x100] = VMAProperty(
            vma=0x100,
            instruction=from_inst,
            cct_id=1,
            prof_metrics={METRIC_GINS_EXE: 0.0},  # Zero execution
            has_profile_data=True,
        )
        vma_map._map[0x110] = VMAProperty(
            vma=0x110,
            instruction=to_inst,
            cct_id=2,
            prof_metrics={METRIC_GINS_EXE: 100.0},
            has_profile_data=True,
        )

        # Prune should remove the edge
        removed = prune_execution_constraints(graph, vma_map)

        assert removed == 1
        assert graph.edge_count() == 0

    def test_keep_nonzero_execution_count(self):
        """Edges from instructions with non-zero execution count should be kept."""
        # Create a graph with two nodes
        graph = CCTDepGraph()

        from_inst = InstructionStat(op="FADD", pc=0x100, dsts=[0], srcs=[1, 2])
        from_node = CCTDepNode(
            cct_id=1,
            vma=0x100,
            instruction=from_inst,
            stall_cycles=0,
        )

        to_inst = InstructionStat(op="FMUL", pc=0x110, dsts=[3], srcs=[0, 4])
        to_node = CCTDepNode(
            cct_id=2,
            vma=0x110,
            instruction=to_inst,
            stall_cycles=100.0,
        )

        graph.add_edge(from_node, to_node, "exec_dep")
        assert graph.edge_count() == 1

        # Create VMAPropertyMap with non-zero execution count for source
        vma_map = VMAPropertyMap()
        vma_map._map[0x100] = VMAProperty(
            vma=0x100,
            instruction=from_inst,
            cct_id=1,
            prof_metrics={METRIC_GINS_EXE: 50.0},  # Non-zero execution
            has_profile_data=True,
        )
        vma_map._map[0x110] = VMAProperty(
            vma=0x110,
            instruction=to_inst,
            cct_id=2,
            prof_metrics={METRIC_GINS_EXE: 100.0},
            has_profile_data=True,
        )

        # Prune should not remove the edge
        removed = prune_execution_constraints(graph, vma_map)

        assert removed == 0
        assert graph.edge_count() == 1

    def test_keep_edge_when_no_execution_metric(self):
        """Edges should be kept when no execution metric is available (conservative)."""
        graph = CCTDepGraph()

        from_inst = InstructionStat(op="FADD", pc=0x100, dsts=[0], srcs=[1, 2])
        from_node = CCTDepNode(
            cct_id=1,
            vma=0x100,
            instruction=from_inst,
        )

        to_inst = InstructionStat(op="FMUL", pc=0x110, dsts=[3], srcs=[0, 4])
        to_node = CCTDepNode(
            cct_id=2,
            vma=0x110,
            instruction=to_inst,
            stall_cycles=100.0,
        )

        graph.add_edge(from_node, to_node, "exec_dep")

        # Create VMAPropertyMap without execution metric
        vma_map = VMAPropertyMap()
        vma_map._map[0x100] = VMAProperty(
            vma=0x100,
            instruction=from_inst,
            cct_id=1,
            prof_metrics={"gcycles": 1000.0},  # No ginst metric
            has_profile_data=True,
        )
        vma_map._map[0x110] = VMAProperty(
            vma=0x110,
            instruction=to_inst,
            cct_id=2,
            prof_metrics={"gcycles": 2000.0},
            has_profile_data=True,
        )

        # Prune should not remove the edge (conservative behavior)
        removed = prune_execution_constraints(graph, vma_map)

        assert removed == 0
        assert graph.edge_count() == 1

    def test_keep_edge_when_no_profile_data(self):
        """Edges should be kept when source has no profile data (conservative)."""
        graph = CCTDepGraph()

        from_inst = InstructionStat(op="FADD", pc=0x100, dsts=[0], srcs=[1, 2])
        from_node = CCTDepNode(
            cct_id=1,
            vma=0x100,
            instruction=from_inst,
        )

        to_inst = InstructionStat(op="FMUL", pc=0x110, dsts=[3], srcs=[0, 4])
        to_node = CCTDepNode(
            cct_id=2,
            vma=0x110,
            instruction=to_inst,
            stall_cycles=100.0,
        )

        graph.add_edge(from_node, to_node, "exec_dep")

        # Create empty VMAPropertyMap (no profile data)
        vma_map = VMAPropertyMap()

        # Prune should not remove the edge
        removed = prune_execution_constraints(graph, vma_map)

        assert removed == 0
        assert graph.edge_count() == 1

    def test_multiple_edges_selective_pruning(self):
        """Only edges with zero execution should be pruned in a multi-edge graph."""
        graph = CCTDepGraph()

        # Three source instructions
        inst1 = InstructionStat(op="FADD", pc=0x100, dsts=[0], srcs=[1])
        node1 = CCTDepNode(cct_id=1, vma=0x100, instruction=inst1)

        inst2 = InstructionStat(op="FMUL", pc=0x110, dsts=[1], srcs=[2])
        node2 = CCTDepNode(cct_id=2, vma=0x110, instruction=inst2)

        inst3 = InstructionStat(op="FADD", pc=0x120, dsts=[2], srcs=[3])
        node3 = CCTDepNode(cct_id=3, vma=0x120, instruction=inst3)

        # One destination
        inst_dst = InstructionStat(op="STG", pc=0x130, dsts=[], srcs=[0, 1, 2])
        node_dst = CCTDepNode(cct_id=4, vma=0x130, instruction=inst_dst, stall_cycles=500.0)

        graph.add_edge(node1, node_dst, "exec_dep")
        graph.add_edge(node2, node_dst, "exec_dep")
        graph.add_edge(node3, node_dst, "exec_dep")
        assert graph.edge_count() == 3

        # Set up: node1 and node3 have non-zero execution, node2 has zero
        vma_map = VMAPropertyMap()
        vma_map._map[0x100] = VMAProperty(
            vma=0x100, instruction=inst1, cct_id=1,
            prof_metrics={METRIC_GINS_EXE: 10.0},
            has_profile_data=True,
        )
        vma_map._map[0x110] = VMAProperty(
            vma=0x110, instruction=inst2, cct_id=2,
            prof_metrics={METRIC_GINS_EXE: 0.0},  # Zero execution
            has_profile_data=True,
        )
        vma_map._map[0x120] = VMAProperty(
            vma=0x120, instruction=inst3, cct_id=3,
            prof_metrics={METRIC_GINS_EXE: 20.0},
            has_profile_data=True,
        )
        vma_map._map[0x130] = VMAProperty(
            vma=0x130, instruction=inst_dst, cct_id=4,
            prof_metrics={METRIC_GINS_EXE: 100.0},
            has_profile_data=True,
        )

        removed = prune_execution_constraints(graph, vma_map)

        assert removed == 1
        assert graph.edge_count() == 2
        # Edge from node2 should be removed
        assert not graph.has_edge(node2.cct_id, node_dst.cct_id)
        # Edges from node1 and node3 should remain
        assert graph.has_edge(node1.cct_id, node_dst.cct_id)
        assert graph.has_edge(node3.cct_id, node_dst.cct_id)
