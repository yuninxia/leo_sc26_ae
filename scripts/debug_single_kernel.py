"""Debug Leo analysis on a single kernel with detailed edge tracing."""

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <measurements_dir> <arch>")
        sys.exit(1)

    meas_dir = Path(sys.argv[1])
    arch_name = sys.argv[2]

    from leo.analysis.program_analysis import analyze_program
    from leo.analysis.graph import (
        build_cct_dep_graph, prune_opcode_constraints, prune_latency_constraints,
    )
    from leo.analyzer import KernelAnalyzer, AnalysisConfig
    from leo.db.reader import HPCToolkitReader
    from leo.db.discovery import discover_databases
    from leo.arch import get_architecture

    arch = get_architecture(arch_name)
    discovery = discover_databases(meas_dir)
    reader = HPCToolkitReader(discovery.database)

    # Use the program analysis to find the top kernel
    result = analyze_program(
        meas_dir, arch_name=arch_name, top_n=1,
        sort_by="stall_cycles",
    )

    if not result.kernels:
        print("No kernels found")
        return

    k = result.kernels[0]
    print(f"Kernel: {k.kernel_name}")
    print(f"GPU Binary: {k.gpubin}")
    print(f"Total stall cycles: {k.total_stall_cycles}")
    print()

    # Now re-run the analysis manually with edge tracing
    config = AnalysisConfig(arch_name=arch_name, debug=True)
    analyzer = KernelAnalyzer(
        gpubin_path=Path(k.gpubin),
        hpctoolkit_reader=reader,
        config=config,
    )

    # Access internal state after analysis
    ar = k.analysis_result
    if ar is None:
        print("No analysis result")
        return

    # Check the raw dependency graph from the analyzer
    # We need to trace through the actual backslice engine
    print("=" * 80)
    print("EDGE ANALYSIS")
    print("=" * 80)

    # Re-run the kernel analyzer to get access to internals
    from leo.analysis.backslice import BackSliceEngine
    from leo.analysis.vma_property import VMAPropertyMap
    from leo.binary.instruction import InstructionStat

    # Access the analysis result's debug info
    if hasattr(ar, '_debug_info'):
        print(f"Debug info: {ar._debug_info}")

    # Print the stall table with root causes
    print("\nStall table from analysis:")
    for entry in ar.stall_table:
        stall_op = entry.get('stall_opcode', '?')
        root_op = entry.get('root_opcode', '?')
        root_loc = entry.get('root_cause_location', '?')
        stall_loc = entry.get('stall_location', '?')
        cycles = entry.get('stall_cycles', 0)
        print(f"  {stall_op:20s} <- {root_op:25s}  ({cycles:>12,} cycles)  {stall_loc} <- {root_loc}")

    # Now let's manually rebuild the graph to debug edge types
    print("\n" + "=" * 80)
    print("MANUAL GRAPH REBUILD WITH EDGE TRACING")
    print("=" * 80)

    # Get functions and instructions from the analysis
    # We need to hook into the internal state
    # The best approach: patch the pruning functions to print debug info

    # Check if there's a way to get the graph
    if hasattr(ar, 'dependency_graph'):
        graph = ar.dependency_graph
        print(f"Graph: {len(list(graph.nodes()))} nodes, {len(list(graph.edges()))} edges")
        for edge in graph.edges():
            etype = graph.get_edge_type(edge.from_cct_id, edge.to_cct_id)
            fn = graph.get_node(edge.from_cct_id)
            tn = graph.get_node(edge.to_cct_id)
            fi = fn.instruction if fn else None
            ti = tn.instruction if tn else None
            fop = fi.op if fi else "?"
            top = ti.op if ti else "?"
            print(f"  {fop:25s} -> {top:25s}  type={etype}")


if __name__ == "__main__":
    main()
