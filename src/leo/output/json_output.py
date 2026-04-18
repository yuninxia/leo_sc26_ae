"""JSON output formatter for Leo analysis results.

This module provides the canonical raw data format for Leo's output.
All other presentation formats (summary table, annotated assembly, graphs)
are derived from this JSON representation.

Design Principle: Every field in the JSON is directly extracted from
Leo's analysis - no generated explanations, no inferred suggestions.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from leo.utils.location import format_source_location

if TYPE_CHECKING:
    from leo.analyzer import AnalysisConfig, AnalysisResult
    from leo.analysis.vma_property import VMAPropertyMap


def _location_dict(
    loc: Optional[Tuple[str, int] | Tuple[str, int, int]]
) -> Optional[Dict[str, Any]]:
    """Convert a source location tuple into a dict."""
    if loc is None:
        return None
    loc_dict: Dict[str, Any] = {
        "file": loc[0],
        "line": loc[1],
    }
    if len(loc) > 2 and loc[2] > 0:
        loc_dict["column"] = loc[2]
    return loc_dict


def _format_location_key(loc: Optional[Dict[str, Any]]) -> Optional[str]:
    """Format a location dict as a string key for aggregation."""
    if loc is None:
        return None
    return format_source_location(loc["file"], loc["line"], col=0, short=False)


def to_json_dict(
    result: "AnalysisResult",
    config: "AnalysisConfig",
) -> Dict[str, Any]:
    """Convert analysis result to JSON-serializable dict.

    This is the canonical raw data format. All fields are directly
    extracted from Leo's analysis.

    The output includes:
    - "edges": Raw individual blame edges (for detailed analysis)
    - "aggregated_edges": Edges aggregated by source location and opcode
      (handles loop unrolling by combining multiple PCs at same source line)

    Args:
        result: The AnalysisResult from KernelAnalyzer.analyze()
        config: The AnalysisConfig used for analysis

    Returns:
        Dict suitable for json.dumps()
    """
    blame_result = result.backslice_result
    total_stall_cycles = result.total_stall_cycles
    total_execution_cycles = result.vma_map.get_total_cycles()

    # Get kernel name: prefer resolved name (offset-based, critical for AMD),
    # then config override, then first function name from binary
    kernel_name = result.resolved_kernel_name or config.function_name
    if not kernel_name and result.stats.get("function_names"):
        kernel_name = result.stats["function_names"][0]
    kernel_name = kernel_name or "unknown"

    # Build JSON structure
    json_data: Dict[str, Any] = {
        "version": "1.0",
        "tool": "leo",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kernel": {
            "name": kernel_name,
            "architecture": config.arch,
            "vendor": config.vendor,
        },
        "totals": {
            "stall_cycles": int(total_stall_cycles),
            "execution_cycles": int(total_execution_cycles),
            "blame_edges": len(blame_result.blames),
        },
        "edges": [],
        "aggregated_edges": [],
        "chains": [],
    }

    # For aggregation: group by (stall_location, stall_opcode, cause_location, cause_opcode)
    aggregation: Dict[tuple, Dict[str, Any]] = {}

    # Convert blame edges
    for edge in blame_result.blames:
        stall_loc = _location_dict(
            result.get_source_location(edge.dst_pc, include_column=True)
        )
        root_loc = _location_dict(
            result.get_source_location(edge.src_pc, include_column=True)
        )

        edge_data: Dict[str, Any] = {
            "stall_pc": f"0x{edge.dst_pc:x}",
            "stall_opcode": edge.dst_opcode,
            "root_cause_pc": f"0x{edge.src_pc:x}",
            "root_cause_opcode": edge.src_opcode,
            "cycles": int(edge.total_blame()),
            "distance": int(edge.distance) if edge.distance >= 0 else -1,
        }

        # Add source locations if available
        if stall_loc:
            edge_data["stall_location"] = stall_loc
        if root_loc:
            edge_data["root_cause_location"] = root_loc

        json_data["edges"].append(edge_data)

        # Aggregate by (stall_location, stall_opcode, cause_location, cause_opcode)
        agg_key = (
            _format_location_key(stall_loc),
            edge.dst_opcode,
            _format_location_key(root_loc),
            edge.src_opcode,
        )

        if agg_key not in aggregation:
            aggregation[agg_key] = {
                "stall_location": stall_loc,
                "stall_opcode": edge.dst_opcode,
                "root_cause_location": root_loc,
                "root_cause_opcode": edge.src_opcode,
                "cycles": 0,
                "edge_count": 0,
            }
        aggregation[agg_key]["cycles"] += int(edge.total_blame())
        aggregation[agg_key]["edge_count"] += 1

    # Build aggregated edges list, sorted by cycles descending
    for agg_data in sorted(aggregation.values(), key=lambda x: -x["cycles"]):
        cycles = agg_data["cycles"]
        pct = (cycles / total_stall_cycles * 100) if total_stall_cycles > 0 else 0
        speedup = (
            total_execution_cycles / (total_execution_cycles - cycles)
            if total_execution_cycles > cycles
            else 1.0
        )

        agg_edge: Dict[str, Any] = {
            "stall_opcode": agg_data["stall_opcode"],
            "root_cause_opcode": agg_data["root_cause_opcode"],
            "cycles": cycles,
            "percent": round(pct, 2),
            "speedup": round(speedup, 3),
            "edge_count": agg_data["edge_count"],
        }

        if agg_data["stall_location"]:
            agg_edge["stall_location"] = agg_data["stall_location"]
        if agg_data["root_cause_location"]:
            agg_edge["root_cause_location"] = agg_data["root_cause_location"]

        json_data["aggregated_edges"].append(agg_edge)

    # Update totals with aggregated count
    json_data["totals"]["aggregated_edges"] = len(json_data["aggregated_edges"])

    # Extract multi-hop dependency chains
    if blame_result.blame_chains:
        for i, bc in enumerate(blame_result.blame_chains, 1):
            path_nodes: List[Dict[str, Any]] = []
            for node in bc.nodes:
                node_data: Dict[str, Any] = {
                    "pc": f"0x{node.pc:x}",
                    "opcode": node.opcode,
                }
                loc = _location_dict(
                    result.get_source_location(node.pc, include_column=True)
                )
                if loc:
                    node_data["location"] = loc
                path_nodes.append(node_data)

            chain: Dict[str, Any] = {
                "id": i,
                "stall_pc": f"0x{bc.stall_pc:x}",
                "cycles": int(bc.total_blame),
                "depth": bc.depth,
                "path": path_nodes,
            }
            json_data["chains"].append(chain)
    else:
        # Fallback: generate 2-node chains from blame edges
        for i, edge in enumerate(blame_result.blames, 1):
            stall_loc = _location_dict(
                result.get_source_location(edge.dst_pc, include_column=True)
            )
            root_loc = _location_dict(
                result.get_source_location(edge.src_pc, include_column=True)
            )

            chain = {
                "id": i,
                "stall_pc": f"0x{edge.dst_pc:x}",
                "cycles": int(edge.total_blame()),
                "depth": 1,
                "path": [
                    {
                        "pc": f"0x{edge.dst_pc:x}",
                        "opcode": edge.dst_opcode,
                    },
                    {
                        "pc": f"0x{edge.src_pc:x}",
                        "opcode": edge.src_opcode,
                    },
                ],
            }
            if stall_loc:
                chain["path"][0]["location"] = stall_loc
            if root_loc:
                chain["path"][1]["location"] = root_loc

            json_data["chains"].append(chain)

    return json_data
