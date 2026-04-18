"""Assessment of analysis quality based on debug info and coverage."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from leo.analysis.vma_property import VMAPropertyMap
from leo.binary.debug_info import DebugInfoLevel, DebugInfoResult


class AnalysisQuality(Enum):
    """Assessed quality of analysis results."""

    GOOD = "good"  # > 20% coverage or has debug info
    FAIR = "fair"  # 5-20% coverage
    POOR = "poor"  # < 5% coverage and no debug info


@dataclass
class QualityAssessment:
    """Overall assessment of analysis quality."""

    quality: AnalysisQuality
    profile_coverage: float  # 0.0-1.0: instructions with profile data
    source_coverage: float  # 0.0-1.0: instructions with source mapping
    debug_level: DebugInfoLevel
    warnings: List[str]


def assess_quality(
    vma_map: VMAPropertyMap,
    debug_info: Optional[DebugInfoResult] = None,
    source_mapping: Optional[Dict[int, Tuple[str, int]]] = None,
) -> QualityAssessment:
    """Assess overall quality of analysis.

    Args:
        vma_map: VMAPropertyMap with profile data
        debug_info: Results from inspect_debug_info()
        source_mapping: PC-to-source mapping from hpcstruct

    Returns:
        QualityAssessment with warnings
    """
    warnings: List[str] = []

    # Calculate coverage metrics
    total = len(vma_map)
    sampled = vma_map.get_sampled_count()
    profile_coverage = sampled / total if total > 0 else 0.0

    # Calculate source mapping coverage
    source_coverage = 0.0
    if source_mapping:
        mapped = sum(1 for pc in vma_map.keys() if pc in source_mapping)
        source_coverage = mapped / total if total > 0 else 0.0

    # Get debug level
    debug_level = debug_info.level if debug_info else DebugInfoLevel.UNKNOWN

    # Determine quality level
    if profile_coverage > 0.2 or debug_level in (DebugInfoLevel.FULL, DebugInfoLevel.LINEINFO):
        quality = AnalysisQuality.GOOD
    elif profile_coverage > 0.05:
        quality = AnalysisQuality.FAIR
    else:
        quality = AnalysisQuality.POOR

    # Generate warnings
    if debug_level == DebugInfoLevel.NONE:
        warnings.append("Binary has no debug info (-lineinfo or -G not used)")
        if debug_info:
            warnings.extend(debug_info.recommendations)

    if profile_coverage < 0.05 and total > 100:
        warnings.append(
            f"Very low profile coverage: {sampled}/{total} ({profile_coverage*100:.1f}%)"
        )

    if source_coverage < 0.1 and source_mapping and total > 100:
        warnings.append(
            f"Low source mapping: {source_coverage*100:.1f}% of instructions"
        )

    return QualityAssessment(
        quality=quality,
        profile_coverage=profile_coverage,
        source_coverage=source_coverage,
        debug_level=debug_level,
        warnings=warnings,
    )
