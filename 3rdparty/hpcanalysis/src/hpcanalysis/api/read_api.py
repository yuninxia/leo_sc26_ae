# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from hpcanalysis.cct_pruner import CCTPruner


class ReadAPI(ABC):
    def __init__(
        self,
        dir_path: str,
        cct_pruner: CCTPruner = None,
    ) -> None:
        self._dir_path = dir_path
        self._cct_pruner = cct_pruner

    def has_cct_pruner(self) -> bool:
        return (
            self._cct_pruner is not None
            and len(self._cct_pruner._pruning_predicates) > 0
        )

    @abstractmethod
    def read_cct(
        self,
    ) -> Tuple[
        Dict[int, Dict[str, str | int]],
        List[Dict[str, str | int]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        Dict[int, Dict[str, str | int]],
    ]:
        pass

    @abstractmethod
    def read_metric_descriptions(self) -> List[Dict[str, str | int]]:
        pass

    @abstractmethod
    def read_profile_descriptions(self) -> List[Dict[str, str | int]]:
        pass

    @abstractmethod
    def read_profile_slices(
        self,
        profile_indices: Dict[int, List[Tuple[int, int]]],
        thread_count: int = None,
    ) -> List[Dict[str, int | float]]:
        pass

    @abstractmethod
    def read_trace_slices(
        self,
        profile_indices: Dict[int, List[Tuple[int, int]]],
        thread_count: int = None,
    ) -> List[Dict[str, int]]:
        pass
