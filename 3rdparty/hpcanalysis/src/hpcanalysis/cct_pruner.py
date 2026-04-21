# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from hpcanalysis.api.query_api import QueryAPI


class CCTPruningPredicate(ABC):

    @abstractmethod
    def retain_node(
        self,
        node_name: str,
        node_type: str,
        node_metrics: Dict[int, int | float],
        entry_metrics: Dict[int, int | float],
    ) -> Tuple[bool, bool]:
        pass

    def verify_predicate(self, query_api: QueryAPI) -> None:
        pass


class OmitOpenMPImplementation(CCTPruningPredicate):

    def __init__(self, function_pattern: str = "__kmp") -> None:
        super().__init__()
        self._function_pattern = function_pattern

    def retain_node(
        self,
        node_name: str,
        node_type: str,
        node_metrics: Dict[int, int | float],
        entry_metrics: Dict[int, int | float],
    ) -> Tuple[bool, bool]:

        include_node = (
            not node_name.startswith(self._function_pattern)
            if node_name is not None
            else True
        )
        include_subtree = include_node

        return (include_node, include_subtree)


class OmitMPIImplementation(CCTPruningPredicate):

    def retain_node(
        self,
        node_name: str,
        node_type: str,
        node_metrics: Dict[int, int | float],
        entry_metrics: Dict[int, int | float],
    ) -> Tuple[bool, bool]:
        include_node = True
        include_subtree = (
            not (node_name.startswith("MPI_") or node_name.startswith("PMPI_"))
            if node_name is not None
            else True
        )
        return (include_node, include_subtree)


class OmitLinesAndLoops(CCTPruningPredicate):

    def retain_node(
        self,
        node_name: str,
        node_type: str,
        node_metrics: Dict[int, int | float],
        entry_metrics: Dict[int, int | float],
    ) -> Tuple[bool, bool]:
        include_node = node_type == "function"
        include_subtree = True
        return (include_node, include_subtree)


class OmitLowCostNodes(CCTPruningPredicate):

    def __init__(self, cost_threshold: int, metrics: List[str] = ["time"]) -> None:
        super().__init__()
        self._cost_threshold = cost_threshold
        self._metrics = metrics
        self._metric_ids = []

    def retain_node(
        self,
        node_name: str,
        node_type: str,
        node_metrics: Dict[int, int | float],
        entry_metrics: Dict[int, int | float],
    ) -> Tuple[bool, bool]:

        for metric_id in self._metric_ids:
            try:
                cost_percentage = (
                    node_metrics[metric_id] / entry_metrics[metric_id] * 100
                )
            except:
                cost_percentage = 0
            if cost_percentage < self._cost_threshold:
                return (False, False)

        return (True, True)

    def verify_predicate(self, query_api: QueryAPI) -> None:
        metric_queries = list(map(lambda x: f"{x}:sum (i)", self._metrics))
        metric_ids = list(
            query_api.query_metric_descriptions(metric_queries)["id"].to_numpy()
        )
        if not metric_ids:
            raise ValueError(f"None of {self._metrics} metrics found in the database!")
        self._metric_ids = metric_ids


class CCTPruner:

    def __init__(self) -> None:
        self._pruning_predicates: List[CCTPruningPredicate] = []

    def add_pruning_predicate(self, predicate: CCTPruningPredicate) -> None:
        self._pruning_predicates.append(predicate)

    def verify_predicates(self, query_api: QueryAPI) -> None:
        for predicate in self._pruning_predicates:
            predicate.verify_predicate(query_api)

    def prune(
        self,
        node_name: str,
        node_type: str,
        node_metrics: Dict[int, int | float],
        entry_metrics: Dict[int, int | float],
    ) -> Tuple[bool, bool]:
        include_node = True
        include_subtree = True
        for predicate in self._pruning_predicates:
            temp = predicate.retain_node(
                node_name, node_type, node_metrics, entry_metrics
            )
            include_node = include_node and temp[0]
            include_subtree = include_subtree and temp[1]

        return include_node, include_subtree
