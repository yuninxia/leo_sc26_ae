# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Tuple

from hpcanalysis.hpc_dataframe import HpcDataFrame


class QueryAPI(ABC):
    @abstractmethod
    def query_cct(self, cct_exp: str | List[str]) -> HpcDataFrame:
        pass

    @abstractmethod
    def query_metric_descriptions(self, metrics_exp: str | List[str]) -> HpcDataFrame:
        pass

    @abstractmethod
    def query_profile_descriptions(self, profiles_exp: str | List[str]) -> HpcDataFrame:
        pass

    @abstractmethod
    def query_profile_slices(
        self,
        cct_exp: str | List[str],
        profiles_exp: str | List[str],
        metrics_exp: str | List[str],
        thread_count: int = None,
    ) -> HpcDataFrame:
        pass

    @abstractmethod
    def query_trace_slices(
        self,
        profiles_exp: str | List[str],
        time_frame: Tuple[int, int] = None,
        thread_count: int = None,
    ) -> HpcDataFrame:
        pass
