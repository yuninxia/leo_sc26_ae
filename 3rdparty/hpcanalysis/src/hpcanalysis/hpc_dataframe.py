# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import pandas as pd

try:
    import cudf

    CUDF_AVAILABLE = True
    HpcDataFrameType = Literal["pandas", "cudf"]
    HpcDataFrame = pd.DataFrame | cudf.core.dataframe.DataFrame
except ImportError:
    cudf = None
    CUDF_AVAILABLE = False
    HpcDataFrameType = Literal["pandas"]
    HpcDataFrame = pd.DataFrame


class MakeHpcDataFrame:

    def __init__(self, dataframe_type: str) -> None:
        self._dataframe_type = dataframe_type

    def make(
        self,
        data: list,
        format_fn: Callable[[pd.DataFrame], pd.DataFrame] = None,
    ) -> HpcDataFrame:
        dataframe = pd.DataFrame(data)
        if format_fn is not None:
            dataframe = format_fn(dataframe)

        if self._dataframe_type == "cudf":
            dataframe = cudf.from_pandas(dataframe)

        return dataframe

    def concat(self, data: list) -> HpcDataFrame:
        return (
            pd.concat(data) if self._dataframe_type == "pandas" else cudf.concat(data)
        )

    def make_index(self, index: list) -> HpcDataFrame:
        return (
            pd.Index(index) if self._dataframe_type == "pandas" else cudf.Index(index)
        )

    def index_to_list(self, index: Any) -> list:
        return (
            index.tolist()
            if self._dataframe_type == "pandas"
            else index.to_arrow().to_pylist()
        )
