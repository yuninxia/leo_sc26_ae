# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

from hpcanalysis.cct_pruner import CCTPruner
from hpcanalysis.data_analysis import DataAnalysis
from hpcanalysis.hpc_dataframe import HpcDataFrameType


def open_db(
    dir_path: str,
    cct_pruner: CCTPruner = None,
    use_cpp_parser: bool = True,
    hpc_dataframe_type: HpcDataFrameType = "pandas",
    exclude_empty_profiles: bool = True,
    collect_function_index: bool = False,
) -> DataAnalysis:
    if collect_function_index and cct_pruner is not None:
        raise ValueError("Can't collect function index while CCT pruning enabled!")

    from hpcanalysis.data_query import DataQuery
    from hpcanalysis.data_read import DataRead

    data_read = DataRead(
        dir_path, cct_pruner=cct_pruner, collect_function_index=collect_function_index
    )
    if use_cpp_parser:
        from . import hpcanalysis_cpp

        data_read_cpp = hpcanalysis_cpp.DataRead(dir_path)
        data_read.read_metric_descriptions = data_read_cpp.read_metric_descriptions
        data_read.read_profile_descriptions = data_read_cpp.read_profile_descriptions
        data_read.read_profile_slices = data_read_cpp.read_profile_slices
        data_read.read_trace_slices = data_read_cpp.read_trace_slices

    data_query = DataQuery(
        data_read,
        hpc_dataframe_type=hpc_dataframe_type,
        exclude_empty_profiles=exclude_empty_profiles,
    )
    if cct_pruner:
        cct_pruner.verify_predicates(data_query)
    data_analysis = DataAnalysis(data_query)

    return data_analysis
