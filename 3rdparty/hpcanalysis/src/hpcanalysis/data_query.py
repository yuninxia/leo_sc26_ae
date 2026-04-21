# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import List, Tuple

from hpcanalysis.api.query_api import QueryAPI
from hpcanalysis.api.read_api import ReadAPI
from hpcanalysis.hpc_dataframe import (HpcDataFrame, HpcDataFrameType,
                                       MakeHpcDataFrame)
from hpcanalysis.metrics import GPU_METRICS, TIME_METRICS
from hpcanalysis.queries import CCT_QUERY, METRICS_QUERY, PROFILES_QUERY
from hpcanalysis.tables_format import (format_cct_table,
                                       format_functions_table,
                                       format_load_modules_table,
                                       format_metric_descriptions_table,
                                       format_profile_descriptions_table,
                                       format_source_files_table)


def validate_exp(exp: List[str], pattern: str) -> None:
    if not len(exp):
        raise ValueError("ERROR: Wrong format of query expression!!")
    for item in exp:
        if item != "*" and not re.match(f"^{pattern}$", item):
            raise ValueError("ERROR: Wrong format of query expression!!")


def profiles_exp_to_query(exp: str) -> str:
    exp_array = exp.split(".")
    query_values = {}

    for item in exp_array:
        if item.endswith(")"):
            temp_array = item[:-1].split("(")
            item = temp_array[0]
            values_array = temp_array[1].split(",")
            values = []
            for value in values_array:
                if "-" in value:
                    start = int(value.split("-")[0])
                    end = value.split("-")[1]
                    if ":" in end:
                        step = int(end.split(":")[1])
                        end = int(end.split(":")[0])
                    else:
                        step = 1
                        end = int(end)
                    for i in range(start, end, step):
                        values.append(i)
                else:
                    values.append(int(value))

            values = list(set(values))
            values.sort()
            query_values[item] = values

        else:
            query_values[item] = []

    return query_values


class DataQuery(QueryAPI):
    def __init__(
        self,
        read_api: ReadAPI,
        hpc_dataframe_type: HpcDataFrameType = "pandas",
        exclude_empty_profiles: bool = True,
        eager_load_cct: bool = True,
    ) -> None:
        self._read_api = read_api
        self._make_dataframe = MakeHpcDataFrame(hpc_dataframe_type)
        self._hpc_dataframe_type = hpc_dataframe_type

        self._cct = self._make_dataframe.make([])
        self._profile_descriptions = self._make_dataframe.make([])
        self._metric_descriptions = self._make_dataframe.make([])
        self._profile_slices = self._make_dataframe.make([])
        self._trace_slices = self._make_dataframe.make([])

        self._functions = self._make_dataframe.make([])
        self._source_files = self._make_dataframe.make([])
        self._load_modules = self._make_dataframe.make([])

        self._exclude_empty_profiles = exclude_empty_profiles
        self._cct_index = {}
        self._function_index = {}

        # Eagerly load CCT data to populate source_files, functions, and load_modules
        if eager_load_cct:
            self._ensure_cct_loaded()

    def cct_is_child(self, child_id: int, parent_id: int) -> bool:
        child_node = self._cct_index[child_id]
        parent_depth = self._cct_index[parent_id]["depth"]

        while child_node["depth"] > parent_depth:
            if child_node["parent"] == parent_id:
                return True
            child_node = self._cct_index[child_node["parent"]]
        return False

    def _ensure_cct_loaded(self) -> None:
        """Ensure CCT data is loaded, populating source_files, functions, and load_modules."""
        if not len(self._cct):
            cct, functions, source_files, load_modules, function_index = (
                self._read_api.read_cct()
            )
            self._cct_index = cct
            self._function_index = function_index

            self._cct = self._make_dataframe.make(list(cct.values()), format_cct_table)
            self._functions = self._make_dataframe.make(
                functions, format_functions_table
            )
            self._source_files = self._make_dataframe.make(
                source_files, format_source_files_table
            )
            self._load_modules = self._make_dataframe.make(
                load_modules, format_load_modules_table
            )

    def query_cct(self, cct_exp: str | List[str]) -> HpcDataFrame:
        self._ensure_cct_loaded()

        cct_exp = cct_exp if type(cct_exp) == list else [cct_exp]
        validate_exp(cct_exp, CCT_QUERY)

        if "*" in cct_exp:
            return self._cct.copy()

        final_result = []

        for c_exp in cct_exp:
            valid_expression = True

            exp_array = c_exp.split(").")
            cct_ids = []

            for item in exp_array[::-1]:
                ids = []
                if "(" in item:
                    node_type = item.split("(")[0].strip().lower()
                    identifier = item.split("(")[1].strip()
                    if identifier.endswith(")"):
                        identifier = identifier[:-1]
                else:
                    node_type = item
                    identifier = None

                temp = self._cct[self._cct["type"] == node_type]

                if identifier is not None:
                    if node_type == "function":
                        temp2 = self._make_dataframe.index_to_list(
                            self._functions[
                                self._functions["name"].str.match(
                                    f'^{identifier.replace("*", ".+")}$'
                                )
                            ].index
                        )
                        temp = temp[temp["name"].isin(temp2)]
                    elif node_type == "instruction":
                        module_path = identifier.split(":")[0].strip()
                        offset = int(identifier.split(":")[1].strip())
                        temp2 = self._make_dataframe.index_to_list(
                            self._load_modules[
                                self._load_modules["module_path"].str.match(
                                    f'^{module_path.replace("*", ".+")}$'
                                )
                            ].index
                        )
                        temp = temp[
                            temp["module_path"].isin(temp2) & temp["offset"] == offset
                        ]
                    else:
                        file_path = identifier.split(":")[0].strip()
                        line = int(identifier.split(":")[1].strip())
                        temp2 = self._make_dataframe.index_to_list(
                            self._source_files[
                                self._source_files["file_path"].str.match(
                                    f'^{file_path.replace("*", ".+")}$'
                                )
                            ].index
                        )
                        temp = temp[
                            temp["file_path"].isin(temp2) & temp["line"] == line
                        ]

                ids = self._make_dataframe.index_to_list(temp.index)

                if not ids:
                    valid_expression = False
                    break

                cct_ids.append(ids)

            if not valid_expression:
                continue

            if len(cct_ids) == 1:
                final_result.extend(cct_ids[0])
                continue

            stack_of_ids = [cct_ids[0][:]]
            stack_of_paths = []
            result = []

            while True:

                if len(stack_of_ids) >= len(cct_ids):
                    result.append(stack_of_paths.pop())
                    stack_of_ids.pop()
                    while len(stack_of_ids) > 1:
                        stack_of_ids.pop()

                if not stack_of_ids[-1]:
                    break

                up_level = cct_ids[len(stack_of_ids)]
                leaf_node = False
                if len(stack_of_ids) == 1:
                    leaf_node = True
                next_id = stack_of_ids[-1].pop()

                if leaf_node:
                    stack_of_paths.append(next_id)

                filtered_list = list(
                    filter(lambda x: self.cct_is_child(next_id, x), up_level)
                )

                if filtered_list:
                    stack_of_ids.append(filtered_list)
                else:
                    stack_of_paths.pop()

            final_result.extend(result)

        return self._cct.loc[final_result]

    def query_profile_descriptions(self, profiles_exp: str | List[str]) -> HpcDataFrame:
        if not len(self._profile_descriptions):
            self._profile_descriptions = self._make_dataframe.make(
                self._read_api.read_profile_descriptions(),
                format_profile_descriptions_table,
            )

            if self._exclude_empty_profiles:
                self._profile_descriptions = self._profile_descriptions[
                    (self._profile_descriptions["ctx_samples"] > 0)
                    | (self._profile_descriptions.index == 0)
                ]

        profiles_exp = profiles_exp if type(profiles_exp) == list else [profiles_exp]
        validate_exp(profiles_exp, PROFILES_QUERY)

        if "*" in profiles_exp:
            return self._profile_descriptions.copy()

        profiles_indices = []
        for item in profiles_exp:
            item_query = profiles_exp_to_query(item)
            boolean_exp = True
            for item_key in item_query:
                if item_query[item_key] != []:
                    boolean_exp &= self._profile_descriptions[item_key].isin(
                        item_query[item_key]
                    )
                else:
                    boolean_exp &= self._profile_descriptions[item_key].notna()
            profiles_indices.extend(
                self._make_dataframe.index_to_list(
                    self._profile_descriptions[boolean_exp].index.unique()
                )
            )

        return self._profile_descriptions.loc[profiles_indices]

    def query_metric_descriptions(self, metrics_exp: str | List[str]) -> HpcDataFrame:
        if not len(self._metric_descriptions):
            self._metric_descriptions = self._make_dataframe.make(
                self._read_api.read_metric_descriptions(),
                format_metric_descriptions_table,
            )

        metrics_exp = metrics_exp if type(metrics_exp) == list else [metrics_exp]
        validate_exp(metrics_exp, METRICS_QUERY)

        if "*" in metrics_exp:
            return self._metric_descriptions.copy()

        metric_ids = []

        # AMD hierarchical metric prefixes (e.g., gcycles:stl:mem, gpipe:isu:vec)
        amd_hierarchical_prefixes = ("gcycles", "gpipe")

        for item in metrics_exp:
            name = item.split(" (")[0].strip() if " (" in item else item
            aggregations = []
            use_pattern_match = False

            # Check if this is an AMD hierarchical metric
            if name.lower().startswith(amd_hierarchical_prefixes):
                # AMD metrics: colons are part of the name hierarchy, not aggregation
                # Check if the last segment is an aggregation
                if ":" in name:
                    base, last_segment = name.rsplit(":", 1)
                    last_segment = last_segment.strip().lower()
                    valid_aggs = {"sum", "prop", "min", "max"}

                    if last_segment.startswith("(") and last_segment.endswith(")"):
                        values = [v.strip().lower() for v in last_segment[1:-1].split(",") if v.strip()]
                        if values and all(v in valid_aggs for v in values):
                            name = base.strip()
                            aggregations = values
                    elif last_segment in valid_aggs:
                        name = base.strip()
                        aggregations = [last_segment]

                # Check for wildcard pattern matching
                if "*" in name:
                    use_pattern_match = True
            elif ":" in name:
                # Existing logic for non-AMD metrics (gins, etc.)
                base, aggregation = name.rsplit(":", 1)
                aggregation = aggregation.strip()
                valid_aggs = {"sum", "prop", "min", "max"}

                if aggregation.startswith("(") and aggregation.endswith(")"):
                    values = [value.strip().lower() for value in aggregation[1:-1].split(",") if value.strip()]
                    if values and all(value in valid_aggs for value in values):
                        name = base.strip()
                        aggregations = values
                else:
                    aggregation = aggregation.lower()
                    if aggregation in valid_aggs:
                        name = base.strip()
                        aggregations = [aggregation]

            names = (
                TIME_METRICS
                if name == "time" or name in TIME_METRICS
                else list(GPU_METRICS.keys()) if name == "gpu" else [name]
            )
            scopes = item.split(" (")[1][:-1].split(",") if " (" in item else []

            # Build the name filter condition
            if use_pattern_match:
                # Convert query pattern to regex (e.g., gcycles:stl:* -> ^gcycles:stl:.+$)
                pattern = "^" + name.replace("*", ".+") + "$"
                name_filter = self._metric_descriptions["name"].str.match(pattern, case=False)
            else:
                # Exact match (case-insensitive for AMD metrics)
                if name.lower().startswith(amd_hierarchical_prefixes):
                    name_filter = self._metric_descriptions["name"].str.lower().isin([n.lower() for n in names])
                else:
                    name_filter = self._metric_descriptions["name"].isin(names)

            metric_ids.extend(
                self._make_dataframe.index_to_list(
                    self._metric_descriptions[
                        name_filter
                        & (
                            self._metric_descriptions["scope"].isin(scopes)
                            if len(scopes)
                            else True
                        )
                        & (
                            self._metric_descriptions["aggregation"].isin(aggregations)
                            if len(aggregations)
                            else True
                        )
                    ].index.unique()
                )
            )

        return self._metric_descriptions.loc[metric_ids]

    def query_profile_slices(
        self,
        cct_exp: str | List[str] | List[int],
        profiles_exp: str | List[str],
        metrics_exp: str | List[str],
        thread_count: int = None,
    ) -> HpcDataFrame:
        thread_count = thread_count or os.cpu_count()

        profile_indices = (
            self.query_profile_descriptions(profiles_exp).index.unique()
            if profiles_exp != "summary"
            else self._make_dataframe.make_index([0])
        )

        add_application = False

        if self._read_api.has_cct_pruner() and not len(self._cct):
            self.query_cct("*")

        cct_exp = cct_exp if type(cct_exp) == list else [cct_exp]
        if type(cct_exp[0]) == int:
            cct_indices = cct_exp
        else:
            if "application" in cct_exp:
                cct_exp.remove("application")
                add_application = True
            cct_indices = (
                self._make_dataframe.index_to_list(
                    self.query_cct(cct_exp).index.unique()
                )
                if cct_exp
                else []
            )

        if add_application:
            cct_indices.append(0)

        metric_indices = self._make_dataframe.index_to_list(
            self.query_metric_descriptions(metrics_exp)["id"].unique()
        )

        read_profile_indices = self._make_dataframe.index_to_list(
            profile_indices.difference(
                self._profile_slices.index.get_level_values(
                    0
                ).to_numpy()  # TODO: check this - for pandas, i might not need 'to_numpy'
            ).unique()
        )
        profile_indices = self._make_dataframe.index_to_list(profile_indices)
        read_cct_indices = (
            self._make_dataframe.index_to_list(self._cct.index.unique())
            if self._read_api.has_cct_pruner()
            else []
        )

        if add_application and len(read_cct_indices):
            read_cct_indices.append(0)
        read_metric_indices = []

        read_cct_indices.sort()
        read_metric_indices.sort()

        table = self._make_dataframe.make(
            self._read_api.read_profile_slices(
                read_profile_indices,
                read_cct_indices,
                read_metric_indices,
                thread_count or os.cpu_count(),
            )
        )

        if len(table):
            table = table.set_index(
                [
                    "profile_id",
                    "cct_id",
                    "metric_id",
                ]
            )

        self._profile_slices = self._make_dataframe.concat(
            [self._profile_slices, table]
        )

        return self._profile_slices[
            self._profile_slices.index.get_level_values(0).isin(profile_indices)
            & self._profile_slices.index.get_level_values(1).isin(cct_indices)
            & self._profile_slices.index.get_level_values(2).isin(metric_indices)
        ]

    def query_trace_slices(
        self,
        profiles_exp: str | List[str],
        time_frame: Tuple[int, int] = None,
        thread_count: int = None,
    ) -> HpcDataFrame:
        thread_count = thread_count or os.cpu_count()

        profiles_index = self.query_profile_descriptions(profiles_exp).index.unique()
        profiles_indices = self._make_dataframe.index_to_list(profiles_index)

        import pandas as pd

        # TODO: fix the bellow code
        def group_f(x: pd.Series) -> pd.Series:
            return pd.Series(
                {
                    "start_timestamp": x.iloc[0]["start_timestamp"],
                    "end_timestamp": x.iloc[-1]["end_timestamp"],
                }
            )

        merge_dict = {}
        # TODO: merge previous arrays

        if len(self._trace_slices):
            temp = (
                (
                    self._trace_slices[
                        self._trace_slices["profile_id"].isin(profiles_indices)
                        & self._trace_slices["start_timestamp"]
                        >= time_frame[0] & self._trace_slices["end_timestamp"]
                        <= time_frame[1]
                    ]
                    if time_frame
                    else self._trace_slices[
                        self._trace_slices["profile_id"].isin(profiles_indices)
                    ]
                )
                .groupby("profile_id")
                .apply(group_f)
            ).index.unique()

            read_dict = {
                prof_index: [time_frame] if time_frame else []
                for prof_index in profiles_index.difference(
                    self._make_dataframe.index_to_list(temp)
                )
                .unique()
                .tolist()
            }

            if time_frame:

                for id, row in temp.iterrows():
                    time_frames = []

                    if (
                        not len(
                            self._trace_slices[
                                self._trace_slices["end_timestamp"]
                                == row["start_timestamp"]
                            ]
                        )
                        and time_frame[0] != row["start_timestamp"]
                    ):
                        time_frames.append((time_frame[0], row["start_timestamp"]))

                    if (
                        not len(
                            self._trace_slices[
                                self._trace_slices["start_timestamp"]
                                == row["end_timestamp"]
                            ]
                        )
                        and row["end_timestamp"] != time_frame[1]
                    ):
                        time_frames.append((row["end_timestamp"], time_frame[1]))

                    if len(time_frames):
                        read_dict[id] = time_frames

        else:
            read_dict = {
                prof_index: [time_frame] if time_frame else []
                for prof_index in profiles_indices
            }

        self._trace_slices = self._make_dataframe.concat(
            [
                self._trace_slices,
                self._make_dataframe.make(
                    self._read_api.read_trace_slices(read_dict, thread_count)
                ),
            ]
        ).sort_values(
            "start_timestamp"
        )  # TODO: sort_values will fail if the table is empty. manage this edge case

        return (
            self._trace_slices[
                self._trace_slices["profile_id"].isin(profiles_indices)
                & self._trace_slices["start_timestamp"]
                >= time_frame[0] & self._trace_slices["end_timestamp"]
                <= time_frame[1]
            ]
            if time_frame
            else self._trace_slices[
                self._trace_slices["profile_id"].isin(profiles_indices)
            ]
        )
