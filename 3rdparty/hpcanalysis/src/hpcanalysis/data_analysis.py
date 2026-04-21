# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import pandas as pd
from hatchet.graphframe import GraphFrame

from hpcanalysis.api.query_api import QueryAPI
from hpcanalysis.metrics import GPU_METRICS
from hpcanalysis.mpi import MPI_OTHER, MPI_TABLE
from hpcanalysis.openmp import OPENMP_OTHER, OPENMP_TABLE
from hpcanalysis.utils import is_notebook


def reconstruct_subcct(
    cct: pd.DataFrame,
    subcct: pd.DataFrame,
) -> pd.DataFrame:

    temp_dict = {}

    for id, node in subcct.iterrows():

        temp_dict[id] = []

        while not pd.isna(node["parent"]):
            parent_id = node["parent"]

            if parent_id not in temp_dict:
                temp_dict[parent_id] = [id]
            elif id not in temp_dict[parent_id]:
                temp_dict[parent_id].append(id)

            id = parent_id
            node = cct.loc[parent_id]

    return cct.loc[temp_dict.keys()]


class DataAnalysis:
    def __init__(self, query_api: QueryAPI) -> None:
        self._query_api = query_api

    def to_hatchet(self) -> GraphFrame:
        from hatchet.graphframe import Frame, Graph, Node

        cct = self._query_api.query_cct("*")
        metrics = self._query_api.query_metric_descriptions("*")
        summary_profile = self._query_api.query_profile_slices(
            "summary", "*", "*"
        ).droplevel(level=0)

        root = cct[(cct["type"] == "entry") & (cct["name"] == 1)]
        entry = root.iloc[0]
        entry_id = root.index.tolist()[0]

        hatchet_roots = []
        metrics_table = []

        def build_node(cct_id: int, parent: Node = None) -> None:
            node = cct.loc[cct_id]
            if node["type"] == "function":
                node_text = self._query_api._functions.loc[node["name"]]["name"]
            elif node["type"] == "instruction":
                node_text = f"{self._query_api._load_modules.loc[node['module_path']]['module_path']}:{node['offset']}"
            elif node["type"] == "entry":
                node_text = "entry"
            else:
                node_text = f"{self._query_api._source_files.loc[node['file_path']]['file_path']}:{node['line']}"

            hatchet_node = Node(
                Frame({"name": node_text, "type": node["type"]}),
                parent=parent,
                hnid=cct_id,
            )
            if parent is not None:
                parent.add_child(hatchet_node)
            else:
                hatchet_roots.append(hatchet_node)

            hatchet_row = {
                "node": hatchet_node,
                "name": node_text,
            }

            node_metrics = summary_profile[
                summary_profile.index.get_level_values(0) == cct_id
            ].droplevel(level=0)
            node_metrics = node_metrics.merge(
                metrics, how="left", left_on="metric_id", right_on="id"
            )
            node_metrics["metric_name"] = (
                node_metrics["name"]
                + ":"
                + node_metrics["aggregation"]
                + " ("
                + node_metrics["scope"]
                + ")"
            )
            node_metrics["name"] = node_text
            node_metrics["node"] = hatchet_node
            node_metrics = node_metrics[["node", "name", "metric_name", "value"]]

            metrics_table.append(node_metrics)

            for child in node["children"]:
                build_node(child, hatchet_node)

        build_node(entry_id)
        metrics_table = (
            pd.concat(metrics_table)
            .pivot(index=["node", "name"], columns="metric_name", values="value")
            .reset_index(level=1)
            .rename_axis(None, axis=1)
        )
        graphframe = GraphFrame(Graph(hatchet_roots), metrics_table, [0], None)
        return graphframe

    def visualize_cct(
        self,
        cct_exp: str = None,
        metrics_exp: str = "time:sum (i)",
        max_depth: int = None,
        text_view: bool = False,
        cct_indices: List[int] = [],
        label_nodes_type: bool = False,
    ) -> None:
        # TODO: visualizing multiple metrics
        jupyter_notebook = is_notebook() and not text_view

        if jupyter_notebook:
            from ipytree import Node, Tree
        else:
            from treelib import Tree

        tree = Tree()

        cct = self._query_api.query_cct("*")

        if cct_exp:
            cct = reconstruct_subcct(cct, self._query_api.query_cct(cct_exp))
        elif len(cct_indices):
            cct = reconstruct_subcct(cct, cct.loc[cct_indices])

        if metrics_exp is not None:
            summary_profile = self._query_api.query_profile_slices(
                "summary", cct.index.unique().tolist(), metrics_exp
            ).droplevel(level=[0, 2])

        def build_node(cct_id: int, parent) -> None:
            node = cct.loc[cct_id]
            if node["type"] == "function":
                try:
                    node_name = node["name"]
                    temp = self._query_api._functions.loc[node_name]
                    identifier = temp["name"]
                except:
                    identifier = "<NOT_FOUND>"
            elif node["type"] == "instruction":
                identifier = f"{self._query_api._load_modules.loc[node['module_path']]['module_path']}:{node['offset']}"
            else:
                identifier = f"{self._query_api._source_files.loc[node['file_path']]['file_path']}:{node['line']}"

            if metrics_exp is not None:

                try:
                    metric_value = f'{summary_profile.loc[cct_id]["value"]:.2e}'
                except:
                    metric_value = None
            else:
                metric_value = None

            node_text = (
                f"{metric_value}: {identifier}"
                if metric_value is not None
                else identifier
            )
            node_text = (
                f"{node['type']}: {node_text}" if label_nodes_type else node_text
            )

            if jupyter_notebook:
                tree_node = Node(node_text, opened=False)
                parent.add_node(tree_node)

            else:
                tree_node = tree.create_node(
                    node_text,
                    parent=parent,
                )

            if max_depth is None or node["depth"] < max_depth:
                for child in node["children"]:
                    if child in cct.index:
                        build_node(child, tree_node)

        root = cct[(cct["type"] == "entry") & (cct["name"] == 1)]  # TODO: check this
        entry = root.iloc[0]
        entry_id = root.index.tolist()[0]

        try:
            entry_text = f"{summary_profile.loc[entry_id]['value']:.2e}: entry"
        except:
            entry_text = "entry"  # TODO: check this

        if jupyter_notebook:
            tree_node = Node(entry_text, opened=False)
            tree.add_node(tree_node)
        else:
            tree_node = tree.create_node(entry_text)

        for child in entry["children"]:
            build_node(child, tree_node)

        if jupyter_notebook:
            return tree
        else:
            print(tree.show(stdout=False))

    def hpcreport(self, verbose: bool = False) -> pd.DataFrame:
        cct = self._query_api.query_cct("*")

        cpu_table = (
            self._query_api.query_profile_slices(
                "summary",
                [
                    "application",
                    "function(MPI_*)",
                    "function(PMPI_*)",
                    "function(<omp *>)",
                ],
                "time:sum (i)",
            )
            .droplevel(level=[0, 2])
            .reset_index()
        )

        def cpu_category_f(x: pd.Series) -> str:
            if x["cct_id"] == 0:
                return "CPU total"
            temp: str = self._query_api._functions.loc[cct.loc[x["cct_id"]]["name"]][
                "name"
            ]
            if temp.startswith("PMPI_"):
                temp = temp[1:]
            if verbose:
                return temp
            return (
                MPI_TABLE.get(temp, MPI_OTHER)
                if temp.startswith("MPI_")
                else OPENMP_TABLE.get(temp, OPENMP_OTHER)
            )

        cpu_table["MINOR"] = cpu_table.apply(cpu_category_f, axis=1)
        cpu_table = (
            cpu_table.drop("cct_id", axis=1)
            .groupby("MINOR")
            .sum()
            .reset_index()
            .rename({"value": "TIME"}, axis=1)
            .sort_values("TIME", ascending=False)
        )

        cpu_table["PERCENTAGE (%)"] = cpu_table.apply(
            lambda x: round(
                x["TIME"]
                / cpu_table[cpu_table["MINOR"] == "CPU total"].iloc[0]["TIME"]
                * 100,
                2,
            ),
            axis=1,
        )

        cpu_unit = (
            "sec"
            if self._query_api.query_metric_descriptions("time (i)").iloc[0]["unit"]
            == "sec"
            else "cycles"
        )

        cpu_table["MAJOR"] = f"CPU ({cpu_unit})"

        gpu_table = (
            self._query_api.query_profile_slices(
                "summary", "application", "gpu:sum (i)"
            )
            .droplevel(level=[0, 1])
            .reset_index()
        )

        def gpu_category_f(x: pd.Series) -> str:
            metrics = self._query_api.query_metric_descriptions("*")

            temp = GPU_METRICS.get(
                metrics[
                    (metrics["id"] == x["metric_id"])
                    & (metrics["aggregation"] == "sum")
                ]["name"].iloc[0],
                "GPU Other",
            )
            return temp if temp != "GPU all operations" else "GPU total"

        gpu_table["MINOR"] = gpu_table.apply(gpu_category_f, axis=1)
        gpu_table = (
            gpu_table.drop("metric_id", axis=1)
            .rename({"value": "TIME"}, axis=1)
            .sort_values("TIME", ascending=False)[["MINOR", "TIME"]]
        )

        gpu_table["PERCENTAGE (%)"] = round(
            gpu_table["TIME"]
            / gpu_table[gpu_table["MINOR"] == "GPU total"].iloc[0]["TIME"]
            * 100,
            2,
        )

        gpu_table["MAJOR"] = "GPU (sec)"

        return pd.concat([cpu_table, gpu_table], ignore_index=True).set_index(
            ["MAJOR", "MINOR"]
        )

    def flat_profile(
        self,
        cct_exp: Union[str, List[str]],
        profiles_exp: str = "rank",
        include_percentage: bool = True,
    ) -> pd.DataFrame:

        cct = self._query_api.query_cct(cct_exp)
        cct_types = list(cct["type"].unique())
        if cct_types != [] and cct_types != ["function"]:
            raise ValueError("ERROR: Flat profile allowed only for the functions!")

        cct = (
            (cct.rename({"name": "name_id"}, axis=1))
            .reset_index()
            .merge(
                self._query_api._functions,
                how="left",
                left_on="name_id",
                right_on="id",
            )[["id", "name"]]
            .set_index("id")
            .rename({"name": "function"}, axis=1)
        )

        cct_exp = cct_exp if type(cct_exp) == list else [cct_exp]
        if include_percentage:
            cct_exp.append("application")

        table = (
            self._query_api.query_profile_slices(
                profiles_exp,
                cct_exp,
                "time:sum (i)",
            )
            .droplevel(level=2)
            .reset_index(level=1)
        )

        profile_tuples = list(map(lambda x: x.split("(")[0], profiles_exp.split(".")))

        profile_descriptions = self._query_api.query_profile_descriptions(profiles_exp)[
            profile_tuples
        ]

        percentage_table = None

        if include_percentage:
            percentage_table = (
                table[table["cct_id"] == 0][["value"]]
                .merge(
                    profile_descriptions, how="left", left_index=True, right_index=True
                )
                .groupby(profile_tuples)
                .sum()
                .rename({"value": "percentage (%)"}, axis=1)
            )
            table = table[table["cct_id"] != 0]

        table = (
            table.reset_index()
            .merge(cct, how="left", left_on="cct_id", right_on="id")
            .drop("cct_id", axis=1)
            .merge(
                profile_descriptions, how="left", left_on="profile_id", right_on="id"
            )
            .drop("profile_id", axis=1)
            .groupby(profile_tuples + ["function"])
            .sum()
        )

        if include_percentage:
            table = (
                table.reset_index()
                .merge(
                    percentage_table,
                    how="left",
                    left_on=profile_tuples,
                    right_on=profile_tuples,
                )
                .set_index(profile_tuples + ["function"])
            )
            table["percentage (%)"] = round(
                table["value"] / table["percentage (%)"] * 100, 2
            )

        cpu_unit = (
            "sec"
            if self._query_api.query_metric_descriptions("time (i)").iloc[0]["unit"]
            == "sec"
            else "cycles"
        )

        function_list = table.index.get_level_values("function").unique()

        if len(function_list) == 1:
            table = table.droplevel("function").rename(
                {"value": f"{function_list[0]} ({cpu_unit})"}, axis=1
            )

        else:
            table = table.rename({"value": f"time ({cpu_unit})"}, axis=1)

        return table

    def gpu_idleness(self, ranks: List[int] = []) -> pd.DataFrame:
        profiles_exp = f"rank{'' if ranks == [] else '(' + ','.join(str(rank) for rank in ranks) + ')'}"
        profile_descriptions = self._query_api.query_profile_descriptions(profiles_exp)[
            ["rank"]
        ]
        metric_descriptions = self._query_api.query_metric_descriptions(
            ["time:sum (i)", "gpuop:sum (i)"]
        )[["id", "name"]]

        table = (
            self._query_api.query_profile_slices(
                profiles_exp,
                "application",
                ["time:sum (i)", "gpuop:sum (i)"],
            )
            .droplevel(level=1)
            .reset_index()
            .merge(metric_descriptions, how="left", left_on="metric_id", right_on="id")
            .drop(["id", "metric_id"], axis=1)
        )

        table = table[
            table["profile_id"].isin(table[table["name"] == "gpuop"]["profile_id"])
        ].set_index("profile_id")

        table["GPU idle time"] = table[table["name"] != "gpuop"]["value"]

        table = (
            table[table["name"] == "gpuop"]
            .drop("name", axis=1)
            .rename({"value": "GPU total time"}, axis=1)
        )

        table["GPU idle time"] -= table["GPU total time"]

        table = (
            table.merge(
                profile_descriptions, how="left", left_index=True, right_index=True
            )
            .groupby("rank")
            .sum()
            .sort_values("GPU total time", ascending=False)
        )

        return table

    def load_imbalance(
        self,
        cct_exp: Union[str, List[str]],
        profiles_exp: Union[str, List[str]],
        metrics_exp: str = "time (i)",
        tasks_count: int = 1,
    ) -> pd.DataFrame:
        self._query_api._read_api._tasks_count = tasks_count

        cct = self._query_api.query_cct(cct_exp)
        cct_types = list(cct["type"].unique())
        if cct_types != [] and cct_types != ["function"]:
            raise ValueError("ERROR: Load imbalance allowed only for the functions!")

        cct = (
            (cct.rename({"name": "name_id"}, axis=1))
            .reset_index()
            .merge(
                self._query_api._functions,
                how="left",
                left_on="name_id",
                right_on="id",
            )[["id", "name"]]
            .rename({"name": "function"}, axis=1)
        )

        table = self._query_api.query_profile_slices(
            profiles_exp,
            cct_exp,
            metrics_exp,
        )

        def group_f(x: pd.DataFrame) -> pd.Series:
            mean = x["value"].mean()

            return pd.Series(
                {
                    "mean": mean,
                    "mean / max": mean / x["value"].max(),
                    "variance": x["value"].var(),
                }
            )

        table = (
            table.droplevel(level=[0, 2])
            .merge(cct, how="left", left_on="cct_id", right_on="id")
            .drop("id", axis=1)
            .groupby("function")
            .apply(group_f)
            .sort_values(["variance", "mean / max"], ascending=False)
        )

        return table
