# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import struct
from collections import defaultdict
from typing import Dict, List, Tuple

from joblib import Parallel, delayed

from hpcanalysis.api.read_api import ReadAPI
from hpcanalysis.binary_search import (cct_metrics_binary_search,
                                       trace_binary_search)
from hpcanalysis.cct_pruner import CCTPruner
from hpcanalysis.parsing import FILE_HEADER_OFFSET, read_string, safe_unpack

NODE_TYPE_MAPPING = {0: "function", 1: "loop", 2: "line", 3: "instruction"}

METRIC_SCOPE_MAPPING = {
    "execution": "i",
    "function": "e",
    "point": "p",
    "lex_aware": "c",
}

SUMMARY_METRIC_MAPPING = {
    0: "sum",
    1: "min",
    2: "max",
}


def parse_trace(
    trace_file: str, profile_index: int, time_frames: List[Tuple[int, int]]
) -> List[Dict[str, int]]:
    total_rows = []

    with open(trace_file, "rb") as file:
        file.seek(FILE_HEADER_OFFSET)
        formatCtxTraces = "<QQ"
        trace_db = file.read(struct.calcsize(formatCtxTraces))
        (szCtxTraces, pCtxTraces) = safe_unpack(formatCtxTraces, trace_db, 0)

        file.seek(pCtxTraces)
        trace_db = file.read(szCtxTraces)
        (pTraces, _, szTrace) = safe_unpack("<QLB", trace_db, 0)
        (minTimestamp, maxTimestamp) = safe_unpack("<QQ", trace_db, 16)

        pTrace = pTraces + (profile_index - 1) * szTrace
        (profIndex, _, pStart, pEnd) = safe_unpack(
            "<LLQQ", trace_db, pTrace - pCtxTraces
        )
        file.seek(pStart)
        trace_db = file.read(pEnd - pStart)

    sample_format = "<QL"
    sample_size = struct.calcsize(sample_format)
    time_frames = time_frames if len(time_frames) else [(minTimestamp, maxTimestamp)]

    for time_frame in time_frames:
        rows = []

        start_timestamp = time_frame[0]
        end_timestamp = time_frame[1]
        sample_index = trace_binary_search(
            trace_db, 0, (pEnd - pStart) // sample_size, start_timestamp
        )

        while True:
            (timestamp, ctxId) = safe_unpack(
                sample_format, trace_db, sample_index * sample_size
            )

            if timestamp > end_timestamp:
                if len(rows):
                    rows[-1]["end_timestamp"] = timestamp
                break

            if len(rows):
                rows[-1]["end_timestamp"] = timestamp

            if not len(rows) or rows[-1]["cct_id"] != ctxId:
                rows.append(
                    {
                        "profile_id": profile_index,
                        "cct_id": ctxId,
                        "start_timestamp": timestamp,
                        "end_timestamp": timestamp,
                    }
                )

            sample_index += 1
            if sample_index >= (pEnd - pStart) // sample_size:
                if len(rows):
                    rows[-1]["end_timestamp"] = maxTimestamp
                break

        total_rows.extend(rows)

    return total_rows


def extract_metrics(
    profile_index: int,
    cct_id: int,
    metric_indices: List[int],
    formatValues: str,
    values_sub_db: bytes,
    startIndex: int,
    end_index: int,
    rows: list | defaultdict,
) -> None:
    metric_indices_local = metric_indices[:]

    if len(metric_indices_local):
        while len(metric_indices_local):
            metric_id = metric_indices_local.pop()

            result_2 = cct_metrics_binary_search(
                formatValues, values_sub_db, startIndex, end_index, metric_id
            )

            if result_2 != -1:
                (mid_2, metricId, value) = result_2

                if type(rows) == list:

                    rows.append(
                        {
                            "profile_id": profile_index,
                            "cct_id": cct_id,
                            "metric_id": metricId,
                            "value": value,
                        }
                    )

                else:
                    rows[cct_id][metricId] = value

                end_index = mid_2

    else:
        for j in range(startIndex, end_index):
            (metricId, value) = safe_unpack(formatValues, values_sub_db, 0, j)

            if type(rows) == list:

                rows.append(
                    {
                        "profile_id": profile_index,
                        "cct_id": cct_id,
                        "metric_id": metricId,
                        "value": value,
                    }
                )

            else:
                rows[cct_id][metricId] = value


def parse_profile(
    profile_file: str,
    profile_index: int,
    cct_indices: List[int],
    metric_indices: List[int],
    summary_profile_only: bool = False,
) -> List[Dict[str | int, int | float]]:
    rows = defaultdict(dict) if summary_profile_only else []

    with open(profile_file, "rb") as file:
        file.seek(FILE_HEADER_OFFSET)
        formatProfileInfos = "<QQ"
        profile_db = file.read(struct.calcsize(formatProfileInfos))
        (szProfileInfos, pProfileInfos) = safe_unpack(formatProfileInfos, profile_db, 0)

        file.seek(pProfileInfos)
        profile_db = file.read(szProfileInfos)
        (pProfiles, _, szProfile) = safe_unpack("<QLB", profile_db, 0)

        (nValues, pValues, nCtxs, _, pCtxIndices) = safe_unpack(
            "<QQLLQ", profile_db, pProfiles + profile_index * szProfile - pProfileInfos
        )

        file.seek(pCtxIndices)
        formatCtxs = "<LQ"
        cct_sub_db = file.read(nCtxs * struct.calcsize(formatCtxs))

        file.seek(pValues)
        formatValues = "<Hd"
        values_sub_db = file.read(nValues * struct.calcsize(formatValues))

        if len(cct_indices):
            low = 0
            high = nCtxs - 1

            while len(cct_indices):
                cct_id = cct_indices.pop()

                result = cct_metrics_binary_search(
                    formatCtxs, cct_sub_db, low, high, cct_id
                )

                if result != -1:
                    (mid, ctxId, startIndex) = result

                    if mid == nCtxs - 1:
                        end_index = nValues
                    else:
                        (_, end_index) = safe_unpack(formatCtxs, cct_sub_db, 0, mid + 1)

                    high = mid

                    extract_metrics(
                        profile_index,
                        ctxId,
                        metric_indices,
                        formatValues,
                        values_sub_db,
                        startIndex,
                        end_index,
                        rows,
                    )

        else:
            for i in range(nCtxs):
                (ctxId, startIndex) = safe_unpack(formatCtxs, cct_sub_db, 0, i)
                cct_id = ctxId

                if i == nCtxs - 1:
                    end_index = nValues
                else:
                    (_, end_index) = safe_unpack(formatCtxs, cct_sub_db, 0, i + 1)

                extract_metrics(
                    profile_index,
                    ctxId,
                    metric_indices,
                    formatValues,
                    values_sub_db,
                    startIndex,
                    end_index,
                    rows,
                )

    return rows


class DataRead(ReadAPI):
    def __init__(
        self,
        dir_path: str,
        cct_pruner: CCTPruner = None,
        collect_function_index: bool = False,
    ) -> None:
        super().__init__(dir_path, cct_pruner)

        self._context = {}
        self._functions = {}
        self._source_files = {}
        self._load_modules = {}
        self._function_index = {}

        self._collect_function_index = collect_function_index
        self._summary_profile = {}
        self._entry_id = None

        for file_path in os.listdir(self._dir_path):
            if file_path.split(".")[-1] == "db":
                file_path = os.path.join(self._dir_path, file_path)
                try:
                    with open(file_path, "rb") as file:
                        file.seek(10)
                        db = file.read(4)
                    try:
                        format = db.decode("ascii")
                        if format == "meta":
                            self._meta_file = file_path
                        elif format == "prof":
                            self._profile_file = file_path
                        elif format == "ctxt":
                            self._cct_file = file_path
                        elif format == "trce":
                            self._trace_file = file_path
                    except:
                        pass
                except:
                    pass

        for item in ["meta", "profile", "cct"]:
            if not hasattr(self, f"_{item}_file"):
                raise ValueError(f"ERROR: {item}.db not found.")

    def _parse_source_file(self, meta_db: bytes, pFile: int) -> Dict[str, str]:
        if pFile not in self._source_files:
            (pPath,) = safe_unpack(
                "<Q",
                meta_db,
                pFile + struct.calcsize("<Q"),
            )
            self._source_files[pFile] = {
                "id": pFile,
                "file_path": read_string(meta_db, pPath),
            }

        return self._source_files[pFile]

    def _parse_load_module(self, meta_db: bytes, pModule: int) -> Dict[str, str]:
        if pModule not in self._load_modules:
            (pPath,) = safe_unpack(
                "<Q",
                meta_db,
                pModule + struct.calcsize("<Q"),
            )
            self._load_modules[pModule] = {
                "id": pModule,
                "module_path": read_string(meta_db, pPath),
            }

        return self._load_modules[pModule]

    def _parse_function(self, meta_db: bytes, pFunction: int) -> Dict[str, str | int]:
        if pFunction not in self._functions:
            (pName, pModule, offset, pFile, line) = safe_unpack(
                "<QQQQL", meta_db, pFunction
            )

            name = read_string(meta_db, pName)

            if re.fullmatch(
                "P?MPI_.+",
                name,
            ):
                if name.startswith("P"):
                    name = name[1:]
                name = name[: re.match("^P?MPI_[a-zA-Z_]+", name).end()]

            for item in [" [", ".", "@", "(", "<"]:
                if item in name and not name.startswith(item):
                    name = name[: name.index(item)]

            self._functions[pFunction] = {
                "id": pFunction,
                "name": name,
                "line": line,
                "offset": offset,
            }
            if pFile:
                self._functions[pFunction]["file_id"] = self._parse_source_file(
                    meta_db, pFile
                )["id"]
            if pModule:
                self._functions[pFunction]["module_id"] = self._parse_load_module(
                    meta_db, pModule
                )["id"]

        return self._functions[pFunction]

    def _parse_context(
        self,
        current_offset: int,
        total_size: int,
        parent: Dict[str, str | int],
        meta_db: bytes,
        function_parent: Dict[str, str | int] = None,
        function_depth: int = 0,
    ) -> None:
        final_offset = current_offset + total_size

        while current_offset < final_offset:
            (szChildren, pChildren, ctxId, _, lexicalType, nFlexWords) = safe_unpack(
                "<QQLHBB", meta_db, current_offset
            )

            flex_offset = current_offset + 32
            current_offset += 32 + nFlexWords * 8

            node_type = NODE_TYPE_MAPPING[lexicalType]

            node = {
                "id": ctxId,
                "type": node_type,
                "parent": parent["id"],
                "children": [],
                "depth": parent["depth"] + 1,
            }

            name = None

            if nFlexWords:

                if node_type == "function":
                    (pFunction,) = safe_unpack("<Q", meta_db, flex_offset)
                    function_data = self._parse_function(meta_db, pFunction)

                    node["name"] = function_data["id"]
                    node["line"] = function_data["line"]
                    node["offset"] = function_data["offset"]

                    name = function_data["name"]

                    if "file_id" in function_data:
                        node["file_path"] = function_data["file_id"]

                    if "module_id" in function_data:
                        node["module_path"] = function_data["module_id"]

                elif node_type == "instruction":
                    (pModule, offset) = safe_unpack("<QQ", meta_db, flex_offset)
                    node["module_path"] = self._parse_load_module(meta_db, pModule)[
                        "id"
                    ]
                    node["offset"] = offset

                else:
                    (pFile, line) = safe_unpack("<QL", meta_db, flex_offset)
                    node["file_path"] = self._parse_source_file(meta_db, pFile)["id"]
                    node["line"] = line

            include_node, include_subtree = (
                self._cct_pruner.prune(
                    name,
                    node_type,
                    self._summary_profile[ctxId],
                    self._summary_profile[self._entry_id],
                )
                if self._cct_pruner is not None
                else (True, True)
            )

            if self._collect_function_index and node_type == "function":
                self._function_index[node["id"]] = {
                    "function_id": node["id"],
                    "function_parent": (
                        function_parent["id"] if function_parent else None
                    ),
                    "function_children": [],
                    "function_depth": function_depth,
                    "function_name": self._functions[node["name"]]["name"],
                }
                if function_parent:
                    self._function_index[function_parent["id"]][
                        "function_children"
                    ].append(node["id"])

            if include_node:
                parent["children"].append(node["id"])
                self._context[node["id"]] = node
            else:
                node = parent

            if include_subtree:
                self._parse_context(
                    pChildren,
                    szChildren,
                    node,
                    meta_db,
                    function_parent if node_type != "function" else node,
                    function_depth if node_type != "function" else function_depth + 1,
                )

    def read_cct(
        self,
    ) -> Tuple[
        Dict[int, Dict[str, str | int]],
        List[Dict[str, str | int]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        Dict[int, Dict[str, str | int]],
    ]:
        # TODO: for now, parse always summary profile
        self._summary_profile = parse_profile(
            self._profile_file,
            0,
            [],
            [],
            summary_profile_only=True,
        )

        with open(self._cct_file, "rb") as file:
            file.seek(FILE_HEADER_OFFSET)
            formatCtxInfos = "<QQ"
            cct_db = file.read(struct.calcsize(formatCtxInfos))
            (szCtxInfos, pCtxInfos) = safe_unpack(formatCtxInfos, cct_db, 0)

            file.seek(pCtxInfos)
            cct_db = file.read(szCtxInfos)
            (pCtxs, nCtxs, szCtx) = safe_unpack("<QLB", cct_db, 0)

            file.seek(pCtxs)
            cct_db = file.read(nCtxs * szCtx)

        with open(self._meta_file, "rb") as file:
            meta_db = file.read()

        (szContext, pContext) = safe_unpack("<QQ", meta_db, FILE_HEADER_OFFSET + 6 * 8)
        (pEntryPoints, nEntryPoints, szEntryPoint) = safe_unpack(
            "<QHB", meta_db, pContext
        )

        for i in range(nEntryPoints):
            (szChildren, pChildren, ctxId, entryPoint) = safe_unpack(
                "<QQLH",
                meta_db,
                pEntryPoints,
                i,
                szEntryPoint,
            )

            if self._entry_id is None:
                self._entry_id = ctxId

            node = {
                "id": ctxId,
                "type": "entry",
                "name": entryPoint,
                "children": [],
                "depth": 0,
            }
            self._context[node["id"]] = node
            self._parse_context(
                pChildren,
                szChildren,
                node,
                meta_db,
            )

        result = (
            self._context,
            self._functions.values(),
            self._source_files.values(),
            self._load_modules.values(),
            self._function_index,
        )
        self._reset_cct_cache()
        return result

    def _reset_cct_cache(self) -> None:
        self._context = {}
        self._functions = {}
        self._source_files = {}
        self._load_modules = {}
        self._function_index = {}

    def read_metric_descriptions(self) -> List[Dict[str, str | int]]:
        with open(self._meta_file, "rb") as file:
            file.seek(FILE_HEADER_OFFSET + 4 * 8)
            formatMetrics = "<QQ"
            meta_db = file.read(struct.calcsize(formatMetrics))
            (
                szMetrics,
                pMetrics,
            ) = safe_unpack(formatMetrics, meta_db, 0)

            file.seek(pMetrics)
            meta_db = file.read(szMetrics)

        rows = []
        pMetrics_old = pMetrics

        (pMetrics, nMetrics, szMetric, szScopeInst, szSummary) = safe_unpack(
            "<QLBBB", meta_db, 0
        )

        for i in range(nMetrics):
            (pName, pScopeInsts, pSummaries, nScopeInsts, nSummaries) = safe_unpack(
                "<QQQHH", meta_db, pMetrics - pMetrics_old, i, szMetric
            )

            name = read_string(meta_db, pName - pMetrics_old).lower().strip()
            unit = None
            if name.endswith(")"):
                name = name[:-1]
                unit = name.split("(")[1].lower().strip()
                name = name.split("(")[0].lower().strip()

            for j in range(nScopeInsts):
                (pScope, propMetricId) = safe_unpack(
                    "<QH", meta_db, pScopeInsts - pMetrics_old, j, szScopeInst
                )
                (pScopeName,) = safe_unpack("<Q", meta_db, pScope - pMetrics_old)
                scope_name = METRIC_SCOPE_MAPPING[
                    read_string(meta_db, pScopeName - pMetrics_old).lower().strip()
                ]

                rows.append(
                    {
                        "id": propMetricId,
                        "name": name,
                        "aggregation": "prop",
                        "scope": scope_name,
                        "unit": unit,
                    }
                )

            for j in range(nSummaries):
                (pScope, _, combine, _, statMetricId) = safe_unpack(
                    "<QQBBH", meta_db, pSummaries - pMetrics_old, j, szSummary
                )
                (pScopeName,) = safe_unpack("<Q", meta_db, pScope - pMetrics_old)
                scope_name = METRIC_SCOPE_MAPPING[
                    read_string(meta_db, pScopeName - pMetrics_old).lower().strip()
                ]

                summary_name = SUMMARY_METRIC_MAPPING[combine]

                rows.append(
                    {
                        "id": statMetricId,
                        "name": name,
                        "aggregation": summary_name,
                        "scope": scope_name,
                        "unit": unit,
                    }
                )

        return rows

    def read_profile_descriptions(self) -> List[Dict[str, str | int]]:
        check_traces = hasattr(self, "_trace_file")
        identifiers = {}

        if check_traces:
            with open(self._trace_file, "rb") as file:
                file.seek(FILE_HEADER_OFFSET)
                formatCtxTraces = "<QQ"
                trace_db = file.read(struct.calcsize(formatCtxTraces))
                (szCtxTraces, pCtxTraces) = safe_unpack(formatCtxTraces, trace_db, 0)

                file.seek(pCtxTraces)
                trace_db = file.read(szCtxTraces)
                (pTraces, _, szTrace) = safe_unpack("<QLB", trace_db, 0)

        with open(self._meta_file, "rb") as file:
            file.seek(FILE_HEADER_OFFSET + 2 * 8)
            formatIdNames = "<QQ"
            meta_db = file.read(struct.calcsize(formatIdNames))
            (szIdNames, pIdNames) = safe_unpack(formatIdNames, meta_db, 0)

            file.seek(pIdNames)
            meta_db = file.read(szIdNames)
            (ppNames, nKinds) = safe_unpack("<QB", meta_db, 0)

            for i in range(nKinds):
                (pName,) = safe_unpack("<Q", meta_db, ppNames - pIdNames, i)
                identifiers[i] = read_string(meta_db, pName - pIdNames).lower().strip()

        rows = []

        with open(self._profile_file, "rb") as file:
            file.seek(FILE_HEADER_OFFSET)
            formatProfileInfosIdTuples = "<QQQQ"
            profile_db = file.read(struct.calcsize(formatProfileInfosIdTuples))
            (szProfileInfos, pProfileInfos, szIdTuples, pIdTuples) = safe_unpack(
                formatProfileInfosIdTuples, profile_db, 0
            )

            file.seek(pIdTuples)
            profile_db = file.read(szIdTuples)

            file.seek(pProfileInfos)
            profile_db_2 = file.read(szProfileInfos)
            (pProfiles, _, szProfile) = safe_unpack("<QLB", profile_db_2, 0)

            current_offset = 0
            current_index = 1

            while current_offset < szIdTuples:
                row = {"id": current_index}
                rows.append(row)

                (nValues, pValues, nCtxs, _, pCtxIndices) = safe_unpack(
                    "<QQLLQ",
                    profile_db_2,
                    pProfiles + current_index * szProfile - pProfileInfos,
                )
                row["ctx_samples"] = nCtxs
                row["metric_samples"] = nValues

                if check_traces:
                    pTrace = pTraces + (current_index - 1) * szTrace
                    (profIndex, _, pStart, pEnd) = safe_unpack(
                        "<LLQQ", trace_db, pTrace - pCtxTraces
                    )
                    formatSample = "<QL"
                    row["trace_samples"] = (pEnd - pStart) / struct.calcsize(
                        formatSample
                    )

                (nIds,) = safe_unpack("<H", profile_db, current_offset)
                current_offset += 8

                for i in range(nIds):
                    formatIds = "<BBHLQ"
                    (kind, _, _, logicalId, physicalId) = safe_unpack(
                        formatIds, profile_db, current_offset
                    )
                    row[identifiers[kind]] = physicalId if kind in [1, 7] else logicalId
                    current_offset += struct.calcsize(formatIds)

                current_index += 1

        return rows

    def read_profile_slices(
        self,
        profile_indices: List[int],
        cct_indices: List[int],
        metric_indices: List[int],
        thread_count: int = None,
    ) -> List[Dict[str, int | float]]:
        thread_count = thread_count or os.cpu_count()

        with Parallel(n_jobs=thread_count) as parallel:
            delayed_jobs = [
                delayed(parse_profile)(
                    self._profile_file,
                    profile_index,
                    cct_indices[:],
                    metric_indices,
                )
                for profile_index in profile_indices
            ]
            profiles = parallel(delayed_jobs)

        return sum(profiles, [])

    def read_trace_slices(
        self,
        profile_indices: Dict[int, List[Tuple[int, int]]],
        thread_count: int = None,
    ) -> List[Dict[str, int]]:
        if not hasattr(self, "_trace_file"):
            raise ValueError(f"ERROR: trace.db not found.")

        thread_count = thread_count or os.cpu_count()

        with Parallel(n_jobs=thread_count) as parallel:
            delayed_jobs = [
                delayed(parse_trace)(
                    self._trace_file, profile_index, profile_indices[profile_index]
                )
                for profile_index in profile_indices
            ]
            traces = parallel(delayed_jobs)

        return sum(traces, [])
