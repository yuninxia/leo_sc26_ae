# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

FUNCTIONS_INT_COLUMNS = [
    "id",
    "file_id",
    "line",
    "module_id",
    "offset",
]

FUNCTIONS_STRING_COLUMNS = ["name"]

SOURCE_FILES_INT_COLUMNS = ["id"]

SOURCE_FILES_STRING_COLUMNS = ["file_path"]

LOAD_MODULES_INT_COLUMNS = ["id"]

LOAD_MODULES_STRING_COLUMNS = ["module_path"]


METRIC_DESCRIPTIONS_INT_COLUMNS = [
    "id",
]

METRIC_DESCRIPTIONS_STRING_COLUMNS = [
    "name",
    "aggregation",
    "scope",
    "unit",
]

CCT_INT_COLUMNS = [
    "id",
    "parent",
    "depth",
    "name",
    "file_path",
    "line",
    "module_path",
    "offset",
]

CCT_STRING_COLUMNS = [
    "type",
]


PROFILE_DESCRIPTIONS_INT_COLUMNS = [
    "id",
    "node",
    "rank",
    "thread",
    "core",
    "gpustream",
    "gpucontext",
    "gpudevice",
    "ctx_samples",
    "metric_samples",
    "trace_samples",
]

import pandas as pd


def format_profile_descriptions_table(table: pd.DataFrame) -> pd.DataFrame:
    for column in PROFILE_DESCRIPTIONS_INT_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    table[PROFILE_DESCRIPTIONS_INT_COLUMNS] = table[
        PROFILE_DESCRIPTIONS_INT_COLUMNS
    ].astype("Int64")
    table = table[PROFILE_DESCRIPTIONS_INT_COLUMNS]
    return table.set_index("id")


def format_source_files_table(table: pd.DataFrame) -> pd.DataFrame:
    table[SOURCE_FILES_INT_COLUMNS] = table[SOURCE_FILES_INT_COLUMNS].astype("Int64")
    table[SOURCE_FILES_STRING_COLUMNS] = table[SOURCE_FILES_STRING_COLUMNS].astype(
        "string"
    )
    table = table[SOURCE_FILES_INT_COLUMNS + SOURCE_FILES_STRING_COLUMNS]
    return table.set_index("id")


def format_load_modules_table(table: pd.DataFrame) -> pd.DataFrame:
    table[LOAD_MODULES_INT_COLUMNS] = table[LOAD_MODULES_INT_COLUMNS].astype("Int64")
    table[LOAD_MODULES_STRING_COLUMNS] = table[LOAD_MODULES_STRING_COLUMNS].astype(
        "string"
    )
    table = table[LOAD_MODULES_INT_COLUMNS + LOAD_MODULES_STRING_COLUMNS]
    return table.set_index("id")


def format_functions_table(table: pd.DataFrame) -> pd.DataFrame:
    table[FUNCTIONS_INT_COLUMNS] = table[FUNCTIONS_INT_COLUMNS].astype("Int64")
    table[FUNCTIONS_STRING_COLUMNS] = table[FUNCTIONS_STRING_COLUMNS].astype("string")
    table = table[
        FUNCTIONS_INT_COLUMNS[0:1]
        + FUNCTIONS_STRING_COLUMNS
        + FUNCTIONS_INT_COLUMNS[1:]
    ]
    return table.set_index("id")


def format_cct_table(table: pd.DataFrame) -> pd.DataFrame:
    table[CCT_INT_COLUMNS] = table[CCT_INT_COLUMNS].astype("Int64")
    table[CCT_STRING_COLUMNS] = table[CCT_STRING_COLUMNS].astype("string")
    table = table[
        CCT_STRING_COLUMNS[0:1]
        + CCT_INT_COLUMNS[0:2]
        + ["children"]
        + CCT_INT_COLUMNS[2:]
    ]
    return table.set_index("id")


def format_metric_descriptions_table(table: pd.DataFrame) -> pd.DataFrame:
    table[METRIC_DESCRIPTIONS_INT_COLUMNS] = table[
        METRIC_DESCRIPTIONS_INT_COLUMNS
    ].astype("Int64")
    table[METRIC_DESCRIPTIONS_STRING_COLUMNS] = table[
        METRIC_DESCRIPTIONS_STRING_COLUMNS
    ].astype("string")
    table = table[METRIC_DESCRIPTIONS_INT_COLUMNS + METRIC_DESCRIPTIONS_STRING_COLUMNS]
    return table.sort_values("id")
