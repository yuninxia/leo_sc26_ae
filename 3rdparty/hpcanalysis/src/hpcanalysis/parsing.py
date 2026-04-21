# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

import struct

FILE_HEADER_OFFSET = 16


def read_string(data: bytes, offset: int) -> str:
    result = ""
    while True:
        (letter,) = struct.unpack("<c", data[offset : offset + 1])
        letter = letter.decode("ascii")
        if letter == "\x00":
            return result
        result += letter
        offset += 1


def safe_unpack(
    format: str, data: bytes, offset: int, index: int = None, index_length: int = None
) -> tuple:
    length = struct.calcsize(format)
    if index:
        offset += index * (length if index_length is None else index_length)
    return struct.unpack(format, data[offset : offset + length])
