"""Intel SWSB (Software Scoreboarding) decoder for XeHPC.

Decodes the SWSB field from GED_GetSWSB() for FourDistPipe encoding mode
(used by Xe HPC / Ponte Vecchio).

Encoding reference:
  Intel Graphics Compiler (IGC) iga_types_swsb.hpp / iga_types_swsb.cpp
  SWSB_ENCODE_MODE::FourDistPipe (value 3) — also covers FourDistPipeReduction

FourDistPipe 8-bit encoding:

  RegDist-only (bits [5:3] = pipe type, bits [2:0] = distance 1-7):
    0x00+d: REG_DIST  (generic)    0x08+d: REG_DIST_ALL
    0x10+d: REG_DIST_FLOAT         0x18+d: REG_DIST_INT
    0x20+d: REG_DIST_LONG          0x28+d: REG_DIST_MATH

  SBID-only (bits [7:5] = token type, bits [4:0] = SBID 0-31):
    0x80+n: DST wait   0xA0+n: SRC wait   0xC0+n: SET (allocate)

  Special:
    0x00: no SWSB      0xF0: NOACCSBSET
"""

from leo.binary.instruction import IntelSWSB


# RegDist pipe-type base values (bits [5:3], masked with 0x38)
_REGDIST_PIPES = {
    0x00: "generic",
    0x08: "all",
    0x10: "float",
    0x18: "int",
    0x20: "long",
    0x28: "math",
}

# SBID token-type base values (bits [7:5], masked with 0xE0)
_SBID_TYPES = {
    0x80: "dst_wait",
    0xA0: "src_wait",
    0xC0: "set",
}


def decode_swsb_xehpc(raw: int) -> IntelSWSB:
    """Decode SWSB field for XeHPC FourDistPipe mode.

    Args:
        raw: Raw uint32_t value from GED_GetSWSB().

    Returns:
        IntelSWSB with decoded fields.
    """
    byte0 = raw & 0xFF

    if byte0 == 0:
        return IntelSWSB(raw=raw)

    # Special token: NOACCSBSET
    if byte0 == 0xF0:
        return IntelSWSB(raw=raw)

    # Check SBID-only encoding first (bit 7 set)
    sbid_base = byte0 & 0xE0
    if sbid_base in _SBID_TYPES:
        return IntelSWSB(
            raw=raw,
            has_sbid=True,
            sbid_type=_SBID_TYPES[sbid_base],
            sbid=byte0 & 0x1F,
        )

    # Check RegDist-only encoding (bits [5:3] = pipe, bits [2:0] = distance)
    dist = byte0 & 0x07
    pipe_base = byte0 & 0x38
    if dist > 0 and pipe_base in _REGDIST_PIPES:
        return IntelSWSB(
            raw=raw,
            has_reg_dist=True,
            reg_dist_pipe=_REGDIST_PIPES[pipe_base],
            reg_dist_distance=dist,
        )

    # Unknown encoding — store raw value for debugging
    return IntelSWSB(raw=raw)
