"""Opcode classification tables for NVIDIA and AMD architectures."""

# NVIDIA opcode prefix tables (uppercase matching)
NVIDIA_MEMORY_PREFIXES = (
    "LD", "LDG", "LDS", "LDL", "LDC",
    "ST", "STG", "STS", "STL",
    "ATOM", "RED", "LDGDEPBAR", "LDGSTS",
)

NVIDIA_SYNC_PREFIXES = (
    "BAR", "MEMBAR", "DEPBAR", "WARPSYNC",
    "BSSY", "BSYNC", "SYNC",
)

NVIDIA_ATOMIC_PREFIXES = (
    "ATOM", "RED",
)

NVIDIA_FLOAT_PREFIXES = (
    "FADD", "FMUL", "FFMA", "FMNMX", "FSET", "FSETP",
    "FCMP", "FCHK", "FABS", "FNEG", "FSWZ",
    "DADD", "DMUL", "DFMA", "DMNMX", "DSET", "DSETP",
    "HADD", "HMUL", "HFMA", "HMNMX", "HSET", "HSETP",
)

NVIDIA_MUFU_PREFIXES = (
    "MUFU", "RCP", "RSQ", "LG2", "EX2", "SIN", "COS",
)

NVIDIA_TENSOR_PREFIXES = (
    "HMMA", "IMMA", "DMMA", "BMMA",
)

# Hopper-specific instruction prefix tables
NVIDIA_WGMMA_PREFIXES = (
    "WGMMA",
)

NVIDIA_DPX_PREFIXES = (
    "VIMNMX", "VIADDMAX", "VIADDMIN", "VIMAX3", "VIBMAX", "VIADD3",
)

NVIDIA_TMA_PREFIXES = (
    "UTMALDG", "USTMG",
)

NVIDIA_MISC_PREFIXES = (
    "MOV", "PRMT", "SEL", "SHFL", "VOTE", "MATCH",
    "S2R", "CS2R", "R2B", "B2R", "LEPC",
    "NOP", "NANOSLEEP",
)

NVIDIA_INTEGER_PREFIXES = (
    "IADD", "IMAD", "IMUL", "ISUB", "IABS", "INEG",
    "IMNMX", "ISETP", "ISET", "ICMP",
    "SHL", "SHR", "SHF", "BFE", "BFI", "FLO", "POPC",
    "LOP", "LOP3", "AND", "OR", "XOR", "NOT",
)

NVIDIA_PREDICATE_PREFIXES = (
    "PSETP", "PSET", "PLOP3", "P2R", "R2P",
)

NVIDIA_CONVERT_PREFIXES = (
    "I2F", "F2I", "I2I", "F2F", "FRND",
)

NVIDIA_CONTROL_PREFIXES = (
    "BRA", "BRX", "JMP", "JMX", "RET", "EXIT",
    "CALL", "CAL", "PRET", "BREAK", "CONT",
    "SYNC", "BAR", "MEMBAR", "DEPBAR", "WARPSYNC",
    "BSSY", "BSYNC", "YIELD",
)

# AMD opcode prefix tables (lowercase matching)
AMD_TENSOR_PREFIXES = (
    "mfma_", "v_mfma",
)

AMD_MEMORY_PREFIXES = (
    "global_", "flat_", "buffer_", "ds_",
    "scratch_", "s_load_", "s_store_", "s_buffer_",
)

AMD_CONVERT_PREFIXES = (
    "v_cvt_", "v_frexp_", "v_ldexp_", "s_cvt_",
)

AMD_MUFU_FUNCS = (
    "rcp", "rsq", "sqrt", "sin", "cos", "log", "exp",
    "rcp_f", "rsq_f", "sqrt_f",
)

AMD_FLOAT_PREFIXES = (
    "v_add_f", "v_sub_f", "v_mul_f", "v_fma_f", "v_mac_f",
    "v_mad_f", "v_min_f", "v_max_f", "v_cmp_f",
    "v_add_co", "v_sub_co",
    "v_fmac", "v_fmaak", "v_fmamk",
    "v_pk_add_f", "v_pk_mul_f", "v_pk_fma_f",
    "v_dot2", "v_dot4", "v_dot8",
)

AMD_FLOAT_SUFFIXES = ("_f16", "_f32", "_f64")

AMD_INTEGER_PREFIXES = (
    "v_add_i", "v_sub_i", "v_mul_i", "v_mad_i",
    "v_add_u", "v_sub_u", "v_mul_u", "v_mad_u",
    "v_lshl", "v_lshr", "v_ashr",
    "v_and_b", "v_or_b", "v_xor_b", "v_not_b",
    "v_bfe", "v_bfi", "v_bcnt", "v_ffb",
    "v_min_i", "v_max_i", "v_min_u", "v_max_u",
    "v_cmp_i", "v_cmp_u", "v_cmp_eq", "v_cmp_ne",
    "v_cmp_lt", "v_cmp_le", "v_cmp_gt", "v_cmp_ge",
    "s_add_i", "s_sub_i", "s_mul_i",
    "s_add_u", "s_sub_u", "s_mul_u",
    "s_lshl", "s_lshr", "s_ashr",
    "s_and_b", "s_or_b", "s_xor_b", "s_not_b",
    "s_bfe", "s_bfi", "s_bcnt", "s_ff",
)

AMD_INTEGER_SUFFIXES = ("_i32", "_u32", "_i64", "_u64")

AMD_CONTROL_PREFIXES = (
    "s_barrier", "s_waitcnt", "s_wait",
    "s_branch", "s_cbranch",
    "s_setpc", "s_swappc", "s_getpc",
    "s_call", "s_return", "s_endpgm",
    "s_sleep", "s_nop", "s_trap",
    "s_sendmsg", "s_sendmsghalt",
    "s_cbranch_g_fork", "s_cbranch_i_fork", "s_cbranch_join",
)

AMD_SYNC_PREFIXES = (
    "s_barrier", "s_waitcnt", "s_wait", "s_sleep",
)

AMD_MISC_PREFIXES = (
    "v_mov", "v_movreld", "v_movrels", "v_readfirstlane",
    "v_readlane", "v_writelane", "v_swap_b",
    "s_mov", "s_movreld", "s_cmov", "s_cselect",
    "s_setreg", "s_getreg",
)

AMD_PREDICATE_PREFIXES = (
    "s_cmp_", "s_bitcmp", "v_cmpx_",
)

AMD_ATOMIC_PREFIXES = (
    "global_atomic_", "flat_atomic_", "buffer_atomic_", "image_atomic_", "s_atomic_",
    "ds_add_", "ds_sub_", "ds_rsub_",
    "ds_inc_", "ds_dec_",
    "ds_min_", "ds_max_",
    "ds_and_", "ds_or_", "ds_xor_",
    "ds_mskor_",
    "ds_cmpst_", "ds_cmpswap_",
    "ds_wrxchg_", "ds_wrap_",
    "ds_add_rtn_", "ds_sub_rtn_",
    "ds_min_rtn_", "ds_max_rtn_",
    "ds_and_rtn_", "ds_or_rtn_", "ds_xor_rtn_",
    "ds_cmpst_rtn_", "ds_cmpswap_rtn_",
)
