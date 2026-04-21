# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

# Metric name pattern: supports NVIDIA gins:stl_* and AMD hierarchical gcycles:stl:mem, gpipe:isu:vec, wildcards
METRICS_QUERY = "(time|cputime|realtime|cycles|gpu|gpuop|gker|gmem|gmset|gxcopy|gicopy|gsync|gins(?::(?!sum|prop|min|max)[a-z0-9_]+)*|(?:gcycles|gpipe)(?::(?!sum|prop|min|max)[a-z0-9_*]+)*)(:(sum|prop|min|max|\((sum|prop|min|max)(,(sum|prop|min|max))*\)))?( \((i|e|p|c)(,(i|e|p|c))*\))?"
CCT_QUERY = "(function|loop|line|instruction)(\(.+\))?(\.(function|loop|line|instruction)(\(.+\))?)*"
PROFILES_QUERY = "(node|rank|thread|gpudevice|gpucontext|gpustream|core)(\([0-9]+(-[0-9]+(:[0-9]+)?)?(,[0-9]+(-[0-9]+)?)*\))?(.(node|rank|thread|gpudevice|gpucontext|gpustream|core)(\([0-9]+(-[0-9]+(:[0-9]+)?)?(,[0-9]+(-[0-9]+)?)*\))?)*"
