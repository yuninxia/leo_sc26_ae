# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

OPENMP_IDLE = "OpenMP Idle"
OPENMP_OVERHEAD = "OpenMP Overhead"
OPENMP_WAIT = "OpenMP Wait"
OPENMP_WORK = "OpenMP Work"
OPENMP_OTHER = "OpenMP Other"

OPENMP_TABLE = {
    "<omp idle>": OPENMP_IDLE,
    "<omp overhead>": OPENMP_OVERHEAD,
    "<omp barrier wait>": OPENMP_WAIT,
    "<omp task wait>": OPENMP_WAIT,
    "<omp mutex wait>": OPENMP_WAIT,
    "<omp region unresolved>": OPENMP_WORK,
    "<omp work>": OPENMP_WORK,
    "<omp expl task>": OPENMP_WORK,
    "<omp impl task>": OPENMP_WORK,
}
