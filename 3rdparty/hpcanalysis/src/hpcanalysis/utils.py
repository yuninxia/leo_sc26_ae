# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

from IPython import get_ipython


def is_notebook() -> bool:
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except:
        return False
