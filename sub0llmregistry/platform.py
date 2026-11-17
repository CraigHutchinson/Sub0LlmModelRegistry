"""Platform differences, from Sub0Platform (github.com/CraigHutchinson/Sub0Platform): Windows, Linux and WSL.

Install it (`pip install sub0llmregistry` pulls it in), or check Sub0Platform out beside this repository
(../Sub0Platform), which is found without installing.
"""
import os
import sys

try:
    import sub0platform  # noqa: F401
except ImportError:
    _SIBLING = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "Sub0Platform", "python")
    if not os.path.isdir(os.path.join(_SIBLING, "sub0platform")):
        raise ImportError("sub0llmregistry needs the sub0platform package: pip install "
                          "git+https://github.com/CraigHutchinson/Sub0Platform@v0.1.0, or clone Sub0Platform beside "
                          "this repository") from None
    sys.path.append(_SIBLING)

from sub0platform import dirs, env, fs, paths  # noqa: E402,F401
