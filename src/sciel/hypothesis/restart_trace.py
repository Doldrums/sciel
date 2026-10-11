import sys

from ..evidence import restart_trace

sys.modules[__name__] = restart_trace
