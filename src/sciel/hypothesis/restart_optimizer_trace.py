import sys

from ..evidence import restart_optimizer_trace

sys.modules[__name__] = restart_optimizer_trace
