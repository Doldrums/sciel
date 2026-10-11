import sys

from ..evidence import restart_trace_values

sys.modules[__name__] = restart_trace_values
