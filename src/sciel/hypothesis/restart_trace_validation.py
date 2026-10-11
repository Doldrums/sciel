import sys

from ..evidence import restart_trace_validation

sys.modules[__name__] = restart_trace_validation
