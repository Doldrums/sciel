import sys

from ..evidence import restart_trace_recorder

sys.modules[__name__] = restart_trace_recorder
