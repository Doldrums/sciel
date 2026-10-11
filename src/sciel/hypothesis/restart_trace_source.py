import sys

from ..evidence import restart_trace_source

sys.modules[__name__] = restart_trace_source
