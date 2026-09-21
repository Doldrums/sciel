"""Typed, explicitly authorized abandonment of an unresolved provider request.

This is not a response, a free request, or permission to retry its physical key.
Only the authenticated parallel recovery ledger constructs it during replay.
"""
from copy import deepcopy

from .llm.client import LLMError


class AbandonedProviderRequest(LLMError):
    def __init__(self, receipt, metadata):
        super().__init__('Provider request explicitly abandoned; outcome and cost remain unknown')
        self.receipt = deepcopy(receipt)
        self._metadata = deepcopy(metadata)
