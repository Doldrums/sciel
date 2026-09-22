"""Exact local chat input counting with the pinned, already cached tokenizer."""
from copy import deepcopy
import hashlib
import importlib.metadata


class PinnedChatTokenCounter:
    def __init__(self, config, *, tokenizer_snapshot=None):
        # Transformers is already an optional local-training/serving dependency.
        # Loading metadata must never download weights or silently resolve main.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot or config['base_model'],
            revision=config['tokenizer_revision'], trust_remote_code=False,
            local_files_only=True)
        template = self.tokenizer.get_chat_template()
        if (type(template) is not str or hashlib.sha256(template.encode('utf-8')).hexdigest()
                != config['chat_template_sha256']):
            raise ValueError('local token counter chat template differs from pinned identity')
        self._identity = {
            'schema': 'sciel-pinned-chat-token-counter/1',
            'base_model': config['base_model'],
            'tokenizer_revision': config['tokenizer_revision'],
            'chat_template_sha256': config['chat_template_sha256'],
            'transformers_version': importlib.metadata.version('transformers'),
            'tokenizers_version': importlib.metadata.version('tokenizers'),
            'messages': 'single_user', 'add_generation_prompt': True,
            'enable_thinking': False,
        }

    @property
    def identity(self):
        return deepcopy(self._identity)

    def count(self, prompt):
        if type(prompt) is not str:
            raise ValueError('local token counting requires a prompt string')
        ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if type(ids) is not list or any(type(item) is not int for item in ids):
            raise ValueError('local tokenizer did not return a flat token ID list')
        return len(ids)
