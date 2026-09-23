"""Exact pre-dispatch auxiliary bounds; no rewriting or paid-call retry."""
from copy import deepcopy
import hashlib

SCHEMA = 'sciel-auxiliary-token-budget/1'


def measure(llm, prompt, *, context_tokens, output_tokens, role, tokenizer=None):
    if (getattr(llm, 'local_inference', False) is not True
            or not callable(getattr(llm, 'count_prompt_tokens', None))
            or type(context_tokens) is not int or type(output_tokens) is not int
            or not 0 < output_tokens < context_tokens
            or role not in ('conclusion', 'research_tool')):
        raise ValueError('auxiliary token budget requires pinned local counter and valid limits')
    identity = deepcopy(llm.prompt_tokenizer_identity)
    if not isinstance(identity, dict) or not identity or (tokenizer is not None and tokenizer != identity):
        raise ValueError('auxiliary token budget tokenizer identity differs')
    limits = getattr(llm, 'limits', None)
    if isinstance(limits, dict) and context_tokens > limits.get('context_tokens', context_tokens):
        raise ValueError('auxiliary token budget exceeds provider context limit')
    tokens = llm.count_prompt_tokens(prompt)
    if type(tokens) is not int or tokens < 1:
        raise ValueError('invalid exact auxiliary prompt token count')
    return dict(schema=SCHEMA, role=role, context_tokens=context_tokens,
                reserved_output_tokens=output_tokens, tokenizer=identity,
                prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                prompt_tokens=tokens, total_reserved_tokens=tokens + output_tokens,
                fits=tokens + output_tokens <= context_tokens)
