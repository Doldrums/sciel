"""Opt-in input selection policy for a local R request, never text truncation."""
from copy import deepcopy
import hashlib

SCHEMA = 'sciel-reflection-token-budget/1'


def validate_context_tokens(config):
    if 'reflection_context_tokens' not in config:
        return None
    value = config['reflection_context_tokens']
    if type(value) is not int or value <= 0:
        raise ValueError('reflection_context_tokens must be a positive integer')
    return value


def contract(config, llm):
    context = validate_context_tokens(config)
    if context is None:
        return None
    if getattr(llm, 'local_inference', False) is not True:
        raise ValueError('reflection_context_tokens requires the pinned local backend')
    output = getattr(llm, 'max_tokens', None)
    if type(output) is not int or output <= 0 or output >= context:
        raise ValueError('reflection context must exceed its reserved output tokens')
    if not callable(getattr(llm, 'count_prompt_tokens', None)):
        raise ValueError('reflection token budget requires an exact local token counter')
    return dict(schema=SCHEMA, context_tokens=context, reserved_output_tokens=output,
                tokenizer=deepcopy(llm.prompt_tokenizer_identity))


class ReflectionBudgetExceeded(ValueError):
    def __init__(self, diagnostic):
        super().__init__('reflection fixed framing exceeds configured token budget')
        self.diagnostic = deepcopy(diagnostic)


class PromptBudget:
    def __init__(self, identity, count):
        if (type(identity) is not dict or set(identity) != {
                'schema', 'context_tokens', 'reserved_output_tokens', 'tokenizer'}
                or identity.get('schema') != SCHEMA or not callable(count)
                or type(identity['context_tokens']) is not int
                or type(identity['reserved_output_tokens']) is not int
                or not 0 < identity['reserved_output_tokens'] < identity['context_tokens']
                or type(identity['tokenizer']) is not dict or not identity['tokenizer']):
            raise ValueError('reflection token budget requires its frozen contract and counter')
        self.identity, self.count = deepcopy(identity), count

    def measure(self, prompt):
        tokens = self.count(prompt)
        if type(tokens) is not int or tokens < 0:
            raise ValueError('reflection token counter returned an invalid count')
        return dict(self.identity, prompt_tokens=tokens,
                    total_reserved_tokens=tokens + self.identity['reserved_output_tokens'],
                    prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest())

    def fits(self, prompt):
        return self.measure(prompt)['total_reserved_tokens'] <= self.identity['context_tokens']

    def require_framing(self, prompt):
        measured = self.measure(prompt)
        if measured['total_reserved_tokens'] > self.identity['context_tokens']:
            raise ReflectionBudgetExceeded(dict(measured, prompt=prompt,
                reason='fixed_framing_exceeds_context', generation_dispatch='not_dispatched'))
