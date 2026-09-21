"""Explicit inert provider answer for tests concerned with other research paths.

This supplies no source interpretation, proof, or new experimental evidence.
Historical response fixtures must remain untouched and use legacy parsing.
"""


def unknown_mechanism(record):
    hypothesis = record.get('hypothesis') or {}
    claim = hypothesis.get('text') or hypothesis.get('reasoning')
    return [{'claim_quote': claim[:512] if claim else None,
             'source_quotes': [], 'assessment': 'unknown',
             'direct_effect': 'The inert provider fixture does not assess the source mechanism.',
             'untested_downstream': 'No causal solver effect is established by this fixture.'}]
