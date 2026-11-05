"""Semantic search over an archive directory; build_archive.py installs this file there as search.py.

Runs offline inside a tool: import search; search.search('query', k=8, collection='papers'|'programs'|None).
"""
import json
import os

# A tool's child process does not inherit the image's environment.
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME='/tmp/hf', TOKENIZERS_PARALLELISM='false',
                  TQDM_DISABLE='1')

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
QUERY_PROMPT = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
PER_SOURCE = 2
_index = None
_encoder = None


def _load_index():
    global _index
    if _index is None:
        with np.load(os.path.join(ROOT, 'index.npz')) as saved:
            _index = {key: saved[key] for key in saved.files}
        _index['vectors'] = _index['vectors'].astype(np.float32)
        _index['names'] = [str(name) for name in _index['collections']]
    return _index


def _load_encoder():
    global _encoder
    if _encoder is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(1)
        path = os.path.join(ROOT, 'model')
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left', local_files_only=True)
        model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True).eval()
        _encoder = torch, tokenizer, model
    return _encoder


def embed(query):
    torch, tokenizer, model = _load_encoder()
    batch = tokenizer([QUERY_PROMPT + query], truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        vector = model(**batch).last_hidden_state[:, -1].float()
    return torch.nn.functional.normalize(vector, dim=-1)[0].numpy()


def _row(index, position):
    with open(os.path.join(ROOT, 'meta.jsonl'), 'rb') as meta:
        meta.seek(int(index['offsets'][position]))
        return json.loads(meta.readline())


def search(query, k=8, collection=None):
    """The k best chunks, at most PER_SOURCE from one source, with repeated text removed. Each result has score
    (cosine), collection, source (URL, PR or run program, for crediting), task, program_score, chunk and text."""
    index = _load_index()
    scores = index['vectors'] @ embed(query)
    if collection is not None:
        if collection not in index['names']:
            raise ValueError(f"collection is one of {index['names']} or None")
        scores[index['collection'] != index['names'].index(collection)] = -2.0
    results, seen, per_source = [], set(), {}
    for position in np.argsort(-scores):
        if len(results) >= min(k, 50) or scores[position] <= -2.0:
            break
        source = int(index['source_ids'][position])
        if per_source.get(source, 0) >= PER_SOURCE:
            continue
        row = _row(index, position)
        if row['text'] in seen:
            continue
        seen.add(row['text'])
        per_source[source] = per_source.get(source, 0) + 1
        results.append({'score': round(float(scores[position]), 4), **row})
    return results


def document(source):
    """Every stored chunk of one result's source, joined; long numeric data was elided at indexing."""
    with open(os.path.join(ROOT, 'meta.jsonl'), 'rb') as meta:
        rows = [json.loads(line) for line in meta]
    return '\n'.join(row['text'] for row in rows if row['source'] == source)
