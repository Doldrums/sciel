#!/usr/bin/env python3
"""Build a searchable archive directory for research_tools.data_env: vectors, metadata, model and search.py.

    build_archive.py --runs DIR --program-files DIR --texts DIR --model SNAPSHOT --output OUT

--runs: directories holding SCiEL runs; each checkpoint.json gives its best programs, deduplicated by source hash.
--program-files: loose .py/.c sources. --texts: .pdf/.md/.txt files; a text file may open with 'source: ...' and
'task: ...' lines naming what a result should credit. Embeddings are cached in OUT.work by chunk, so an interrupted
or repeated build embeds only what is new. Stages can run on different machines by copying OUT.work.
A rebuild changes the directory's digest, which runs bind in their identity: write new archives to a new OUT.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np

CHUNK_CHARS = 1800
OVERLAP_CHARS = 200
MAX_PROGRAM_CHUNKS = 12
MAX_TOKENS = 512
RUN_DEPTH = 4
MODEL_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors')


def elide_data(source):
    """Drop what embeds badly and what a reader cannot use in a snippet: long lines of literal numbers."""
    kept = []
    for line in source.splitlines():
        if len(line) > 400 and sum(ch.isdigit() for ch in line) > .5 * len(line):
            kept.append(f'{line[:60]}... <{len(line)} chars of data elided>')
        else:
            kept.append(line[:2000])
    return '\n'.join(kept)


def chunks_of(text):
    """Pieces of at most CHUNK_CHARS, cut at paragraph or line ends where there is one, with a little overlap."""
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            floor = start + CHUNK_CHARS // 2
            for mark in ('\n\n', '\n', '. '):
                cut = text.rfind(mark, floor, end)
                if cut > 0:
                    end = cut + len(mark)
                    break
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return [piece for piece in pieces if piece]


def run_programs(roots):
    """Best-per-island and population-best programs of every checkpoint under roots."""
    found, checkpoints = {}, 0
    for root in roots:
        for here, dirs, files in os.walk(root):
            if 'checkpoint.json' not in files:
                if here[len(root):].count(os.sep) >= RUN_DEPTH:
                    dirs[:] = []
                continue
            dirs[:] = []
            try:
                with open(os.path.join(here, 'checkpoint.json')) as saved:
                    state = json.load(saved)
                db, task = state['db'], state['identity']['task']
                programs = {program['id']: program for program in db['programs']}
                best = {db.get('pop_best_id')} | {island.get('best_id') for island in db['islands']}
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f'skip {here}: {type(error).__name__}', flush=True)
                continue
            checkpoints += 1
            for ident in best - {None}:
                program = programs.get(ident)
                if program is None or not program.get('code') or program.get('score') is None:
                    continue
                key = hashlib.sha256(program['code'].encode()).hexdigest()
                if key not in found or program['score'] > found[key]['score']:
                    found[key] = dict(sha256=key, code=program['code'], task=task, score=float(program['score']),
                                      source=f'{os.path.relpath(here, root)}:{ident}')
    print(f'{checkpoints} checkpoints, {len(found)} distinct best programs', flush=True)
    return found.values()


def loose_programs(roots):
    for root in roots:
        for here, _, files in os.walk(root):
            for name in sorted(files):
                if name.endswith(('.py', '.c')):
                    path = os.path.join(here, name)
                    with open(path, errors='replace') as source:
                        code = source.read()
                    relative = os.path.relpath(path, root)
                    yield dict(sha256=hashlib.sha256(code.encode()).hexdigest(), code=code, score=None,
                               task=relative.split(os.sep)[0], source=relative)


def read_text(path):
    if path.endswith('.pdf'):
        return subprocess.run(['pdftotext', '-layout', path, '-'], capture_output=True, text=True, check=True).stdout
    with open(path, errors='replace') as source:
        return source.read()


def texts(roots):
    for root in roots:
        for here, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if not name.startswith('.')]
            for name in sorted(files):
                if not name.endswith(('.pdf', '.md', '.txt')):
                    continue
                path = os.path.join(here, name)
                body = read_text(path)
                fields = {'source': os.path.relpath(path, root), 'task': None}
                while (match := re.match(r'(source|task): (.*)\n', body)):
                    fields[match[1]] = match[2].strip()
                    body = body[match.end():]
                yield fields, body


def collect(args, work):
    with open(work + '/chunks.tmp', 'w') as out:
        def emit(collection, source, task, score, sha256, pieces):
            for number, piece in enumerate(pieces):
                out.write(json.dumps(dict(collection=collection, source=source, task=task, program_score=score,
                    sha256=sha256, chunk=number, text=piece), ensure_ascii=False) + '\n')
        seen = set()
        for program in list(run_programs(args.runs)) + list(loose_programs(args.program_files)):
            if program['sha256'] not in seen:
                seen.add(program['sha256'])
                pieces = chunks_of(elide_data(program['code']))[:MAX_PROGRAM_CHUNKS]
                emit('programs', program['source'], program['task'], program['score'], program['sha256'], pieces)
        for fields, body in texts(args.texts):
            emit('papers', fields['source'], fields['task'], None, hashlib.sha256(body.encode()).hexdigest(),
                 chunks_of(body))
    os.replace(work + '/chunks.tmp', work + '/chunks.jsonl')


def embed_key(row):
    return hashlib.sha256(f"{row['task']}\0{row['text']}".encode()).hexdigest()


def embed_text(row):
    return f"task: {row['task']}\n{row['text']}" if row['task'] else row['text']


def load_cache(work, dimension=1024):
    """Keys and vectors written so far; a torn tail from an interrupted build is cut back to whole records."""
    keys_path, vectors_path = work + '/keys.txt', work + '/vectors.f16'
    keys = open(keys_path).read().split() if os.path.exists(keys_path) else []
    count = os.path.getsize(vectors_path) // (2 * dimension) if os.path.exists(vectors_path) else 0
    count = min(count, len(keys))
    if keys or count:
        with open(vectors_path, 'r+b') as vectors:
            vectors.truncate(count * 2 * dimension)
        with open(keys_path, 'w') as out:
            out.writelines(key + '\n' for key in keys[:count])
    return keys[:count]


def embed(args, work):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=True)
    dtype = torch.float16 if args.device == 'mps' else torch.bfloat16
    model = AutoModel.from_pretrained(args.model, dtype=dtype, local_files_only=True).to(args.device).eval()
    done = set(load_cache(work))
    todo = {}
    with open(work + '/chunks.jsonl') as rows:
        for line in rows:
            row = json.loads(line)
            key = embed_key(row)
            if key not in done:
                todo[key] = embed_text(row)
    order = sorted(todo, key=lambda key: len(todo[key]))
    print(f'{len(done)} cached, {len(order)} to embed', flush=True)
    begun = time.time()
    with open(work + '/vectors.f16', 'ab') as vectors, open(work + '/keys.txt', 'a') as keys:
        for start in range(0, len(order), args.batch_size):
            batch_keys = order[start:start + args.batch_size]
            batch = tokenizer([todo[key] for key in batch_keys], padding=True, truncation=True,
                              max_length=MAX_TOKENS, return_tensors='pt').to(args.device)
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state[:, -1].float()
            normal = torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy().astype(np.float16)
            vectors.write(normal.tobytes())
            vectors.flush()
            keys.writelines(key + '\n' for key in batch_keys)
            keys.flush()
            if start % (args.batch_size * 20) == 0:
                rate = (start + len(batch_keys)) / (time.time() - begun)
                print(f'{start + len(batch_keys)}/{len(order)} {rate:.2f} chunks/s', flush=True)


README = '''Searchable archive, read-only. In a tool:
    import sys; sys.path.insert(0, '/data'); import search
    hits = search.search('your query', k=8, collection='papers')   # or 'programs', or None for both
Each hit has score (cosine), collection, source, task, program_score, chunk, text. search.document(hit['source'])
returns all stored chunks of that source. {load}
Collections: {counts}.
programs: best programs of earlier runs, tagged with task and score (scores compare only within a task); treat them
as ideas and code to adapt, and evaluate anything you reuse. papers: papers, problem pages and pull requests.
Credit what you use: put the hit's source (URL or PR number) in your tool's observations.
'''


def assemble(args, work):
    keys = load_cache(work)
    position = {key: number for number, key in enumerate(keys)}
    stored = np.memmap(work + '/vectors.f16', dtype=np.float16, mode='r').reshape(len(keys), -1)
    names = ['programs', 'papers']
    os.makedirs(args.output, exist_ok=True)
    picks, offsets, collection, source_ids, sources = [], [], [], [], {}
    chunk_counts, source_sets = dict.fromkeys(names, 0), {name: set() for name in names}
    with open(work + '/chunks.jsonl') as rows, open(work + '/meta.tmp', 'wb') as meta:
        for line in rows:
            row = json.loads(line)
            picks.append(position[embed_key(row)])
            offsets.append(meta.tell())
            meta.write(json.dumps(row, ensure_ascii=False).encode() + b'\n')
            collection.append(names.index(row['collection']))
            source_ids.append(sources.setdefault(row['source'], len(sources)))
            chunk_counts[row['collection']] += 1
            source_sets[row['collection']].add(row['source'])
    with open(work + '/index.tmp', 'wb') as out:
        np.savez(out, vectors=np.asarray(stored[picks]), collection=np.array(collection, np.uint8),
                 offsets=np.array(offsets, np.int64), source_ids=np.array(source_ids, np.int32),
                 collections=np.array(names))
    shutil.move(work + '/meta.tmp', args.output + '/meta.jsonl')
    shutil.move(work + '/index.tmp', args.output + '/index.npz')
    os.makedirs(args.output + '/model', exist_ok=True)
    for name in MODEL_FILES:
        target = f'{args.output}/model/{name}'
        if not os.path.exists(target):
            # copyfile follows the hub cache's symlinks: a data directory must hold only regular files.
            shutil.copyfile(os.path.join(args.model, name), target)
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive_search.py'), args.output + '/search.py')
    summary = ', '.join(f'{name} {len(source_sets[name])} sources in {chunk_counts[name]} chunks' for name in names)
    with open(args.output + '/README.txt', 'w') as out:
        out.write(README.format(load=args.load_note, counts=summary))
    for here, _, files in os.walk(args.output):
        for name in files:
            os.chmod(os.path.join(here, name), 0o644)
        os.chmod(here, 0o755)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='*', default=[])
    parser.add_argument('--program-files', nargs='*', default=[])
    parser.add_argument('--texts', nargs='*', default=[])
    parser.add_argument('--model', required=True, help='local snapshot directory of the embedding model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--work', help='cache directory; default OUTPUT.work, outside the digested directory')
    parser.add_argument('--stage', choices=('all', 'collect', 'embed', 'assemble'), default='all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--load-note', default='The first call in a tool loads the embedding model.')
    args = parser.parse_args()
    args.output = args.output.rstrip('/')
    work = args.work or args.output + '.work'
    os.makedirs(work, exist_ok=True)
    if args.stage in ('all', 'collect'):
        collect(args, work)
    if args.stage in ('all', 'embed'):
        embed(args, work)
    if args.stage in ('all', 'assemble'):
        assemble(args, work)


if __name__ == '__main__':
    main()
