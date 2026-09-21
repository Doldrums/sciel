"""Shared journal hygiene and evaluated-source lookup for offline training."""

from __future__ import annotations

import json
import hashlib
import os
import math

PROVENANCE_SCHEMA = "sciel-checkpoint-provenance/1"


class DirtyRun(Exception):
    pass


def _counter(event, name, path):
    value = event.get(name)
    if value is not None and (type(value) is not int or value < 0):
        raise DirtyRun(f"{path}: invalid {name} in {event.get('event')}")
    return value


def _check_trajectory(events, path):
    """Conservative legacy path: counters cannot establish rollback ancestry."""
    seen_start, resumed = False, False
    previous_iteration = previous_evals = None
    children = set()
    for event in events:
        kind = event.get("event")
        if kind == "checkpoint_restore":
            raise DirtyRun(f"{path}: unsupported checkpoint_restore provenance schema")
        if kind == "run_start":
            resumed = seen_start
            seen_start = True
        elif kind == "iteration":
            iteration = _counter(event, "iteration_no", path)
            evals = _counter(event, "B_evals", path)
            if resumed:
                if previous_iteration is None or iteration is None:
                    raise DirtyRun(f"{path}: resumed segment has no checkpoint provenance")
                if iteration != previous_iteration + 1:
                    raise DirtyRun(f"{path}: rollback or gap at resumed segment; "
                                   "checkpoint branch is unknown")
                resumed = False
            if iteration is not None and previous_iteration is not None \
                    and iteration <= previous_iteration:
                raise DirtyRun(f"{path}: rollback or repeated iteration counter")
            if evals is not None and previous_evals is not None and evals < previous_evals:
                raise DirtyRun(f"{path}: rollback of evaluation counter")
            previous_iteration, previous_evals = iteration, evals
            child = event.get("child_id")
            if child is not None:
                if not isinstance(child, str) or not child or child in children:
                    raise DirtyRun(f"{path}: repeated or invalid child_id; ancestry is ambiguous")
                children.add(child)
        elif kind == "run_end":
            if resumed:
                raise DirtyRun(f"{path}: resumed segment has no iteration provenance")
            iteration = _counter(event, "iteration_no", path)
            if iteration is not None and previous_iteration is not None \
                    and iteration != previous_iteration:
                raise DirtyRun(f"{path}: run_end disagrees with trajectory iteration counter")


def _read(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    events, offsets, offset = [], [], 0
    try:
        for line in raw.splitlines(keepends=True):
            offset += len(line)
            if line.strip():
                events.append(json.loads(line))
                offsets.append(offset)
        if not all(isinstance(event, dict) for event in events):
            raise ValueError("journal records must be objects")
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise DirtyRun(f"{path}: malformed journal: {exc}") from exc
    return raw, events, offsets


def _selected_v2(raw, events, offsets, last, path):
    """Resolve explicit checkpoint edges; never use last-wins by child/counter."""
    def require(ok, reason):
        if not ok:
            raise DirtyRun(f"{path}: {reason}")

    segments, event_segment = {}, {}
    segment = None
    for i, event in enumerate(events):
        require(i == 0 or events[i - 1].get("event") != "run_end"
                or event.get("event") == "run_start", "events follow terminal run_end within segment")
        if event.get("event") == "run_start":
            require(event.get("schema_version") == 2, "mixed legacy/unbound journal ancestry")
            segment = event.get("segment_id")
            require(isinstance(segment, str) and segment and segment not in segments,
                    "missing or repeated segment_id")
            require(isinstance(event.get("run_id"), str) and event["run_id"], "missing run_id")
            segments[segment] = i
        require(segment is not None and event.get("segment_id") == segment,
                "event has no matching segment provenance")
        start = segments[segment]
        require(type(event.get("event_seq")) is int and event["event_seq"] == i - start + 1,
                "missing, repeated or out-of-order event_seq")
        require(event.get("run_id") == events[start]["run_id"], "run_id changed within segment")
        if event.get("event") == "checkpoint_restore":
            require(i == start + 1 and events[start].get("resumed") is True,
                    "unexpected checkpoint_restore outside segment start")
        event_segment[i] = segment

    def anchor_index(anchor):
        require(isinstance(anchor, dict), "checkpoint ancestry is unbound")
        size = anchor.get("byte_offset")
        require(type(size) is int and size in offsets, "checkpoint anchor is not an event boundary")
        i = offsets.index(size)
        require(raw[:size].endswith(b"\n") and hashlib.sha256(raw[:size]).hexdigest()
                == anchor.get("prefix_sha256"), "checkpoint journal prefix hash mismatch")
        require((events[i].get("segment_id"), events[i].get("event_seq"))
                == (anchor.get("segment_id"), anchor.get("event_seq")), "checkpoint anchor identity mismatch")
        return i

    def declaration(event):
        provenance = event.get("provenance")
        require(isinstance(provenance, dict) and provenance.get("schema") == PROVENANCE_SCHEMA,
                "unsupported checkpoint provenance schema")
        require(provenance.get("checkpoint_id") == event.get("checkpoint_id")
                and isinstance(event.get("checkpoint_id"), str), "checkpoint ID mismatch")
        require(isinstance(event.get("checkpoint_sha256"), str)
                and len(event["checkpoint_sha256"]) == 64, "missing checkpoint file hash")
        require(provenance.get("run_id") == event.get("run_id"), "checkpoint run_id mismatch")
        counters = provenance.get("counters")
        require(isinstance(counters, dict), "missing checkpoint counters")
        for name in ("iteration_no", "B_evals"):
            require(_counter(counters, name, path) is not None and counters[name] == event.get(name),
                    "checkpoint declaration counter mismatch")
        return provenance, anchor_index(provenance.get("journal_anchor"))

    seen_checkpoints = {}
    for i, event in enumerate(events):
        if event.get("event") == "checkpoint":
            provenance, anchor = declaration(event)
            require(anchor < i and event_segment[anchor] == event_segment[i], "checkpoint points outside its segment")
            ident = event["checkpoint_id"]
            require(ident not in seen_checkpoints, "repeated checkpoint_id")
            seen_checkpoints[ident] = i

    def branch(last_index):
        start = segments[event_segment[last_index]]
        head = events[start]
        prefix = []
        if head.get("resumed") is True:
            require(last_index > start and events[start + 1].get("event") == "checkpoint_restore",
                    "resumed segment has no checkpoint_restore")
            restore = events[start + 1]
            provenance = restore.get("provenance")
            require(restore.get("ancestry") == "bound" and isinstance(provenance, dict),
                    "checkpoint restore has legacy/unbound ancestry")
            require(not restore.get("identity_mismatches"), "checkpoint restore changed objective/descriptor identity")
            require(isinstance(restore.get("checkpoint_id"), str), "invalid restored checkpoint_id")
            source = seen_checkpoints.get(restore.get("checkpoint_id"))
            require(source is not None and source < start, "restored checkpoint declaration is missing")
            saved = events[source]
            require(restore.get("checkpoint_sha256") == saved.get("checkpoint_sha256")
                    and provenance == saved.get("provenance"), "restored checkpoint hash/provenance mismatch")
            require(head["run_id"] == saved["run_id"], "restored checkpoint belongs to another run")
            require(restore.get("restored_counters") == provenance.get("counters")
                    == head.get("counters"), "restored counters disagree with checkpoint/run_start")
            prefix = branch(anchor_index(provenance.get("journal_anchor")))
        else:
            require(head.get("resumed") is False and start == 0, "fresh segment has unbound earlier ancestry")
        return prefix + list(range(start, last_index + 1))

    indices = branch(last)
    selected = [events[i] for i in indices]
    children, previous_iteration, previous_evals = set(), 0, 0
    for original_index, event in zip(indices, selected):
        kind = event.get("event")
        if kind == "run_start":
            counters = event.get("counters")
            require(isinstance(counters, dict), "invalid run_start counters shape")
            require(_counter(counters, "iteration_no", path) == previous_iteration
                    and _counter(counters, "B_evals", path) == previous_evals,
                    "segment counters disagree with selected prefix")
        elif kind == "iteration":
            iteration, evals = _counter(event, "iteration_no", path), _counter(event, "B_evals", path)
            require(iteration == previous_iteration + 1, "gap/repeated iteration in selected branch")
            child = event.get("child_id")
            measured = False
            if event.get('reason') == 'remeasurement':
                require(child is None and type(event.get('measurement_ok')) is bool,
                        'remeasurement must be separate from a child mutation')
                measured = event['measurement_ok']
                ident, score = event.get('measurement_program_id'), event.get('measurement_score')
                if measured:
                    require(isinstance(ident,str) and ident and ident not in children
                            and type(score) in (int,float) and math.isfinite(score),
                            'successful remeasurement lacks a new measurement identity/finite score')
                    children.add(ident)
                else:
                    require(ident is None and score is None, 'failed remeasurement has a successful measurement identity')
            require(evals == previous_evals + (child is not None or measured), "evaluation counter disagrees with selected branch")
            if child is not None:
                require(isinstance(child, str) and child and child not in children,
                        "repeated or invalid child_id in selected branch")
                children.add(child)
            previous_iteration, previous_evals = iteration, evals
        elif kind == "run_end":
            require(event.get("iteration_no") == previous_iteration and event.get("B_evals") == previous_evals,
                    "run_end counters disagree with selected branch")
            require(isinstance(event.get("checkpoint_id"), str), "invalid final checkpoint_id")
            source = seen_checkpoints.get(event.get("checkpoint_id"))
            require(source in indices and source < original_index
                    and events[source].get("segment_id") == event.get("segment_id")
                    and events[source].get("checkpoint_sha256") == event.get("checkpoint_sha256"),
                    "run_end has no matching checkpoint identity")
            require(events[source].get("iteration_no") == previous_iteration
                    and events[source].get("B_evals") == previous_evals,
                    "run_end checkpoint counters disagree")
    return selected


def trajectory_events(path, require_complete=True):
    """Read a completed trajectory, including verified sequential resume segments.

    The latest end must close the latest segment. An earlier paused/completed end
    cannot supply a final target for a subsequently resumed, unfinished segment.
    Old journals without a reason field remain readable.
    """
    raw, events, offsets = _read(path)
    if any(e.get("schema_version") not in (None, 1, 2) for e in events if e.get("event") == "run_start"):
        raise DirtyRun(f"{path}: unsupported journal provenance schema")
    ends = [i for i, e in enumerate(events) if e.get("event") == "run_end"]
    if not ends and require_complete:
        raise DirtyRun(f"{path}: no run_end (killed run) -- final value unknown")
    last = ends[-1] if require_complete else len(events) - 1
    end = events[last] if last >= 0 and events[last].get("event") == "run_end" else None
    if require_complete and end.get("reason") not in (None, "budget", "backstop", "saturated"):
        raise DirtyRun(f"{path}: latest segment ended with {end.get('reason')!r}")
    if require_complete and any(e.get("event") in ("run_start", "iteration", "checkpoint_restore")
           for e in events[last + 1:]):
        raise DirtyRun(f"{path}: resumed segment has no run_end -- final value unknown")
    if any(e.get("schema_version") == 2 for e in events if e.get("event") == "run_start"):
        if not raw.endswith(b"\n"):
            raise DirtyRun(f"{path}: incomplete v2 journal tail")
        return _selected_v2(raw, events, offsets, last, path), end
    if any(e.get("segment_id") is not None for e in events):
        raise DirtyRun(f"{path}: segment provenance lacks a supported run_start schema")
    _check_trajectory(events[:last + 1], path)
    return events[:last + 1], end


def completed_events(path):
    events, end = trajectory_events(path, require_complete=True)
    if end.get("segment_id") is not None:
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(path)), "checkpoint.json")
        try:
            with open(checkpoint_path, "rb") as source:
                data = source.read()
        except OSError as exc:
            raise DirtyRun(f"{path}: missing bound final checkpoint") from exc
        verify_checkpoint_bytes(data, end, checkpoint_path)
        records=json.loads(data)['db']['programs']
        if any(type(row) is not dict or type(row.get('id')) is not str or not row['id'] for row in records):
            raise DirtyRun(f'{path}: invalid checkpoint program identities')
        programs = {row['id']:row for row in records}
        if len(programs)!=len(records):
            raise DirtyRun(f'{path}: repeated checkpoint program identity')
        for event in events:
            if event.get('reason')=='remeasurement' and event.get('measurement_ok') is True:
                program=programs.get(event.get('measurement_program_id'))
                if (not program or program.get('origin')!='remeasurement'
                        or program.get('score')!=event.get('measurement_score')
                        or program.get('island')!=event.get('island')):
                    raise DirtyRun(f'{path}: remeasurement is not bound to its checkpoint program')
    return events, end


def verify_checkpoint_bytes(data, end, path):
    """Bind source lookup to a v2 selected end; legacy clean inputs stay readable."""
    if end and end.get("segment_id") is not None:
        try:
            state = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise DirtyRun(f"{path}: malformed checkpoint JSON") from exc
        if not isinstance(state, dict) or not isinstance(state.get("provenance"), dict):
            raise DirtyRun(f"{path}: invalid checkpoint provenance shape")
        if state["provenance"].get("schema") != PROVENANCE_SCHEMA:
            raise DirtyRun(f"{path}: unsupported checkpoint provenance schema")
        if (hashlib.sha256(data).hexdigest() != end.get("checkpoint_sha256")
                or (state.get("provenance") or {}).get("checkpoint_id") != end.get("checkpoint_id")):
            raise DirtyRun(f"{path}: final checkpoint is not bound to the selected trajectory")
        db = state.get("db")
        if not isinstance(db, dict) or not isinstance(db.get("programs"), list):
            raise DirtyRun(f"{path}: invalid final checkpoint database")
        best = [p for p in db["programs"] if isinstance(p, dict) and p.get("id") == db.get("pop_best_id")]
        if len(best) != 1 or best[0].get("score") != end.get("pop_best"):
            raise DirtyRun(f"{path}: run_end pop_best differs from bound checkpoint")
        for key in ("iteration_no", "B_evals", "cost_total", "n_events", "n_dups", "n_eval_fail", "n_nochild"):
            if key in end and state.get(key) != end[key]:
                raise DirtyRun(f"{path}: run_end {key} differs from bound checkpoint")


def checkpoint_programs(journal_path, *, end=None):
    """Final evaluated programs for historical journals that lack child_code."""
    path = os.path.join(os.path.dirname(os.path.abspath(journal_path)), "checkpoint.json")
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        if end and end.get('segment_id') is not None:
            raise DirtyRun(f'{path}: missing bound final checkpoint') from exc
        return {}
    try:
        state = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DirtyRun(f"{path}: malformed checkpoint JSON") from exc
    if not isinstance(state, dict) or not isinstance(state.get("db", {}), dict):
        raise DirtyRun(f"{path}: invalid checkpoint state/db shape")
    programs = state.get("db", {}).get("programs", [])
    if not isinstance(programs, list):
        raise DirtyRun(f"{path}: checkpoint programs must be an array")
    indexed = {}
    for program in programs:
        if (not isinstance(program, dict) or not isinstance(program.get("id"), str)
                or not program["id"] or program["id"] in indexed):
            raise DirtyRun(f"{path}: invalid/repeated checkpoint program ID")
        indexed[program["id"]] = program
    if end is None:
        _, end = completed_events(journal_path)
    verify_checkpoint_bytes(data, end, path)
    return indexed


def evaluated_code(event, programs):
    """Never substitute an LLM response for the source the evaluator received."""
    code = event.get("child_code")
    if isinstance(code, str) and code.strip():
        return code
    code = (programs.get(event.get("child_id")) or {}).get("code")
    return code if isinstance(code, str) and code.strip() else None
