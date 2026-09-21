"""Portable CLI paths without retired repository artifacts or actual work."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("explicit", [False, True])
def test_benchmark_writes_outside_checkout_and_creates_parent(tmp_path, monkeypatch, explicit):
    module = script("benchmark_parallel_search")
    monkeypatch.chdir(tmp_path)
    calls = []
    result = {"schema": "fixture", "rows": []}
    monkeypatch.setattr(module, "benchmark", lambda root, count: calls.append(count) or result)
    output = tmp_path / "external" / "nested" / "benchmark.json"
    if explicit:
        args = ["--output", str(output)]
    else:
        monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_: str(output.parent))
        output = output.parent / "parallel_execution.json"
        args = []
    assert module.main(["--calls", "4", *args]) == 0
    assert calls == [4] and json.loads(output.read_text()) == result
    assert not (tmp_path / "reports").exists()


def test_corpus_cli_requires_explicit_revision_before_collecting(tmp_path, monkeypatch):
    module = script("prepare_review_corpus")
    calls = []
    monkeypatch.setattr(module, "prepare", lambda *args: calls.append(args))
    args = ["prepare", str(tmp_path / "runs"), "--out", str(tmp_path / "corpus")]
    with pytest.raises(SystemExit) as rejected:
        module.main(args)
    assert rejected.value.code == 2 and calls == []
    module.main([*args, "--revision", "HEAD"])
    assert calls == [(str(tmp_path / "corpus"), [str(tmp_path / "runs")], "HEAD", 400, 0)]


@pytest.mark.parametrize("name, arguments, expected", [
    ("train_review_models", ["--corpus", "data.jsonl", "--corpus-manifest", "manifest.json"],
        {"out": "sciel-data/review/models", "baseline": "sciel-artifacts/code_embed_v3.pt"}),
    ("audit_review_record_metrics", [],
        {"out": "sciel-data/review/record_metrics", "corpus": "sciel-data/review/corpus"}),
])
def test_review_cli_defaults_are_external_and_independent_of_cwd(tmp_path, monkeypatch, name, arguments, expected):
    if name == "train_review_models":
        pytest.importorskip("torch")
    module = script(name)
    monkeypatch.chdir(tmp_path)
    captured = {}
    parse = module.argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def capture(parser, args):
        captured.update(vars(parse(parser, args)))
        raise Parsed

    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(Parsed):
        module.main(arguments)
    assert {key: captured[key] for key in expected} == {
        key: SCRIPTS.parent.parent / relative for key, relative in expected.items()}
    assert not list(tmp_path.iterdir())
