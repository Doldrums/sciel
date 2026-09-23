"""Read-only host stage evidence authentication; no private objective access."""
import hashlib
import json
from .facade import EvalResult

def files(directory):
    from .oracle_evidence import ordinary, need
    result = {}
    for path in sorted(ordinary(directory).rglob("*")):
        ordinary(path)
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(directory).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        else:
            need(path.is_dir(), "oracle evidence contains a special file")
    return result


def inspect_stage(directory, invocation, result):
    from .oracle_evidence import task_module, canonical, need, digest
    module = task_module()
    original = result.to_json()
    state, report_hash, error = "incomplete", None, None
    if (directory / "raw/evaluation.json").exists():
        try:
            ref = json.loads(result.artifacts["oracle_evaluation"])
            envelope = (
                json.loads(result.artifacts["evaluation_receipt"])
                if invocation["stage"] != "smoke"
                else None
            )
            report = module.receipts.verify_retained(
                directory / "raw", ref, envelope=envelope, **invocation["expected"]
            )
            need(
                canonical(report["result"])
                == canonical({key: original[key] for key in ("ok", "score", "cases")}),
                "runner result differs from retained oracle evaluation",
            )
            state, report_hash = "complete", report["sha256"]
        except (KeyError, ValueError, TypeError, OSError) as exc:
            state, error = "invalid", str(exc)
    elif result.ok:
        error = "successful oracle result has no retained report"
    if error is not None:
        result = EvalResult(
            ok=False,
            score=0.0,
            artifacts={"error_type": "OracleEvidenceInvalid", "error": error},
            timing=result.timing,
            cost=result.cost,
        )
    entry = dict(
        invocation=invocation,
        invocation_sha256=digest(invocation),
        state=state,
        report_sha256=report_hash,
        runner_result=original,
        error=error,
        files=files(directory),
    )
    return result, entry
