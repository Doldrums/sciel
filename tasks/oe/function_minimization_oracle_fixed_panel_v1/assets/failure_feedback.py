"""Host-only, untrusted exception hints; never score or protocol evidence.

Candidate stderr can impersonate a traceback. Only fixed vocabulary and an
identifier present on the exact source line may reach the next prompt. Raw
messages, traceback lines, paths and values remain in the retained trial only.
"""

import ast
import hashlib
import re

from . import oracle_runtime as runtime
from .panels import need


CLASSES = frozenset(
    {
        "AttributeError",
        "NameError",
        "UnboundLocalError",
        "TypeError",
        "ValueError",
        "IndexError",
        "KeyError",
        "ZeroDivisionError",
        "ImportError",
        "ModuleNotFoundError",
        "AssertionError",
        "RuntimeError",
        "OverflowError",
        "RecursionError",
        "MemoryError",
        "SyntaxError",
        "IndentationError",
        "TabError",
    }
)
FRAME = re.compile(
    r'  File "/input/candidate[.]py", line ([1-9][0-9]{0,5})(?:, in [^\r\n]*)?'
)
IDENTIFIER = r"([A-Za-z_][A-Za-z_0-9]{0,63})"


def _hint(source, stderr):
    if type(stderr) is not str or not stderr or len(stderr.encode()) > 262144:
        return None
    lines = stderr.splitlines()
    terminal = re.fullmatch(r"([A-Za-z]+)(?:: (.*))?", lines[-1])
    if terminal is None or terminal[1] not in CLASSES:
        return None
    # An exact internal candidate frame is required. Never expose other frames.
    frames = [FRAME.fullmatch(line) for line in lines[:-1]]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return None
    number = int(frames[-1][1])
    try:
        text = source.decode("utf-8")
        if number > len(text.splitlines()):
            return None
        tree = ast.parse(text)
    except (UnicodeError, SyntaxError, RecursionError):
        return None
    nodes = [node for node in ast.walk(tree) if getattr(node, "lineno", None) == number]
    # A blank/comment-only alleged frame cannot bind an executable source line.
    if not nodes:
        return None
    fields = [terminal[1], f"candidate_line={number}"]
    message = terminal[2] or ""
    if (
        terminal[1] == "AttributeError"
        and message == "'float' object has no attribute 'append'"
        and any(
            isinstance(node, ast.Attribute) and node.attr == "append" for node in nodes
        )
    ):
        fields += ["builtin_type=float", "missing_attribute=append"]
    elif terminal[1] in {"NameError", "UnboundLocalError"}:
        patterns = (
            (
                r"name '" + IDENTIFIER + r"' is not defined",
                r"cannot access free variable '"
                + IDENTIFIER
                + r"' where it is not associated with a value in enclosing scope",
            )
            if terminal[1] == "NameError"
            else (
                r"cannot access local variable '"
                + IDENTIFIER
                + r"' where it is not associated with a value",
            )
        )
        match = next((m for p in patterns if (m := re.fullmatch(p, message))), None)
        if match and any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == match[1]
            for node in nodes
        ):
            fields.append("source_identifier=" + match[1])
    return "candidate_stderr_untrusted: " + "; ".join(fields) + "."


def append_hint(report, source, artifacts):
    """Append at most one bounded hint to an already verified failed summary.

    The caller obtains artifacts from receipts.failure_artifacts first. This helper
    does not replace that host replay or derive any trusted cause from stderr.
    """
    if report["result"]["ok"] is not False:
        return artifacts
    need(type(source) is bytes, "invalid feedback source")
    sha = hashlib.sha256(source).hexdigest()
    need(report["source_sha256"] == sha, "feedback source hash differs")
    need(
        all(t["source_sha256"] == sha for t in report["trials"]),
        "feedback trial source differs",
    )
    need(
        artifacts.get("error_type") == "OracleTrialFailure"
        and type(artifacts.get("error")) is str
        and len(artifacts["error"]) <= 600,
        "feedback requires bounded host failure summary",
    )
    # The evaluator already enforces this same bound before dispatch. A future
    # unsupported optional projection must not replace its failed result.
    if not 0 < len(source) <= runtime.MAX_SOURCE_BYTES:
        return artifacts
    for trial in report["trials"]:
        if trial["status"] != "candidate_error":
            continue
        hint = _hint(source, trial["stderr"])
        if hint and len(artifacts["error"]) + 1 + len(hint) <= 600:
            return {**artifacts, "error": artifacts["error"] + " " + hint}
    return artifacts
