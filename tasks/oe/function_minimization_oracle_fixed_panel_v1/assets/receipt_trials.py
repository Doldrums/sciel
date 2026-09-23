"""Pure trial receipt and public host-journal replay; never open private banks."""
import math
import hashlib
import json
import re
from . import oracle_runtime as runtime
from .panels import need, require_hash

STATUSES = {
    "ok",
    "startup_timeout",
    "infrastructure_error",
    "timeout",
    "candidate_error",
    "protocol_error",
}
INFRASTRUCTURE = {"startup_timeout", "infrastructure_error"}




def number(value, *, nonnegative=False):
    need(
        type(value) in (int, float)
        and math.isfinite(value)
        and (not nonnegative or value >= 0),
        "invalid finite receipt number",
    )
    return value


def trial(value, *, source_sha256, panel, position, binding):
    fields = {
        "schema",
        "status",
        "score",
        "queries",
        "query_budget",
        "best_point",
        "best_value",
        "source_sha256",
        "image_id",
        "container_name",
        "elapsed_seconds",
        "startup_seconds",
        "broker_cpu_seconds",
        "algorithm_seed",
        "bootstrap",
        "bootstrap_initialized",
        "finished",
        "objective_failed",
        "cleanup",
        "journal_sha256",
        "observations",
        "error",
        "stderr",
        "panel_position",
        "instance_index",
        "instance_sha256",
    }
    need(
        type(value) is dict
        and set(value) == fields
        and value["schema"] == "sciel-fixed-oracle-trial/1",
        "invalid trial receipt fields/schema",
    )
    need(
        value["source_sha256"] == source_sha256
        and value["image_id"] == binding["image_id"],
        "trial source/image differs",
    )
    need(
        type(value["panel_position"]) is int
        and value["panel_position"] == position
        and type(value["instance_index"]) is int
        and value["instance_index"] == panel["indices"][position]
        and value["instance_sha256"] == panel["instance_sha256"][position]
        and type(value["algorithm_seed"]) is int
        and value["algorithm_seed"] == panel["algorithm_seeds"][position]
        and value["bootstrap"] == runtime.BOOTSTRAP,
        "trial panel/order/seed/bootstrap differs",
    )
    require_hash(value["journal_sha256"])
    need(
        type(value["container_name"]) is str
        and re.fullmatch(r"sciel-oracle-[0-9a-f]{32}", value["container_name"]),
        "invalid owned container identity",
    )
    need(
        value["status"] in STATUSES
        and type(value["queries"]) is int
        and 0 <= value["queries"] <= 50
        and type(value["query_budget"]) is int
        and value["query_budget"] == 50,
        "invalid trial status/query budget",
    )
    for key in ("bootstrap_initialized", "finished", "objective_failed"):
        need(type(value[key]) is bool, "invalid trial state flag")
    need(value["error"] is None or type(value["error"]) is str, "invalid trial error")
    need(
        type(value["stderr"]) is str and len(value["stderr"].encode()) <= 262144,
        "invalid bounded stderr",
    )
    for field in ("elapsed_seconds", "broker_cpu_seconds"):
        number(value[field], nonnegative=True)
    if value["startup_seconds"] is not None:
        number(value["startup_seconds"], nonnegative=True)
        need(
            value["startup_seconds"] <= value["elapsed_seconds"],
            "startup exceeds trial wall time",
        )
    cleanup = value["cleanup"]
    need(
        type(cleanup) is dict
        and set(cleanup)
        == {
            "remove_returncode",
            "remove_error",
            "docker_client_returncode",
            "container_cpu_seconds",
            "container_cpu_scope",
        },
        "invalid cleanup evidence",
    )
    need(
        cleanup["container_cpu_seconds"] is None
        and cleanup["container_cpu_scope"] == "not observed",
        "container CPU measurement is not implemented",
    )
    for key in ("remove_returncode", "docker_client_returncode"):
        need(
            cleanup[key] is None or type(cleanup[key]) is int,
            "invalid observed process code",
        )
    need(
        cleanup["remove_error"] is None or type(cleanup["remove_error"]) is str,
        "invalid cleanup error",
    )
    rows = value["observations"]
    need(type(rows) is list and len(rows) <= 50, "invalid query transcript")
    best, point = None, None
    for index, row in enumerate(rows, 1):
        need(
            type(row) is dict
            and set(row) == {"query", "point", "value"}
            and type(row["query"]) is int
            and row["query"] == index,
            "query numbering differs",
        )
        need(
            type(row["point"]) is list and len(row["point"]) == 2, "invalid query point"
        )
        for coordinate in row["point"]:
            number(coordinate)
            need(-1 <= coordinate <= 1, "query outside declared domain")
        current = number(row["value"], nonnegative=True)
        if best is None or current < best:
            best, point = current, row["point"]
    need(
        value["queries"] == len(rows) + int(value["objective_failed"]),
        "charged query count differs",
    )
    if value["objective_failed"]:
        need(
            value["status"] == "infrastructure_error"
            and value["error"] == "trusted objective failed"
            and value["finished"] is False,
            "trusted objective failure must invalidate infrastructure",
        )
    need(
        value["best_value"] == best and value["best_point"] == point,
        "best queried point/value differs",
    )
    if best is not None:
        number(value["best_value"], nonnegative=True)
        need(
            type(value["best_point"]) is list and len(value["best_point"]) == 2,
            "invalid retained best point",
        )
        for coordinate in value["best_point"]:
            number(coordinate)
    need(
        value["bootstrap_initialized"] or not rows and value["queries"] == 0,
        "queries precede trusted seed initialization",
    )
    if value["status"] == "ok":
        need(
            value["bootstrap_initialized"]
            and value["finished"]
            and best is not None
            and not value["objective_failed"]
            and value["error"] is None,
            "invalid successful trial",
        )
    else:
        need(not value["finished"], "failed trial cannot claim successful finish")
    expected_score = 1 / (1 + best) if value["status"] == "ok" else 0.0
    number(value["score"], nonnegative=True)
    need(value["score"] == expected_score, "trial score arithmetic differs")
    return value


def replay_journal(raw, value):
    """Replay actual host frames, including a charged failed objective attempt.

    Observed objective values come from the host receipt, not from a surrogate.
    This checks the protocol/transcript connection without opening a private bank.
    """
    need(
        type(raw) is bytes
        and len(raw) <= 8 * 1024 * 1024
        and hashlib.sha256(raw).hexdigest() == value["journal_sha256"],
        "journal bytes/hash differ",
    )
    rows = [
        json.loads(
            line,
            object_pairs_hook=runtime._unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("nonfinite journal")
            ),
        )
        for line in raw.splitlines()
    ]
    need(
        len(rows) >= 2
        and rows[0]
        == dict(
            event="trial_intent",
            container_name=value["container_name"],
            image_id=value["image_id"],
            source_sha256=value["source_sha256"],
            algorithm_seed=value["algorithm_seed"],
            bootstrap=runtime.BOOTSTRAP,
            timeout=runtime.TRIAL_TIMEOUT,
            startup_timeout=runtime.STARTUP_TIMEOUT,
        ),
        "journal intent differs",
    )
    need(
        rows[-1]
        == dict(
            event="trial_terminal", status=value["status"], cleanup=value["cleanup"]
        ),
        "journal terminal differs",
    )
    observed = iter(value["observations"])

    def objective(x, y):
        row = next(observed, None)
        if row is None:
            need(
                value["objective_failed"],
                "journal contains an unrecorded objective attempt",
            )
            raise RuntimeError("recorded trusted objective failure")
        need(row["point"] == [x, y], "journal query point differs")
        return row["value"]

    session = runtime.OracleSession(objective, algorithm_seed=value["algorithm_seed"])
    phase, index, failed = "startup", 1, False
    while index < len(rows) - 1:
        item = rows[index]
        need(
            not failed
            and not session.finished
            and type(item) is dict
            and set(item) == {"event", "phase", "frame_hex"}
            and item["event"] == "received"
            and item["phase"] == phase
            and type(item["frame_hex"]) is str,
            "journal frame ordering differs",
        )
        frame = bytes.fromhex(item["frame_hex"])
        index += 1
        try:
            if phase == "startup":
                need(
                    runtime.decode_frame(frame)
                    == {"status": "ready", "protocol": runtime.PROTOCOL},
                    "invalid supervisor greeting",
                )
                phase = "trial"
                continue
            response = session.handle(frame)
        except (ValueError, runtime.ProtocolError):
            failed = True
            continue
        need(
            index < len(rows) - 1
            and rows[index] == dict(event="response", payload=response),
            "journal host response differs",
        )
        index += 1
    need(
        session.transcript == value["observations"]
        and session.queries == value["queries"]
        and session.initialized == value["bootstrap_initialized"]
        and session.finished == value["finished"]
        and session.objective_failed == value["objective_failed"],
        "journal state differs from trial receipt",
    )
    if failed:
        expected = (
            "infrastructure_error"
            if phase == "startup" or session.objective_failed
            else "protocol_error"
        )
        need(value["status"] == expected, "journal failure classification differs")
    need(
        (value["startup_seconds"] is not None) == (phase == "trial"),
        "journal readiness differs",
    )
    # The replayed host protocol, not receipt.error or candidate stderr, owns
    # this cause. Existing callers use the verification side effect only.
    return session.failure
