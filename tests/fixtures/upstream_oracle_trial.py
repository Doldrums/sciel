"""Synthetic public protocol leaf copied from trusted3c fixture; no container/model."""
import hashlib

def synthetic_trial(
    E, position=0, *, status="ok", source, name, selected, queries=1, objective_value=1.0
):
    P, R = E.panels, E.runtime
    BINDING = {"image_id": "sha256:" + "a" * 64}
    session = R.OracleSession(
        lambda x, y: objective_value, algorithm_seed=selected["algorithm_seeds"][position]
    )
    cleanup = dict(
        remove_returncode=0,
        remove_error=None,
        docker_client_returncode=-9,
        container_cpu_seconds=None,
        container_cpu_scope="not observed",
    )
    rows = [
        dict(
            event="trial_intent",
            container_name=name,
            image_id=BINDING["image_id"],
            source_sha256=hashlib.sha256(source).hexdigest(),
            algorithm_seed=session.algorithm_seed,
            bootstrap=R.BOOTSTRAP,
            timeout=R.TRIAL_TIMEOUT,
            startup_timeout=R.STARTUP_TIMEOUT,
        )
    ]

    def receive(frame, phase="trial"):
        rows.append(
            dict(event="received", phase=phase, frame_hex=P.canonical(frame).hex())
        )
        if phase == "trial":
            try:
                response = session.handle(P.canonical(frame))
            except R.ProtocolError:
                return
            rows.append(dict(event="response", payload=response))

    if status != "startup_timeout":
        receive(dict(status="ready", protocol=R.PROTOCOL), "startup")
        receive(
            dict(
                op="initialized",
                bootstrap=R.BOOTSTRAP,
                algorithm_seed=session.algorithm_seed,
            )
        )
        for _ in range(queries):
            receive(dict(op="query", point=[0.0, 0.0]))
        if status == "ok":
            receive(dict(op="finish"))
        elif status == "infrastructure_error":
            session.objective = lambda x, y: (_ for _ in ()).throw(
                RuntimeError("synthetic")
            )
            receive(dict(op="query", point=[0.1, 0.2]))
        elif status == "protocol_error":
            receive(dict(op="reset"))
    observations = session.transcript
    rows.append(dict(event="trial_terminal", status=status, cleanup=cleanup))
    raw = b"".join(P.canonical(row) + b"\n" for row in rows)
    value = dict(
        schema="sciel-fixed-oracle-trial/1",
        status=status,
        score=1 / (1 + objective_value) if status == "ok" else 0.0,
        queries=session.queries,
        query_budget=50,
        best_point=[0.0, 0.0] if observations else None,
        best_value=objective_value if observations else None,
        source_sha256=hashlib.sha256(source).hexdigest(),
        image_id=BINDING["image_id"],
        container_name=name,
        elapsed_seconds=0.1,
        startup_seconds=0.01 if status != "startup_timeout" else None,
        broker_cpu_seconds=0.02,
        algorithm_seed=session.algorithm_seed,
        bootstrap=R.BOOTSTRAP,
        bootstrap_initialized=session.initialized,
        finished=session.finished,
        objective_failed=session.objective_failed,
        cleanup=cleanup,
        journal_sha256=hashlib.sha256(raw).hexdigest(),
        observations=observations,
        error=session.failure,
        stderr="",
        panel_position=position,
        instance_index=selected["indices"][position],
        instance_sha256=selected["instance_sha256"][position],
    )
    return value, raw
