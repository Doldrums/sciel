"""Trusted host protocol for the opt-in hidden-oracle task.

The candidate and its client are untrusted. Only this process evaluates the
objective, owns the query count and selects the best observed point. Resource
isolation alone is insufficient: production trials require the container backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import selectors
import struct
import subprocess
import time
import uuid


PROTOCOL = "sciel-hidden-oracle/1"
QUERY_BUDGET = 50
MAX_FRAME_BYTES = 4096
MAX_SOURCE_BYTES = 262144
STARTUP_TIMEOUT = 10.0
TRIAL_TIMEOUT = 5.0
BASE_IMAGE = "python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff"


class ProtocolError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate JSON key")
        result[key] = value
    return result


def decode_frame(data: bytes) -> dict:
    if not data or len(data) > MAX_FRAME_BYTES:
        raise ProtocolError("invalid frame length")
    try:
        value = json.loads(data, object_pairs_hook=_unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ProtocolError("nonfinite JSON number")))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("invalid JSON frame") from exc
    if not isinstance(value, dict):
        raise ProtocolError("frame must be an object")
    return value


def _coordinate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("coordinate must be a number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ProtocolError("invalid coordinate") from exc
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ProtocolError("coordinate outside finite domain")
    return value


@dataclass
class OracleSession:
    """A single non-resettable trial, never shared with candidate memory.

    Repeated coordinates consume repeated queries. Malformed input, excess
    requests and objective failures latch failure; there is no recovery opcode.
    A finish request performs no evaluation and cannot override the best point.
    """

    objective: object = field(repr=False)
    queries: int = 0
    best_point: tuple[float, float] | None = None
    best_value: float | None = None
    finished: bool = False
    failed: bool = False
    failure: str | None = None
    transcript: list = field(default_factory=list)

    def hello(self):
        return {"protocol": PROTOCOL, "bounds": [-1.0, 1.0],
                "dimensions": 2, "query_budget": QUERY_BUDGET}

    def handle(self, data: bytes) -> dict:
        if self.finished or self.failed:
            raise ProtocolError("trial already closed")
        try:
            frame = decode_frame(data)
            if frame.get("op") == "finish":
                if set(frame) != {"op"} or not self.queries:
                    raise ProtocolError("invalid finish")
                self.finished = True
                return {"status": "finished", "queries": self.queries}
            if set(frame) != {"op", "point"} or frame["op"] != "query":
                raise ProtocolError("unsupported request")
            point = frame["point"]
            if not isinstance(point, list) or len(point) != 2:
                raise ProtocolError("point must have two coordinates")
            x, y = map(_coordinate, point)
            if self.queries >= QUERY_BUDGET:
                raise ProtocolError("query budget exceeded")
            # Charge before computation, even if the objective itself fails.
            self.queries += 1
            try:
                value = float(self.objective(x, y))
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid trusted objective value")
            except Exception as exc:
                # Do not expose the objective's repr, arguments or traceback.
                raise ProtocolError("trusted objective failed") from exc
            if self.best_value is None or value < self.best_value:
                self.best_point, self.best_value = (x, y), value
            self.transcript.append({"query": self.queries, "point": [x, y], "value": value})
            return {"status": "ok", "query": self.queries, "value": value,
                    "remaining": QUERY_BUDGET - self.queries}
        except ProtocolError as exc:
            self.failed, self.failure = True, str(exc)
            raise


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def instance_from_seed(seed: bytes):
    """Public family; private parameters. The unique analytic minimum is zero."""
    rng = random.Random(int.from_bytes(seed, "big"))
    return {"centre": [rng.uniform(-.75, .75) for _ in range(2)],
            "angle": rng.uniform(0, 2*math.pi),
            "curvature": [rng.uniform(.2, 1.5) for _ in range(2)],
            "waves": [{"weight": rng.uniform(.1, .7),
                       "frequency": [rng.uniform(-8., 8.) for _ in range(2)]}
                      for _ in range(4)]}


def objective_for(instance):
    # Validate private manifests too: malformed trusted input is an infra failure.
    if set(instance) != {"centre", "angle", "curvature", "waves"}:
        raise ValueError("invalid oracle instance schema")
    c = instance["centre"]
    q = instance["curvature"]
    waves = instance["waves"]
    if len(c) != 2 or len(q) != 2 or len(waves) != 4:
        raise ValueError("invalid oracle instance shape")
    values = [*c, instance["angle"], *q]
    for wave in waves:
        if set(wave) != {"weight", "frequency"} or len(wave["frequency"]) != 2:
            raise ValueError("invalid wave")
        values += [wave["weight"], *wave["frequency"]]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in values):
        raise ValueError("invalid oracle parameter")
    if (any(not -.75 <= v <= .75 for v in c) or any(not .2 <= v <= 1.5 for v in q)
            or not 0 <= instance["angle"] <= 2*math.pi
            or any(not .1 <= w["weight"] <= .7 or any(not -8 <= z <= 8 for z in w["frequency"])
                   for w in waves)):
        raise ValueError("oracle parameter outside declared family")
    ca, sa = math.cos(instance["angle"]), math.sin(instance["angle"])

    def objective(x, y):
        dx, dy = x-c[0], y-c[1]
        u, v = ca*dx-sa*dy, sa*dx+ca*dy
        return q[0]*u*u + q[1]*v*v + sum(
            w["weight"]*math.sin(w["frequency"][0]*dx+w["frequency"][1]*dy)**2
            for w in waves)
    return objective


def create_instances(n, *, role="development", private_dir=None, commitment_dir=None):
    """Commit a fresh private batch before execution; never derive it from public RNG."""
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100:
        raise ValueError("trial count must be an integer between 1 and 100")
    if role not in {"development", "sealed_holdout"}:
        raise ValueError("invalid instance role")
    private_dir = Path(private_dir or Path.home()/".local/share/sciel_private/function_minimization_oracle")
    commitment_dir = Path(commitment_dir or Path.home()/".local/share/sciel_oracle_commitments")
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    commitment_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    # Existing directories must not make secrets readable by another local user.
    if private_dir.stat().st_mode & 0o077:
        raise ValueError("private instance directory must have mode 0700")
    seed = secrets.token_bytes(32)
    ident = uuid.uuid4().hex
    body = {"schema": PROTOCOL, "role": role, "batch": ident,
            "seed": seed.hex(), "instances": [instance_from_seed(hashlib.sha256(
                seed + i.to_bytes(8, "big")).digest()) for i in range(n)]}
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    path = private_dir/(ident+".json")
    _exclusive_json(path, body, 0o600)
    commitment = {"schema": PROTOCOL, "batch": ident, "role": role,
                  "instance_count": n, "commitment_sha256": digest,
                  "created_at_unix": time.time()}
    commit_path = commitment_dir/(ident+".json")
    _exclusive_json(commit_path, commitment, 0o644)
    return body["instances"], {**commitment, "commitment_path": str(commit_path)}


def _exclusive_json(path, body, mode):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_canonical(body)+b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_sealed_instances(path, expected_commitment, *, n):
    """Explicit holdout path and commitment only; never a development fallback."""
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("private instance manifest must have mode 0600")
    body = json.loads(path.read_bytes())
    if (hashlib.sha256(_canonical(body)).hexdigest() != expected_commitment
            or body.get("schema") != PROTOCOL or body.get("role") != "sealed_holdout"):
        raise ValueError("sealed instance commitment or role mismatch")
    if len(body.get("instances", [])) != n:
        raise ValueError("sealed instance count differs from requested trials")
    for instance in body["instances"]:
        objective_for(instance)
    return body["instances"], {"schema": PROTOCOL, "role": "sealed_holdout",
            "batch": body["batch"], "instance_count": n,
            "commitment_sha256": expected_commitment}


def docker_argv(image, name, timeout=TRIAL_TIMEOUT):
    if not (image.startswith("sha256:") and len(image)==71):
        raise ValueError("runtime requires an exact local image ID")
    if not name.startswith("sciel-oracle-"):
        raise ValueError("invalid owned container name")
    return ["docker", "run", "--rm", "-i", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=KILL",
            "--security-opt=no-new-privileges", "--pids-limit=64",
            "--memory=256m", "--memory-swap=256m", "--cpus=1",
            "--tmpfs=/input:rw,nosuid,nodev,noexec,size=2m,mode=0755",
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777",
            image, str(timeout)]


def _docker_env():
    # CLI needs its ordinary context, but the container inherits none of this.
    return {k:v for k,v in os.environ.items() if k in
            {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "DOCKER_CONFIG", "DOCKER_CONTEXT"}}


def runtime_assets_sha256():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("Dockerfile", "container_supervisor.py", "oracle_client.py", "candidate_runner.py"):
        digest.update(name.encode()+b"\0"+root.joinpath(name).read_bytes()+b"\0")
    return digest.hexdigest()


def verify_runtime_image(image):
    docker_argv(image, "sciel-oracle-preflight")
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                            timeout=STARTUP_TIMEOUT, env=_docker_env(), check=True)
    record = json.loads(result.stdout)[0]
    if (record["Id"] != image or
            record["Config"].get("Labels", {}).get("sciel.oracle.assets.sha256") != runtime_assets_sha256()
            or record["Config"]["Entrypoint"] !=
                ["python", "-I", "-B", "/opt/oracle/container_supervisor.py"]):
        raise ValueError("oracle image does not match trusted task assets")
    return {"image_id": image, "assets_sha256": runtime_assets_sha256(),
            "base_image": BASE_IMAGE}


def run_container_trial(source: bytes, instance, image: str, *, timeout=TRIAL_TIMEOUT,
                        startup_timeout=STARTUP_TIMEOUT):
    """Untrusted stdout is bounded JSON RPC, never an evaluator result.

    The container watchdog owns its deadline independently of this process. An
    outer stage SIGKILL can orphan the Docker client but cannot disable PID1's
    deadline. --rm removes the namespace after that trusted PID1 exits.
    """
    if not 0 < len(source) <= MAX_SOURCE_BYTES:
        raise ValueError("candidate size outside allowed range")
    if not 0 < timeout <= 30 or not 0 < startup_timeout <= 30:
        raise ValueError("invalid runtime deadline")
    session = OracleSession(objective_for(instance))
    name = "sciel-oracle-"+uuid.uuid4().hex
    started = time.monotonic()
    deadline = started + startup_timeout
    status, phase, stderr = "infrastructure_error", "startup", bytearray()
    proc, ready_at, output_bytes = None, None, 0
    selector = selectors.DefaultSelector()
    outgoing, incoming = bytearray(), bytearray()
    error = None
    try:
        proc = subprocess.Popen(docker_argv(image, name, timeout), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=_docker_env(), bufsize=0)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(proc.stdout, selectors.EVENT_READ, "out")
        selector.register(proc.stderr, selectors.EVENT_READ, "err")
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                status = "startup_timeout" if phase == "startup" else "timeout"
                break
            events = selector.select(min(remaining, .1))
            for key, _ in events:
                if key.data == "in":
                    try:
                        size = os.write(key.fd, outgoing)
                    except BrokenPipeError:
                        raise ProtocolError("candidate input closed")
                    del outgoing[:size]
                    if not outgoing:
                        selector.unregister(proc.stdin)
                    continue
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                output_bytes += len(chunk)
                if output_bytes > 1048576:
                    raise ProtocolError("candidate output limit exceeded")
                if key.data == "err":
                    stderr.extend(chunk[:max(0, 65536-len(stderr))])
                    continue
                incoming.extend(chunk)
                while b"\n" in incoming:
                    frame, _, rest = incoming.partition(b"\n")
                    incoming[:] = rest
                    if phase == "startup":
                        if decode_frame(frame) != {"status":"ready", "protocol":PROTOCOL}:
                            raise ProtocolError("invalid trusted supervisor greeting")
                        ready_at = time.monotonic()
                        deadline, phase = ready_at+timeout, "trial"
                        outgoing.extend(struct.pack("!I",len(source))+source+_canonical(session.hello())+b"\n")
                    else:
                        response = session.handle(frame)
                        outgoing.extend(_canonical(response)+b"\n")
                    if outgoing and not any(k.data=="in" for k in selector.get_map().values()):
                        selector.register(proc.stdin, selectors.EVENT_WRITE, "in")
                    if session.finished:
                        status = "ok"
                        break
                if len(incoming)>MAX_FRAME_BYTES:
                    raise ProtocolError("candidate frame limit exceeded")
                if session.finished:
                    break
            if session.finished:
                break
            if proc.poll() is not None and not events:
                status = "infrastructure_error" if phase == "startup" else "candidate_error"
                if proc.returncode == 124 and phase != "startup":
                    status = "timeout"
                elif proc.returncode == 125:
                    status = "infrastructure_error"
                break
    except ProtocolError as exc:
        status = "infrastructure_error" if phase == "startup" else "protocol_error"
        error = str(exc)
    except (OSError, subprocess.SubprocessError) as exc:
        status, error = "infrastructure_error", type(exc).__name__
    finally:
        selector.close()
        if proc is not None:
            try:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=2, env=_docker_env())
            except (OSError, subprocess.TimeoutExpired):
                pass  # PID1 still owns its bounded deadline, including startup EOF.
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()
    return {"status": status, "score": 1/(1+session.best_value) if status=="ok" else 0.,
            "queries": session.queries, "query_budget": QUERY_BUDGET,
            "best_point": session.best_point, "best_value": session.best_value,
            "source_sha256": hashlib.sha256(source).hexdigest(), "image_id": image,
            "container_name": name, "elapsed_seconds": time.monotonic()-started,
            "startup_seconds": None if ready_at is None else ready_at-started,
            "observations": session.transcript,
            "error": error, "stderr": stderr.decode("utf-8","replace")}
