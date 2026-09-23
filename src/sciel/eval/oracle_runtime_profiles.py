"""Finite historical evidence support, never an image dispatch allowlist.

The NumPy tuple is actual TRAIN1/2 (601875e/ce44304); scientific75 is actual
TRAIN3 (b796fde). Their original version and task identity stay on each row.
Only the installed host feedback projection differs; the shared scorer, broker
and bootstrap are byte-pinned. Ordinary journal/arithmetic/pairing checks remain.
"""

import hashlib
from pathlib import Path


NUMPY_RUNTIME = {
    "image_id": "sha256:e9e81996004825c30f8a71b20b67e38a708c795fa4a550e55729a0d2061c2aa5",
    "assets_sha256": "b79d8cdb9c6654f77d6360b531dfb88f59c5d3dc42d744ab1d5ef7da2d18335b",
    "base_image": "python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff",
}
NUMPY_EVALUATOR_VERSION = (
    "d19b5c88ad24c7b7b584f9137a8d3e2a3b8dd6f99386c0242fdeeaf73f678d09"
)
SCIENTIFIC_RUNTIME = {
    "image_id": "sha256:75d299745c706782be6060114fb19bb7cba9ef84823a5f1e520be92dcf4bfbda",
    "assets_sha256": "192a6ab96a71d34da3fcb8b1974cbdaacbab07c5c30eda5101fa576e9877cbb2",
    "base_image": NUMPY_RUNTIME["base_image"],
}
SCIENTIFIC_EVALUATOR_VERSION = (
    "8cd03883f7464722c7993d7872b75e6adb3fe3a69ccbd436560768e84c5edec3"
)
# Exact evaluator present in both historical task maps, retained unchanged.
HISTORICAL_EVALUATOR_SHA256 = (
    "933cc4205b9e9865e3aceacb9f7965e475ca9590ef7d45dc263bd686e90b78bd"
)
SHARED_FILES = {
    "assets/__init__.py": "43cc49819a16f668edfe24eab612505ba87fb056f0f1202a091a5651751e2953",
    "assets/candidate_runner.py": "d621eadff592336478462ec60e6016fc0a754ca0f43c6b3611873c6d4e92007c",
    "assets/container_supervisor.py": "9cc1ed967c982c4adb25156bbc6050f1ff20f628b1ff793ce0cfc0c54e437ccc",
    "assets/oracle_client.py": "2cced17f78b47594b8ad64f6c4d0acae981fedf349e0c104d783dfccb79441d0",
    "assets/oracle_runtime.py": "05fcaf5bbf6c892841694b6fc848803ab45f44c7ada29249044e1ad35b8f544c",
    "assets/panels.py": "78ab351184b2fe76c53a9d69388890e3f58f9c1a68df5c253decf5b43fb58421",
    "assets/receipts.py": "aac2f0c4c8d202c5641c5ef544fefb454c8c06dd03c5d462f747facd0ee632c5",
}


# Exact reviewed installed host entrypoint/projection, never code from evidence.
FEEDBACK_FILES = {
    "evaluator.py": "493b40fcacfbdf10313ace8d902160b04de5cbdea7888c0cf38d13343cc71506",
    "assets/failure_feedback.py": "9409a4c71b93a38ca2b4e2f360480d9b0f70e5e6812f1943db33e8d916e547b3",
}


# Structurally split installed replayer. Original historical tuples and source
# maps above remain unchanged; these exact helpers are reviewed executable code.
INSTALLED_REPLAY_FILES = {
    "assets/oracle_runtime.py": "47925fb3fc053017b99091c419c9da7b0a4fb94cd9484f28268bf0474e977414",
    "assets/oracle_protocol.py": "d905964d7d73aafd56c0cf342adc1ed98ed29231542936dc42ec2c0ff0832bc3",
    "assets/receipts.py": "f878e19a2ec69144ae43ee7e992cd281cf761f63eedfd767c77d96d0b53e9075",
    "assets/receipt_trials.py": "e07e2aebf149ddb247b0bc625f4c88e96d0df1fe3fced9b0e0963e0a2d114f09",
    "assets/panels.py": "4d5422b99cec5ee4e91f42401149944a12aa24099b97d5ebfda178c648281bd7"
}


def validate_historical_retained(task_root, runtime, version):
    if not any(
        runtime == binding and version == expected
        for binding, expected in (
            (NUMPY_RUNTIME, NUMPY_EVALUATOR_VERSION),
            (SCIENTIFIC_RUNTIME, SCIENTIFIC_EVALUATOR_VERSION),
        )
    ):
        raise ValueError("unsupported retained oracle runtime/version")
    root = Path(task_root)
    # Fail closed after future scorer/replayer/bootstrap or projection changes.
    # These are trusted installed files, never paths/executable code from a row.
    for name, expected in {**SHARED_FILES, **FEEDBACK_FILES, **INSTALLED_REPLAY_FILES}.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("retained oracle replay implementation differs: " + name)
