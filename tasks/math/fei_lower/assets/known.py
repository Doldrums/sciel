"""The best known functions, as exact data: from sciel.task import asset; known = asset("known").

record() is Numaro's function (July 2026, doi 10.5281/zenodo.21497769, file fei_c71_n18_artifact.json): balanced
and monotone on n = 18 variables, a perturbed majority of x1..x9 with nine auxiliary variables each deciding two cells
of the majority boundary; Inf = 261/128 exactly and H / (Inf - 1) = 6.5218457109230465756581439729..., the record of
Tao's table for constant 71a. The truth table is kept as LZMA-compressed bytes of its true_hex, and its SHA-256 binding
is checked on load.
"""

import base64
import hashlib
import lzma

import numpy as np

RECORD_N = 18
RECORD_SHA256 = "f8a7bb2c738d86368305111633376b1a3a83d8ee1f1abcc9a37d841b0f9c7839"
_RECORD = (
    "/Td6WFoAAATm1rRGAgAhARwAAAAQz1jM4H//AaJdAH9p/B++P2Np3naYhZavz/I3cxXTVdNy6yV1xOXq2n0HAWqSwFKO1WIzgyJ+"
    "vWco4CwcRtAyXTOE1NrrmpE/oVt6EqRNmSMLxeacnMM2gw+l44vPdw6utm0VKa7JQZYvusg30pDU3IULV/Yv7clKtooqaYu2Gc9H"
    "DDrgftLmqXmIpgs6DRRVpb3XvYb1BBBhm4ZJT2h50vTaWh5IF1u8hfGeQk7FedpmVF0TNAmqktqy3RAG2wgZxdLuciPsEHx8MV1P"
    "9+XKfI6/pgHNXqSvP0y89BrcpuBDTCcwpZgHnGc0pJIHmBHfgeVwMLZ304y0+phrubArJgv4jvYRsFs8qV+QYyjWA68NZE73TPYb"
    "EPv65ntkZZx/mdtULKz9nd5NcMIISFKn5f+zTQYCnJgN+ZpZR78e4tsICcS+WbTIyLrBVIYNc0FlCJiI+u3CpApy5IA8DQpSlCDg"
    "6gKSfsHTMOzW2woPWdaDRKiAc2E/jZh73Lqbc55kVE+NbOcGQr19dM6o+jn82C3InWkl/INAtsay+GajyqLfvCb3oBGxE/O6CIEA"
    "AADOGd1qijR/vwABvgOAgAIAMRNvmrHEZ/sCAAAAAARZWg=="
)


def record() -> tuple:
    """(18, bits) of the record function, entry x = 1 where f(x) = -1."""
    mask = int(lzma.decompress(base64.b64decode("".join(_RECORD))).hex(), 16)
    N = 1 << RECORD_N
    bits = np.unpackbits(np.frombuffer(mask.to_bytes(N // 8, "little"), dtype=np.uint8), bitorder="little")
    table = ",".join(np.where(bits == 1, "-1", "1").tolist())
    if hashlib.sha256(table.encode()).hexdigest() != RECORD_SHA256:
        raise ValueError("record truth table does not match its SHA-256 binding")
    return RECORD_N, bits.copy()


def majority(m) -> tuple:
    """(m, bits) of majority on an odd number m of variables: TRUE when more than half the inputs are TRUE. It is
    balanced; m = 3 gives H = 2, Inf = 3/2 and H / (Inf - 1) = 4."""
    if m < 1 or m % 2 == 0:
        raise ValueError("majority needs an odd number of variables")
    idx = np.arange(1 << m)
    ones = sum((idx >> i) & 1 for i in range(m))
    return m, (2 * ones > m).astype(np.uint8)
