"""arch-v4 #4b: VQ-Elites niche codebook (VQBook) + its wiring into the map."""

import numpy as np

from sciel.components.policy.code_embed import VQBook
from sciel.engine.db import Database


def _e(*v):
    return np.array(v, dtype=np.float32)


def test_first_vector_seeds_niche_zero():
    b = VQBook()
    assert b.assign(_e(1, 0)) == 0
    assert len(b.centroids) == 1


def test_far_vector_grows_near_vector_joins():
    b = VQBook(d_min=0.3, target_k=6, k_max=12)
    b.assign(_e(1, 0))                       # niche 0
    far = b.assign(_e(0, 1))                 # orthogonal -> cosine dist 1.0 > 0.3
    assert far == 1 and len(b.centroids) == 2
    near = b.assign(_e(0.99, 0.14))          # ~parallel to niche 0
    assert near == 0 and len(b.centroids) == 2


def test_k_max_caps_growth():
    b = VQBook(d_min=0.0, k_max=3)           # d_min 0 -> every distinct point wants a niche
    for ang in np.linspace(0.1, 1.5, 8):
        b.assign(_e(float(np.cos(ang)), float(np.sin(ang))))
    assert len(b.centroids) == 3             # capped, later points fold into nearest


def test_csc_raises_threshold_when_over_target():
    b = VQBook(d_min=0.3, target_k=2, k_max=99, csc_rate=0.5)
    b.assign(_e(1, 0, 0))
    b.assign(_e(0, 1, 0))                    # k=2 == target: growth at k=2 -> factor (1+.5*(2-2))=1
    d_at_target = b.d_min
    b.assign(_e(0, 0, 1))                    # k=3 > target -> threshold must rise
    assert b.d_min > d_at_target


def test_assign_fixed_never_grows():
    b = VQBook(d_min=0.3)
    b.assign(_e(1, 0))
    k = len(b.centroids)
    assert b.assign_fixed(_e(0, 1)) == 0     # far, but fixed picks nearest existing
    assert len(b.centroids) == k


def test_state_round_trip():
    b = VQBook(d_min=0.3, target_k=4)
    for v in [(1, 0), (0, 1), (0.9, 0.1)]:
        b.assign(_e(*v))
    d = b.state()
    b2 = VQBook()
    b2.load_state(d)
    assert b2.d_min == b.d_min and len(b2.centroids) == len(b.centroids)
    assert b2.assign_fixed(_e(0.2, 0.9)) == b.assign_fixed(_e(0.2, 0.9))


def test_deterministic():
    seq = [(1, 0), (0, 1), (0.5, 0.5), (-1, 0), (0.9, 0.1)]
    a, b = VQBook(d_min=0.3), VQBook(d_min=0.3)
    assert [a.assign(_e(*v)) for v in seq] == [b.assign(_e(*v)) for v in seq]


# --- integration: the codebook drives the MAP-Elites cell ---

class _StubNiche:
    """A niche_fn that embeds a program by a tag in its code -- lets the Database
    integration run without torch / a real embedder."""

    def __init__(self):
        self.book = VQBook(d_min=0.3, target_k=4, k_max=8)

    def _vec(self, code, cases):
        v = np.zeros(4, dtype=np.float32)
        v[int(cases.get("tag", 0)) % 4] = 1.0
        return v

    def assign(self, code, cases):
        return self.book.assign(self._vec(code, cases))

    def assign_fixed(self, code, cases):
        return self.book.assign_fixed(self._vec(code, cases))

    def state(self):
        return self.book.state()

    def load_state(self, d):
        self.book.load_state(d)


def test_cell_uses_niche_id_over_descriptor():
    called = []
    db = Database(num_islands=1, grid_bins=10, niche_fn=_StubNiche(),
                  descriptor_fn=lambda c, cs: called.append(1) or 0.5)
    db.add("prog A", 0.4, {"tag": 0}, 0, origin="seed")
    db.add("prog B", 0.6, {"tag": 1}, 0, origin="gen")
    cells = set(db.islands[0].grid)
    # two distinct niches (0 and 1), score bins 4 and 6
    assert cells == {(0, 4), (1, 6)}
    assert not called                        # descriptor bypassed when a codebook is set


def test_reencode_and_state_round_trip():
    db = Database(num_islands=1, grid_bins=10, niche_fn=_StubNiche())
    for i, tag in enumerate([0, 1, 2, 0]):
        db.add(f"p{i}", 0.5, {"tag": tag}, 0, origin="gen")
    db.reencode_niches()                     # must not raise; keeps one elite per cell
    st = db.state()
    assert st["niche"] is not None
    db2 = Database(num_islands=1, grid_bins=10, niche_fn=_StubNiche())
    db2.load_state(st)
    assert db2.niche_fn.book.d_min == db.niche_fn.book.d_min
    assert len(db2.niche_fn.book.centroids) == len(db.niche_fn.book.centroids)
