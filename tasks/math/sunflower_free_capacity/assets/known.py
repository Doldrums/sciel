"""The best known block, as exact data: from sciel.task import asset; known = asset("known").

record() is Koshta's block (Kos2026, github.com/DennisKoshta/sunflower3-49a-certificate, commit c4ee817, certificate
two_pendant_extra_1.5678599.txt): 252292 sets on 28 points, ports 0 and 1, N00 = 33969, N01 = N10 = 64728,
N11 = 88867, rate (64728 + sqrt(3018723123))^(1/26) = 1.5678599024. As nodes: F (75 sets on the edges 01, 02, 03,
04, 12, 13, 14, 23, 24, 34 of K5, in that order) and H (121 sets on Z11) of Cla2026; H glued at 0 to F at edge 03,
plus the 15 edge sets A1 (19 points); a second H glued at 0 there at edge 04 (28 points: the edges 01, 02, 12, 13, 14,
23, 24, 34, then H1 points 1..10, then H2 points 1..10), plus 827 explicit sets: the 806 additions at H2 and the 21
further sets of Kos2026. Rebuilt this way the block equals the certificate's set for set.
"""

from itertools import combinations, permutations

EDGES = list(combinations(range(5), 2))

A1 = """
7f d5 fb fe 11e 163 17d 1b7 1bb 1bd 1cf 1db 1ed 1ee 1f6
"""

TOP = """
3f7f 3fb3 3fdf 3fef 6f7f 6fb3 6fdf 6fef 747f 74b3 74df 74ef 7f5a 7f77 7f7d 7f9f 7fbd 7fbe 7fc5 7feb 7fee 7ff6 7ff9
8e7f 8eb3 8edf 8eef b700 bb7f bbb3 bbdf bbef bd7f bdb3 bddf bdef d77f d7b3 d7df d7ef dd7f ddb3 dddf ddef df5a df77
df7d df9f dfbd dfbe dfc5 dfeb dfee dff6 dff9 e77f e7b3 e7df e7ef e95a e977 e97d e99f e9bd e9be e9c5 e9eb e9ee e9f6
e9f9 f37f f3b3 f3df f3ef fa7f fab3 fadf faef fe5a fe77 fe7d fe9f febd febe fec5 feeb feee fef6 fef9 11d5a 11d77
11d7d 11d9f 11dbd 11dbe 11dc5 11deb 11dee 11df6 11df9 1237f 123b3 123df 123ef 13e7f 13eb3 13edf 13eef 14f7f 14fb3
14fdf 14fef 1577f 157b3 157df 157ef 15e7f 15eb3 15edf 15eef 16d7f 16db3 16ddf 16def 16e00 1775a 17777 1777d 1779f
177bd 177be 177c5 177eb 177ee 177f6 177f9 1797f 179b3 179df 179ef 17a7f 17ab3 17adf 17aef 17b5a 17b77 17b7d 17b9f
17bbd 17bbe 17bc5 17beb 17bee 17bf6 17bf9 1977f 197b3 197df 197ef 19b7f 19bb3 19bdf 19bef 1ad7f 1adb3 1addf 1adef
1af5a 1af77 1af7d 1af9f 1afbd 1afbe 1afc5 1afeb 1afee 1aff6 1aff9 1b57f 1b5b3 1b5df 1b5ef 1b67f 1b6b3 1b6df 1b6ef
1bb5a 1bb77 1bb7d 1bb9f 1bbbd 1bbbe 1bbc5 1bbeb 1bbee 1bbf6 1bbf9 1be5a 1be77 1be7d 1be9f 1bebd 1bebe 1bec5 1beeb
1beee 1bef6 1bef9 1cb7f 1cbb3 1cbdf 1cbef 1cf5a 1cf77 1cf7d 1cf9f 1cfbd 1cfbe 1cfc5 1cfeb 1cfee 1cff6 1cff9 1d25a
1d277 1d27d 1d29f 1d2bd 1d2be 1d2c5 1d2eb 1d2ee 1d2f6 1d2f9 1d97f 1d9b3 1d9df 1d9ef 1dc7f 1dcb3 1dcdf 1dcef 1df00
1e67f 1e6b3 1e6df 1e6ef 1e75a 1e777 1e77d 1e79f 1e7bd 1e7be 1e7c5 1e7eb 1e7ee 1e7f6 1e7f9 1ea7f 1eab3 1eadf 1eaef
1ec7f 1ecb3 1ecdf 1ecef 1f17f 1f1b3 1f1df 1f1ef 1f55a 1f577 1f57d 1f59f 1f5bd 1f5be 1f5c5 1f5eb 1f5ee 1f5f6 1f5f9
1fb00 1fc5a 1fc77 1fc7d 1fc9f 1fcbd 1fcbe 1fcc5 1fceb 1fcee 1fcf6 1fcf9 1fd00 21f7f 21fb3 21fdf 21fef 2377f 237b3
237df 237ef 23a5a 23a77 23a7d 23a9f 23abd 23abe 23ac5 23aeb 23aee 23af6 23af9 2475a 24777 2477d 2479f 247bd 247be
247c5 247eb 247ee 247f6 247f9 25d7f 25db3 25ddf 25def 25e7f 25eb3 25edf 25eef 26b7f 26bb3 26bdf 26bef 26e7f 26eb3
26edf 26eef 2737f 273b3 273df 273ef 2797f 279b3 279df 279ef 27d5a 27d77 27d7d 27d9f 27dbd 27dbe 27dc5 27deb 27dee
27df6 27df9 27f00 2917f 291b3 291df 291ef 29f5a 29f77 29f7d 29f9f 29fbd 29fbe 29fc5 29feb 29fee 29ff6 29ff9 2a77f
2a7b3 2a7df 2a7ef 2ab7f 2abb3 2abdf 2abef 2af5a 2af77 2af7d 2af9f 2afbd 2afbe 2afc5 2afeb 2afee 2aff6 2aff9 2b67f
2b6b3 2b6df 2b6ef 2bc7f 2bcb3 2bcdf 2bcef 2bd5a 2bd77 2bd7d 2bd9f 2bdbd 2bdbe 2bdc5 2bdeb 2bdee 2bdf6 2bdf9 2cb7f
2cbb3 2cbdf 2cbef 2cd7f 2cdb3 2cddf 2cdef 2d67f 2d6b3 2d6df 2d6ef 2da7f 2dab3 2dadf 2daef 2db5a 2db77 2db7d 2db9f
2dbbd 2dbbe 2dbc5 2dbeb 2dbee 2dbf6 2dbf9 2dc00 2e57f 2e5b3 2e5df 2e5ef 2ec7f 2ecb3 2ecdf 2ecef 2ee5a 2ee77 2ee7d
2ee9f 2eebd 2eebe 2eec5 2eeeb 2eeee 2eef6 2eef9 2ef00 2f35a 2f377 2f37d 2f39f 2f3bd 2f3be 2f3c5 2f3eb 2f3ee 2f3f6
2f3f9 2f55a 2f577 2f57d 2f59f 2f5bd 2f5be 2f5c5 2f5eb 2f5ee 2f5f6 2f5f9 2f65a 2f677 2f67d 2f69f 2f6bd 2f6be 2f6c5
2f6eb 2f6ee 2f6f6 2f6f9 2f87f 2f8b3 2f8df 2f8ef 2fb00 30f7f 30fb3 30fdf 30fef 31b7f 31bb3 31bdf 31bef 32e7f 32eb3
32edf 32eef 32f5a 32f77 32f7d 32f9f 32fbd 32fbe 32fc5 32feb 32fee 32ff6 32ff9 3357f 335b3 335df 335ef 3375a 33777
3377d 3379f 337bd 337be 337c5 337eb 337ee 337f6 337f9 3397f 339b3 339df 339ef 33c7f 33cb3 33cdf 33cef 33f00 3487f
348b3 348df 348ef 3537f 353b3 353df 353ef 3557f 355b3 355df 355ef 35b5a 35b77 35b7d 35b9f 35bbd 35bbe 35bc5 35beb
35bee 35bf6 35bf9 35e5a 35e77 35e7d 35e9f 35ebd 35ebe 35ec5 35eeb 35eee 35ef6 35ef9 35f00 3657f 365b3 365df 365ef
3667f 366b3 366df 366ef 36b5a 36b77 36b7d 36b9f 36bbd 36bbe 36bc5 36beb 36bee 36bf6 36bf9 36d5a 36d77 36d7d 36d9f
36dbd 36dbe 36dc5 36deb 36dee 36df6 36df9 37100 3727f 372b3 372df 372ef 3765a 37677 3767d 3769f 376bd 376be 376c5
376eb 376ee 376f6 376f9 37c5a 37c77 37c7d 37c9f 37cbd 37cbe 37cc5 37ceb 37cee 37cf6 37cf9 3877f 387b3 387df 387ef
38b00 38d7f 38db3 38ddf 38def 3975a 39777 3977d 3979f 397bd 397be 397c5 397eb 397ee 397f6 397f9 39a7f 39ab3 39adf
39aef 39c7f 39cb3 39cdf 39cef 39e5a 39e77 39e7d 39e9f 39ebd 39ebe 39ec5 39eeb 39eee 39ef6 39ef9 3a45a 3a477 3a47d
3a49f 3a4bd 3a4be 3a4c5 3a4eb 3a4ee 3a4f6 3a4f9 3a97f 3a9b3 3a9df 3a9ef 3aa7f 3aab3 3aadf 3aaef 3b27f 3b2b3 3b2df
3b2ef 3b35a 3b377 3b37d 3b39f 3b3bd 3b3be 3b3c5 3b3eb 3b3ee 3b3f6 3b3f9 3b95a 3b977 3b97d 3b99f 3b9bd 3b9be 3b9c5
3b9eb 3b9ee 3b9f6 3b9f9 3bd00 3be00 3c37f 3c3b3 3c3df 3c3ef 3c67f 3c6b3 3c6df 3c6ef 3cd5a 3cd77 3cd7d 3cd9f 3cdbd
3cdbe 3cdc5 3cdeb 3cdee 3cdf6 3cdf9 3ce5a 3ce77 3ce7d 3ce9f 3cebd 3cebe 3cec5 3ceeb 3ceee 3cef6 3cef9 3d47f 3d4b3
3d4df 3d4ef 3d55a 3d577 3d57d 3d59f 3d5bd 3d5be 3d5c5 3d5eb 3d5ee 3d5f6 3d5f9 3d700 3d95a 3d977 3d97d 3d99f 3d9bd
3d9be 3d9c5 3d9eb 3d9ee 3d9f6 3d9f9 3e17f 3e1b3 3e1df 3e1ef 3e35a 3e377 3e37d 3e39f 3e3bd 3e3be 3e3c5 3e3eb 3e3ee
3e3f6 3e3f9 3e700 3ea5a 3ea77 3ea7d 3ea9f 3eabd 3eabe 3eac5 3eaeb 3eaee 3eaf6 3eaf9 3ed00 3f07f 3f0b3 3f0df 3f0ef
3f600 3f85a 3f877 3f87d 3f89f 3f8bd 3f8be 3f8c5 3f8eb 3f8ee 3f8f6 3f8f9 3fa00 ffc0034
"""


def _edges(pairs):
    return sum(1 << EDGES.index(tuple(sorted(e))) for e in pairs)


def F() -> list:
    """75 sets on the 10 edges of K5: the 5 stars, the 10 K_{2,3} and the 60 complements of chairs."""
    fam = {_edges([e for e in EDGES if v in e]) for v in range(5)}
    for a, b in combinations(range(5), 2):
        rest = [x for x in range(5) if x not in (a, b)]
        fam.add(_edges([(x, y) for x in (a, b) for y in rest]))
    for c, l1, l2, l3, x in permutations(range(5)):
        fam.add(1023 ^ _edges([(c, l1), (c, l2), (c, l3), (l1, x)]))
    return sorted(fam)


def H() -> list:
    """121 sets on Z11: the images of {0..6}, {0,1,2,3,4,6,7} and {0,1,2,6,9} under x -> ax + b, a a nonzero square."""
    squares = {a * a % 11 for a in range(1, 11)}
    return sorted({sum(1 << (a * x + b) % 11 for x in base)
                   for base in ((0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 6, 7), (0, 1, 2, 6, 9))
                   for a in squares for b in range(11)})


def record() -> dict:
    """Koshta's block as the candidate format: {"nodes": [...], "ports": [0, 1]}."""
    return {"nodes": [{"n": 10, "sets": F()}, {"n": 11, "sets": H()},
                      {"glue": [0, 2, 1, 0], "sets": [int(x, 16) for x in A1.split()]},
                      {"glue": [2, 2, 1, 0], "sets": [int(x, 16) for x in TOP.split()]}],
            "ports": [0, 1]}
