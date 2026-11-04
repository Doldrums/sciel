"""The best known certificate for this task: L. Tang's independent set in C_7^(box 680) (teorth/optimizationproblems
PR 296, github.com/gomyway/c9-shannon-c7-lower-bound-certificate, commit b5d6c2c), 3.258838705444608..., above
R. Tandon's 3.2588326203532663... (d = 500). Nine gadgets: Tandon's left patterns and four nested patterns
N<S<A,B,P>,C> with new counts applied to new right factors, combined by the BPZ rule S3b and terminal code K4a. M is
its exact size; c7.value(certificate()) reproduces it."""

GADGETS = [
    {"name": "g0", "op": "hetgao", "pattern": "N<S<G15X,G10,P15X>,G5>", "R": "G5"},
    {"name": "g1", "op": "hetgao", "pattern": "N<S<G15AX,G5,P15A8>,G5>", "R": "G5"},
    {"name": "g2", "op": "sigma", "of": "g1"},
    {"name": "g3", "op": "hetgao", "pattern": "N<S<G15AX,sG15AX,P15A5>,sG15AX>", "R": "g2"},
    {"name": "g4", "op": "sigma", "of": "g3"},
    {"name": "g5", "op": "hetgao", "pattern": "P10A", "R": "G5"},
    {"name": "g6", "op": "sigma", "of": "g5"},
    {"name": "g7", "op": "hetgao", "pattern": "N<S<G15AX,sG15Ah,P15A8>,sG15Ah>", "R": "g6"},
    {"name": "g8", "op": "sigma", "of": "g7"},
]
TOP = {"rule": "S3b", "rule_inputs": ["Phi(g0)", "Phi(g4)", "Phi(g8)"], "terminal_code": "K4a"}
DIMENSION = 680
M = int(
    "76339401955747173864824254218505743060363298976385241217465816694450794523245982409402152970501661150969980820"
    "11440047894668645049922029677807453678039575894778454413370257013143357628408029229204112500314458621233356035"
    "29462879262495021079197672065842679788522316160813639761222918918189081331278899801664815595960211205428261654"
    "5822402349115086721"
)


def certificate() -> dict:
    return {"gadgets": [dict(g) for g in GADGETS], "top": dict(TOP)}
