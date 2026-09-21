"""espalier-2 directive: compute-budget signal + plateau structural-novelty push."""

from sciel.engine.prompt.build import build, compose_directive
from sciel.hypothesis import hypothesis_instruction, parse_hypothesis


class _Snap:
    task_prompt = "pack circles"
    parent_code = "def run(): pass"
    parent_score = 0.5
    inspirations = []
    kb_lessons = []
    artifacts = {}
    bet_enabled = False
    directive = ""


def test_limit_always_present():
    d = compose_directive(30.0, None, plateau=False)
    assert "30s" in d and "COMPUTE BUDGET" in d
    assert "STAGNATION" not in d
    # the directive must state the kill semantics and demand a deadline guard --
    # quoting a soft "allows up to" figure drove ~75% of evals into timeouts
    assert "KILLS" in d and "deadline" in d


def test_share_quantized_never_raw():
    d = compose_directive(30.0, 3.1210939, plateau=False)
    assert "under a quarter" in d
    # raw wall-clock must never leak into the prompt (journal identity)
    assert "3.1" not in d


def test_share_buckets():
    assert "under a quarter" in compose_directive(100.0, 24.0, False)
    assert "under half" in compose_directive(100.0, 49.0, False)
    assert "most" in compose_directive(100.0, 79.0, False)
    assert "nearly all" in compose_directive(100.0, 99.0, False)


def test_plateau_adds_structural_push():
    d = compose_directive(30.0, 10.0, plateau=True)
    assert "STAGNATION" in d and "STRUCTURAL" in d


def test_refine_pushes_reformulation_not_new_family():
    d = compose_directive(30.0, 28.0, plateau=False, refine=True)
    # refine tells the LLM to KEEP the construction and reformulate the objective
    assert "REFINEMENT" in d and "epigraph" in d and "log-sum-exp" in d
    assert "KEEP this construction" in d
    # and it must NOT emit the change-family push at the same time
    assert "STAGNATION" not in d


def test_refine_takes_precedence_over_plateau():
    # the call site never sets both, but the text must be unambiguous if it does
    d = compose_directive(30.0, 28.0, plateau=True, refine=True)
    assert "REFINEMENT" in d and "STAGNATION" not in d


def test_directive_lands_in_prompt():
    snap = _Snap()
    snap.directive = compose_directive(30.0, 5.0, plateau=True)
    text = build("full_rewrite", snap)
    assert "COMPUTE BUDGET" in text and "STAGNATION" in text


def test_budget_distinguishes_recorded_timing_from_a_speedup_hypothesis():
    text = compose_directive(30.0, 3.1210939, plateau=False)
    assert "recorded parent run used under a quarter" in text
    assert "does not establish a bottleneck" in text
    assert "automatically increase fixed restart or step counts" in text
    assert "additional work and the loop or budget condition" in text
    assert "conditional until the speedup and additional work are measured" in text
    assert "free search power" not in text
    assert "recorded parent run" not in compose_directive(30.0, None, False)


def test_tied_score_is_a_historical_observation_not_a_behavior_verdict():
    text = compose_directive(30.0, None, False, parent_noop=True)
    assert "previous child's primary score" in text
    assert "parent's recorded score by less than 1e-9" in text
    assert "does not establish equivalent behavior" in text
    assert "behaviorally inert" not in text and "score at all" not in text
    assert "SCORE OBSERVATION" not in compose_directive(30.0, None, False)


def test_refinement_is_a_search_instruction_not_a_solver_diagnosis():
    text = compose_directive(30.0, 28.0, False, refine=True)
    assert "search instruction" in text and "has not been established" in text
    assert "Where applicable" in text and "hypotheses to test" in text
    assert "bottleneck is now" not in text and "a gradient solver stalls" not in text


def test_diff_template_respects_both_search_scopes_and_preserves_the_interface():
    snap = _Snap()
    for refine in (False, True):
        snap.directive = compose_directive(30.0, 5.0, plateau=not refine, refine=refine)
        text = build("diff", snap)
        assert "Follow the requested search scope in DIRECTIVE" in text
        assert "return interface EXACTLY" in text and "COMPLETE improved program" in text
        assert "Make a targeted improvement" not in text
        assert "rewrite from scratch" not in text
        if refine:
            assert "KEEP this construction" in text and "STAGNATION" not in text
        else:
            assert "STRUCTURAL change" in text and "KEEP this construction" not in text


def test_forecast_example_is_parseable_without_suggesting_an_algorithm_or_mechanism():
    text = hypothesis_instruction()
    example = text[text.index("\nHYPOTHESIS_JSON:") + 1:]
    forecast = parse_hypothesis(example)
    assert forecast is not None
    assert forecast.text == "<specific change implemented in the candidate>"
    assert forecast.reasoning == "<conditional mechanism and assumptions behind your forecast>"
    assert "replace every placeholder, number and code comment" in text
    for old_anchor in ("Vectorize", "pairwise distance", "loop overhead", "more search steps"):
        assert old_anchor not in text
