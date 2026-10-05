from ragpapers.evaluation.matching import (
    contains_quote,
    locate_quote,
    normalize_for_matching,
    quote_score,
)

TEXT = (
    "Compared to GPT-3 175B fine-tuned with Adam [ref], LoRA can reduce the number of "
    "trainable parameters by 10,000 times and the GPU memory requirement by 3 times."
)


def test_normalization_ignores_case_quotes_dashes_refs_and_math():
    assert normalize_for_matching("The “Best”  model – [ref] $d$-dim") == (
        'the "best" model - d-dim'
    )


def test_exact_quote_scores_100_even_across_ref_placeholder():
    assert quote_score("fine-tuned with Adam, LoRA can reduce", TEXT) == 100


def test_quote_copied_from_rendered_page_with_numeric_citation_matches():
    assert quote_score("fine-tuned with Adam [12], LoRA can reduce", TEXT) == 100


def test_small_typo_is_a_fuzzy_match():
    score = quote_score("LoRA can reduce the number of trainable paramters by 10,000 times", TEXT)
    assert 90 <= score < 100


def test_unrelated_quote_does_not_match():
    assert not contains_quote("the decoder uses masked self-attention over outputs", TEXT, 90)


def test_short_text_does_not_contain_a_long_quote():
    # Regression: partial_ratio slides the shorter string, so "and" used to score 100.
    assert quote_score("the model size and the number of training tokens", "and") < 50


def test_locate_quote_returns_section_of_best_block():
    paper = {
        "sections": [
            {"section_id": "S1", "path": ["1 Intro"], "blocks": [{"text": "Nothing here."}]},
            {"section_id": "S2", "path": ["2 Method"], "blocks": [{"text": TEXT}]},
        ]
    }
    match = locate_quote("reduce the number of trainable parameters by 10,000 times", paper)
    assert match.exact
    assert match.section_id == "S2"
    assert match.section_path == ["2 Method"]
