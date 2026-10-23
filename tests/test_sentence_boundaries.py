"""Synthetic regressions for v2 repairs and the v1 audit's counterexamples."""
import pytest

from src.data.prepare_nucle import make_nlp, segment_paragraph


@pytest.fixture(scope="module")
def nlp():
    return make_nlp()


@pytest.mark.parametrize("text,expected", [
    ("It works. (Rogers, 2003) Next sentence.", ["It works. (Rogers, 2003) ", "Next sentence."]),
    ("It works. (Wilpert.B, 2007) In other words, yes.", ["It works. (Wilpert.B, 2007) ", "In other words, yes."]),
    ("It works. (3) Next sentence.", ["It works. (3) ", "Next sentence."]),
    ("Even though it works (Chapin, et al., 2004), it costs more. Next.",
     ["Even though it works (Chapin, et al., 2004), it costs more. ", "Next."]),
    ("This is useful (Sterbentz, n.d.) for reactors. Thus, yes.",
     ["This is useful (Sterbentz, n.d.) for reactors. ", "Thus, yes."]),
    ("We use vitamins and etc. for specific needs. Next.", ["We use vitamins and etc. for specific needs. ", "Next."]),
    ("This affects life.(Schuett, 2008) There are alternatives.",
     ["This affects life.(Schuett, 2008) ", "There are alternatives."]),
    ("There is solar energy etc...The panels work.", ["There is solar energy etc...", "The panels work."]),
    ("We fund schools etc... With more money, it improves.", ["We fund schools etc... ", "With more money, it improves."]),
    ('"Go!" (He left.) Next.', ['"Go!" ', '(He left.) ', 'Next.']),
    # Counterexamples: repairs must not indiscriminately merge/split punctuation.
    ("We make light bulbs and etc. These products work.", ["We make light bulbs and etc. ", "These products work."]),
    ("Wheelchairs, glasses... have helped.", ["Wheelchairs, glasses... have helped."]),
    ("Wheelchairs, glasses… have helped.", ["Wheelchairs, glasses… have helped."]),
    ("Tax... Because it matters, we pay.", ["Tax... Because it matters, we pay."]),
    ("We see problems, e.g. limited staff. Next.", ["We see problems, e.g. limited staff. ", "Next."]),
    ("Mr. Lee arrived. He left.", ["Mr. Lee arrived. ", "He left."]),
    ("The U.S. team (DOE) works. Next.", ["The U.S. team (DOE) works. ", "Next."]),
    ("United States(U.S.) while others wait.", ["United States(U.S.) while others wait."]),
    ("It works (Smith, n.d.). Thus, yes.", ["It works (Smith, n.d.). ", "Thus, yes."]),
    ("Xiaonei.com grew by 9.7 percent. Next.", ["Xiaonei.com grew by 9.7 percent. ", "Next."]),
    ("Read http://en.wikipedia.org/wiki/Example for details.", ["Read http://en.wikipedia.org/wiki/Example for details."]),
    ("Care the old", ["Care the old"]),
    ('"Think outside the box". It is a sentence.', ['"Think outside the box". ', 'It is a sentence.']),
    ('"It works." they argues. Next.', ['"It works." ', 'they argues. ', 'Next.']),
    ("  It works. (2008)  Next.\t", ["  It works. (2008)  ", "Next.\t"]),
])
def test_v2_boundaries_and_unchanged_tokens(nlp, text, expected):
    actual = segment_paragraph(text, nlp)
    assert [s["sentence"] for s in actual] == expected
    assert "".join(s["sentence"] for s in actual) == text
    original = [(t.idx, t.idx + len(t.text), t.text) for t in nlp(text) if not t.is_space]
    reconstructed = [(s["sentence_start_par"] + t["start"], s["sentence_start_par"] + t["end"], t["text"])
                     for s in actual for t in s["tokens"]]
    assert reconstructed == original


def test_versions_are_explicit_and_unknown_policy_fails(nlp):
    text = "It works. (Smith, 2008) Next."
    v1 = segment_paragraph(text, nlp, "nucle-preparation-v1")
    v2 = segment_paragraph(text, nlp, "nucle-preparation-v2")
    assert [s["sentence"] for s in v1] != [s["sentence"] for s in v2]
    with pytest.raises(ValueError, match="Unknown"):
        segment_paragraph(text, nlp, "unknown")
