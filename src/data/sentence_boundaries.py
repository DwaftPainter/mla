"""Conservative text-only boundary repairs over spaCy source-token offsets.

No annotation offsets, labels, corrections or model predictions enter this API.
Sentence boundaries can divide a source token when punctuation and a new word
are glued together; preparation records the resulting exact text fragments.
"""
import re

RULE_VERSION = "nucle-sentence-boundaries-v2"
CITATION_MAX_CHARS = 300
PARENTHETICAL = re.compile(r"\([^()\r\n]{1,300}\)|\[[^\[\]\r\n]{1,300}\]")
CITATION_MARKER = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b|\bn\s*\.\s*d\s*\.?|https?://|www\.", re.I)
ABBREVIATION_END = re.compile(r"\b(?:etc|et\s+al|n\.d)\.\s*,?\s*$", re.I)
SUBORDINATORS = frozenset("because although though while if unless until when where since as once that which whose whom who".split())
OPENERS = "([{“‘"


def _citation_ranges(text):
    for match in PARENTHETICAL.finditer(text):
        body = match.group()[1:-1]
        # Short numeric references such as (3) are citation candidates too.
        numeric_reference = bool(re.fullmatch(r"\s*\d{1,3}(?:\s*[-,]\s*\d{1,3})*\s*", body))
        if "!" not in body and "?" not in body and (CITATION_MARKER.search(body) or numeric_reference):
            yield match.start(), match.end()


def repaired_starts(doc):
    """Return source code-point sentence starts, including safe cuts in glued tokens."""
    text = doc.text
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return []
    starts = {next(t.idx for t in sent if not t.is_space)
              for sent in doc.sents if any(not t.is_space for t in sent)}
    index_at = {t.idx: i for i, t in enumerate(tokens)}

    # Sentencizer can consume an opening bracket into the previous sentence.
    # A straight quote is opening only when the preceding quote count is even.
    for boundary in sorted(starts):
        index = index_at[boundary]
        shifted = index
        while shifted > 0:
            previous = tokens[shifted - 1]
            opening = previous.text in OPENERS or (
                previous.text == '"' and text[:previous.idx].count('"') % 2 == 0)
            if not opening:
                break
            shifted -= 1
        if shifted != index:
            starts.discard(boundary)
            starts.add(tokens[shifted].idx)

    citations = list(_citation_ranges(text))
    for begin, end in citations:
        starts.difference_update(p for p in tuple(starts) if begin <= p < end)
        # ``et al.) for`` and ``n.d.) for`` continue the sentence; the
        # sentencizer may otherwise treat the abbreviation's period as final.
        citation_text = text[begin:end]
        continuation = re.search(r"(?:\bet\s+al|\bn\s*\.\s*d)\.[\s,]*[)\]]$", citation_text, re.I)
        if continuation:
            following = next((t for t in tokens if t.idx >= end), None)
            if following and following.idx in starts and following.text[:1].islower() and not text[end:following.idx].strip():
                starts.discard(following.idx)
        # Keep a citation with its preceding sentence, then split before the
        # next uppercase word, even when source punctuation is glued to it.
        following = re.match(r"[\s.!?]*([A-Z][A-Za-z]+)\b", text[end:])
        if following:
            starts.add(end + following.start(1))

    # etc. before lowercase continuation versus etc. followed by a new sentence.
    for boundary in tuple(starts):
        if boundary == tokens[0].idx or boundary not in index_at:
            continue
        tail = text[max(0, boundary - 40):boundary]
        token = tokens[index_at[boundary]]
        if ABBREVIATION_END.search(tail) and (token.text[0].islower() or token.text[0].isdigit()):
            starts.discard(boundary)

    # Ellipsis can continue a list; only split before a capitalized independent
    # starter. Subordinate starts stay unchanged/ambiguous, not guessed into a fix.
    for match in re.finditer(r"\.{3,}|…", text):
        if any(a <= match.start() < b for a, b in citations):
            continue
        following = next((t for t in tokens if t.idx >= match.end()), None)
        if following is None or text[match.end():following.idx].strip():
            continue
        if following.text[0].isupper() and following.text.lower() not in SUBORDINATORS:
            starts.add(following.idx)
        else:
            starts.discard(following.idx)

    # Preserve leading whitespace exactly; all other boundaries are token starts.
    starts.discard(tokens[0].idx)
    return [0, *sorted(p for p in starts if p > 0)]


def rule_metadata():
    return {"version": RULE_VERSION, "citation_max_chars": CITATION_MAX_CHARS,
            "citation_pattern": PARENTHETICAL.pattern, "citation_marker": CITATION_MARKER.pattern,
            "abbreviation_pattern": ABBREVIATION_END.pattern,
            "ellipsis_subordinators": sorted(SUBORDINATORS), "spacy_tokenizer_changed": False,
            "sentence_boundary_can_split_source_token": True}
