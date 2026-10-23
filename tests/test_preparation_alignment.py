"""The v1/v2 comparison independently verifies sentence and span alignment."""
from copy import deepcopy

import pytest

from src.data.compare_preparation import check_alignment
from src.data.nucle_sgml import load_nucle
from src.data.parse_nucle import documents_to_dataframe
from src.data.prepare_nucle import build_dataset, file_hash, make_nlp


def test_comparison_alignment_checks_token_assignment_and_partial_flags(tmp_path):
    source = tmp_path / "source.sgml"
    source.write_text('<DOC nid="x"><TEXT><P>Dogs run.</P></TEXT><ANNOTATION>'
                      '<MISTAKE start_par="0" start_off="3" end_par="0" end_off="4">'
                      '<TYPE>Nn</TYPE><CORRECTION></CORRECTION></MISTAKE></ANNOTATION></DOC>', encoding="utf-8")
    docs = load_nucle(source)
    frame = documents_to_dataframe(docs)
    data = build_dataset(docs, frame, {0: []}, {}, file_hash(source), make_nlp())
    sample, sentence, record = data["samples"][0], data["sentences"][0], data["provenance"][0]

    check_alignment(sample, sentence, record, "Dogs run.")
    bad = deepcopy(sample)
    bad["right_token_indices"] = [0, 1]
    with pytest.raises(ValueError, match="Token assignment"):
        check_alignment(bad, sentence, record, "Dogs run.")
    bad = deepcopy(sample)
    bad["cuts_start_token"] = False
    with pytest.raises(ValueError, match="Partial-token"):
        check_alignment(bad, sentence, record, "Dogs run.")
