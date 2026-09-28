from io import StringIO

import pandas as pd

from src.data.nucle_sgml import NucleSGMLParser, NucleDocument, NucleMistake
from src.data.parse_nucle import documents_to_dataframe, reconstruct_span
from src.data.report_nucle import build_report
from src.data.data_download import SGML_FILE


def test_title_unclosed_paragraphs_and_literal_ampersands():
    parser = NucleSGMLParser()
    parser.feed('<DOC nid="1"><TEXT><TITLE>Title</TITLE><P>caused A&A'
                '<P>next</P></TEXT><ANNOTATION>'
                '<MISTAKE start_par="1" start_off="0" end_par="1" end_off="6">'
                '<TYPE>Vform</TYPE><CORRECTION>cause</CORRECTION></MISTAKE>'
                '</ANNOTATION></DOC>')
    parser.close()
    assert parser.documents[0].paragraphs == ["Title", "caused A&A", "next"]
    frame = documents_to_dataframe(parser.documents)
    assert frame.iloc[0].error_span == "caused"
    assert not frame.iloc[0].missing_correction


def test_invalid_spans_are_not_insertions_and_cross_endpoints_are_checked():
    doc = NucleDocument("1", ["Title", "abc"], [
        NucleMistake(1, 9, 1, 10, "Vform", "x"),
        NucleMistake(1, 1, 1, 1, "Vform", "x"),
        NucleMistake(0, 0, 1, 2, "Vform", "x"),
        NucleMistake(0, 0, 9, 2, "Vform", "x"),
        NucleMistake(1, 2, 0, 0, "Vform", "x"),
        NucleMistake(0, 0, 1, 99, "Vform", "x"),
    ])
    frame = documents_to_dataframe([doc])
    assert frame.edit_type.tolist() == [
        "invalid_offset", "insertion", "cross_paragraph",
        "invalid_offset", "invalid_offset", "invalid_offset"]
    assert frame.is_insertion.tolist() == [False, True, False, False, False, False]
    roundtrip = pd.read_csv(StringIO(frame.to_csv(index=False)), keep_default_na=False)
    report = build_report(roundtrip)
    assert "Excluded target rows: **5**" in report
    assert "Empty spans: **1**" in report


def test_empty_input_has_report_schema_and_download_path():
    assert "Total annotation rows: **0**" in build_report(documents_to_dataframe([]))
    assert SGML_FILE.parts[-3:] == ("data", "raw", "nucle3.2.sgml")
    assert SGML_FILE.parent.parent.parent.name != "src"
