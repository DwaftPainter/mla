from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional


@dataclass
class NucleMistake:
    start_par: int
    start_off: int
    end_par: int
    end_off: int
    error_type: str = ""
    correction: str = ""
    comment: str = ""
    missing_correction: bool = True


@dataclass
class NucleDocument:
    essay_id: str
    paragraphs: list[str] = field(default_factory=list)
    mistakes: list[NucleMistake] = field(default_factory=list)


class NucleSGMLParser(HTMLParser):
    """
    Minimal Python 3 parser for NUCLE SGML.

    Why use a custom HTMLParser instead of BeautifulSoup?
    NUCLE annotations use character offsets inside paragraphs. We therefore
    want to preserve paragraph character data as closely as possible and avoid
    cleanup/normalization steps that could shift offsets.

    The parser intentionally extracts only fields needed for Data Understanding:
      DOC/@nid
      TEXT/TITLE and TEXT/P (in source order)
      ANNOTATION/MISTAKE
      TYPE
      CORRECTION
      COMMENT
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.documents: list[NucleDocument] = []

        self._current_doc: Optional[NucleDocument] = None
        self._current_mistake: Optional[NucleMistake] = None

        self._inside_text = False
        self._inside_p = False
        self._inside_type = False
        self._inside_correction = False
        self._inside_comment = False

        self._paragraph_buffer: list[str] = []
        self._type_buffer: list[str] = []
        self._correction_buffer: list[str] = []
        self._comment_buffer: list[str] = []

    @staticmethod
    def _attrs_to_dict(attrs: list[tuple[str, Optional[str]]]) -> dict[str, str]:
        return {k: "" if v is None else v for k, v in attrs}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        tag = tag.lower()
        attr = self._attrs_to_dict(attrs)

        if tag == "doc":
            essay_id = attr.get("nid", "")
            self._current_doc = NucleDocument(essay_id=essay_id)

        elif tag == "text":
            self._inside_text = True

        elif tag in {"title", "p"} and self._inside_text:
            # SGML reference sections can omit </P>; retain each opened block.
            if self._inside_p:
                self.handle_endtag("p")
            self._inside_p = True
            self._paragraph_buffer = []

        elif tag == "mistake":
            self._current_mistake = NucleMistake(
                start_par=int(attr["start_par"]),
                start_off=int(attr["start_off"]),
                end_par=int(attr["end_par"]),
                end_off=int(attr["end_off"]),
            )

        elif tag == "type" and self._current_mistake is not None:
            self._inside_type = True
            self._type_buffer = []

        elif tag == "correction" and self._current_mistake is not None:
            self._current_mistake.missing_correction = False
            self._inside_correction = True
            self._correction_buffer = []

        elif tag == "comment" and self._current_mistake is not None:
            self._inside_comment = True
            self._comment_buffer = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in {"title", "p"} and self._inside_p:
            paragraph = "".join(self._paragraph_buffer)

            # The SGML formatting places a line break immediately after <P>
            # and immediately before </P>. Those formatting newlines are not
            # part of the paragraph offsets used by NUCLE.
            paragraph = paragraph.strip("\n\r")

            if self._current_doc is not None:
                self._current_doc.paragraphs.append(paragraph)

            self._inside_p = False
            self._paragraph_buffer = []

        elif tag in {"reference", "text"}:
            if self._inside_p:
                self.handle_endtag("p")
            if tag == "text":
                self._inside_text = False

        elif tag == "type" and self._current_mistake is not None:
            self._current_mistake.error_type = "".join(self._type_buffer)
            self._inside_type = False
            self._type_buffer = []

        elif tag == "correction" and self._current_mistake is not None:
            self._current_mistake.correction = "".join(self._correction_buffer)
            self._inside_correction = False
            self._correction_buffer = []

        elif tag == "comment" and self._current_mistake is not None:
            self._current_mistake.comment = "".join(self._comment_buffer)
            self._inside_comment = False
            self._comment_buffer = []

        elif tag == "mistake":
            if self._current_doc is not None and self._current_mistake is not None:
                self._current_doc.mistakes.append(self._current_mistake)
            self._current_mistake = None

        elif tag == "doc":
            if self._current_doc is not None:
                self.documents.append(self._current_doc)
            self._current_doc = None

    def handle_data(self, data: str) -> None:
        if self._inside_p:
            self._paragraph_buffer.append(data)

        if self._inside_type:
            self._type_buffer.append(data)

        if self._inside_correction:
            self._correction_buffer.append(data)

        if self._inside_comment:
            self._comment_buffer.append(data)

    def feed(self, data: str) -> None:
        # SGML character offsets refer to literal text. Protect ampersands so
        # HTMLParser neither decodes entities nor adds semicolons to bare &words.
        super().feed(data.replace("&", "&amp;"))


def load_nucle(path: str | Path) -> list[NucleDocument]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"NUCLE SGML file not found: {path}\n"
            "Expected location by default: data_set/raw/nucle3.2.sgml"
        )

    raw = path.read_text(encoding="utf-8")

    parser = NucleSGMLParser()
    parser.feed(raw)
    parser.close()

    return parser.documents
