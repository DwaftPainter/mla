
"""
Download NUCLE (NUS Corpus of Learner English) and parse it into a flat CSV
of (sentence-span, error_type, correction) rows for the 5 target classes.
 
Source (verified via web search, 2026-09-24):
    https://huggingface.co/datasets/nusnlp/NUCLE
    -> data/nucle3.2.sgml  (single SGML file, ~12.6 MB, NUCLE release 3.3)
 
Licensing:
    Distributed under the "standard NUS licensing agreement"
    (see https://huggingface.co/datasets/nusnlp/NUCLE/blob/main/nucle_license.pdf).
    By downloading this file you agree to that license. Read it and cite:
    Dahlmeier, Ng & Wu (2013), "Building a Large Annotated Corpus of Learner
    English: The NUS Corpus of Learner English", BEA 2013.
    https://aclanthology.org/W13-1703/
 
Format (per the official dataset card):
    <DOC nid="840">
      <TEXT><P>...</P>...</TEXT>
      <ANNOTATION teacher_id="173">
        <MISTAKE start_par="0" start_off="0" end_par="0" end_off="26">
          <TYPE>ArtOrDet</TYPE>
          <CORRECTION>The engineering design process</CORRECTION>
        </MISTAKE>
        ...
      </ANNOTATION>
    </DOC>
    ...
 
Full NUCLE 2.1+ has 27 error categories. This project's README only targets
5 of them (ArtOrDet, Prep, Nn, Vform, SVA) -- see LABEL_TO_ID in the project
README. This script keeps ALL parsed rows in the raw CSV (so you can inspect
class distribution / decide later), and separately writes a filtered CSV
containing only the 5 target-class rows, which is what the ML pipeline
should actually train on.
 
NOTE: the raw file is not a single well-formed XML document (it is a bare
sequence of <DOC>...</DOC> blocks with no enclosing root, and may contain
unescaped '&' or stray '<' inside text). This script wraps it in a
synthetic <root> element and falls back to lxml's recovering parser if the
standard library parser chokes on malformed markup.
"""
 
import csv
import time
from pathlib import Path
from xml.etree import ElementTree as ET
 
import requests
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 60  # seconds -- file is ~12.6 MB
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
 
URL = "https://huggingface.co/datasets/nusnlp/NUCLE/resolve/main/data/nucle3.2.sgml?download=true"
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
 
SGML_FILE = RAW_DIR / "nucle3.2.sgml"
CSV_FILE_ALL = RAW_DIR / "nucle_annotations_all.csv"
CSV_FILE_TARGET = RAW_DIR / "nucle_annotations_target5.csv"
 
# The 5 classes this project actually classifies (see project README).
TARGET_LABELS = {"ArtOrDet", "Prep", "Nn", "Vform", "SVA"}
 
 
class DownloadData:
    @staticmethod
    def fetch_data(url: str, dest: Path) -> None:
        """Stream-download `url` to `dest`, creating parent dirs as needed."""
        dest.parent.mkdir(parents=True, exist_ok=True)
 
        if dest.exists():
            print(f"Already downloaded, skipping: {dest}")
            return
 
        last_error = None
 
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with requests.get(
                    url,
                    stream=True,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
 
                print(f"Saved: {dest}")
                return
 
            except requests.exceptions.RequestException as exc:
                last_error = exc
                print(f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if dest.exists():
                    dest.unlink()  # remove partial file before retrying
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
 
        raise RuntimeError(
            f"Could not download {url} after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}. "
            "If the file is gated behind a Hugging Face login, download it "
            "manually from https://huggingface.co/datasets/nusnlp/NUCLE "
            "(accept the license there) and place it at "
            f"{dest}, then re-run this script -- it will skip the download "
            "step since the file already exists."
        ) from last_error
 
 
class DataToCsv:
    @staticmethod
    def parse_sgml_file(file_content: bytes) -> list[dict]:
        """
        Parse the full NUCLE SGML file (a bare sequence of <DOC> blocks).
 
        Returns:
            list[dict]
        """
        wrapped = b"<root>\n" + file_content + b"\n</root>"
 
        try:
            root = ET.fromstring(wrapped)
        except ET.ParseError as exc:
            print(f"Standard XML parser failed ({exc}); retrying with lxml's recovering parser.")
            try:
                from lxml import etree as LET
            except ImportError as imp_exc:
                raise RuntimeError(
                    "The file has malformed markup that the standard library "
                    "parser can't handle. Install lxml (`pip install lxml`) "
                    "so the recovering parser can be used instead."
                ) from imp_exc
 
            parser = LET.XMLParser(recover=True)
            root = LET.fromstring(wrapped, parser=parser)
 
        rows = []
 
        for doc in root.findall(".//DOC"):
            doc_id = doc.get("nid")
 
            paragraphs = []
            text_element = doc.find("TEXT")
 
            if text_element is not None:
                for paragraph in text_element.findall("P"):
                    paragraphs.append("".join(paragraph.itertext()).strip())
 
            annotations = doc.findall(".//MISTAKE")
 
            for mistake in annotations:
                try:
                    start_par = int(mistake.get("start_par"))
                    start_off = int(mistake.get("start_off"))
                    end_par = int(mistake.get("end_par"))
                    end_off = int(mistake.get("end_off"))
                except (TypeError, ValueError):
                    continue  # malformed offsets -- skip this annotation
 
                error_type = mistake.findtext("TYPE")
                correction = mistake.findtext("CORRECTION")
 
                # Make sure paragraph indexes exist
                if start_par >= len(paragraphs) or end_par >= len(paragraphs):
                    continue
 
                if start_par == end_par:
                    original = paragraphs[start_par][start_off:end_off]
                else:
                    # Annotation spans multiple paragraphs
                    parts = [paragraphs[start_par][start_off:]]
                    for par_idx in range(start_par + 1, end_par):
                        parts.append(paragraphs[par_idx])
                    parts.append(paragraphs[end_par][:end_off])
                    original = " ".join(parts)
 
                rows.append(
                    {
                        "doc_id": doc_id,
                        "start_par": start_par,
                        "start_off": start_off,
                        "end_par": end_par,
                        "end_off": end_off,
                        "original": original,
                        "correction": correction or "",
                        "error_type": error_type or "",
                    }
                )
 
        return rows
 
    @staticmethod
    def save_csv(rows: list[dict], dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
 
        fieldnames = [
            "doc_id",
            "start_par",
            "start_off",
            "end_par",
            "end_off",
            "original",
            "correction",
            "error_type",
        ]
 
        with open(dest, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
 
        print(f"Saved CSV: {dest}  ({len(rows)} rows)")
 
 
if __name__ == "__main__":
    DownloadData.fetch_data(URL, SGML_FILE)
 
    file_bytes = SGML_FILE.read_bytes()
    all_rows = DataToCsv.parse_sgml_file(file_bytes)
 
    print(f"\nTotal parsed annotations (all 27 NUCLE error categories): {len(all_rows)}")
 
    DataToCsv.save_csv(all_rows, CSV_FILE_ALL)
 
    target_rows = [r for r in all_rows if r["error_type"] in TARGET_LABELS]
    print(f"Rows matching the project's 5 target classes: {len(target_rows)}")
 
    DataToCsv.save_csv(target_rows, CSV_FILE_TARGET)
 
    # quick sanity check: class distribution among target rows
    from collections import Counter
    dist = Counter(r["error_type"] for r in target_rows)
    print("\nClass distribution (target 5):")
    for label, count in dist.most_common():
        print(f"  {label:10s} {count}")