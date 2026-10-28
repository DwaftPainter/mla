"""Validate report exports, frozen evidence and package a corpus-free handover."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

from docx import Document
from pptx import Presentation
import pymupdf
from PIL import Image, ImageOps, ImageDraw

from src.evaluation.bert_final_analysis import ROOT, csv_read, dump, read, sha
from src.reporting.document_export import blocks, plain

FINAL = ROOT / "reports/summaries/final"
ANALYSIS = ROOT / "reports/bert_final_analysis"
PREVIEW = ROOT / ".cache/bert_delivery_preview"


def check_links():
    missing = []
    paths = [ROOT / "README.md", ROOT / "note.md"] + list((ROOT / "docs").rglob("*.md")) + list((ROOT / "reports").rglob("*.md"))
    checked = 0
    for path in paths:
        if any(p in {"documentation_archive", "before_layout"} for p in path.parts):
            continue
        text = path.read_text(encoding="utf-8-sig")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^\n]+?)\)", text):
            target = match[1].strip().strip("<>")
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            target = target.split("#")[0]
            if not target: continue
            checked += 1
            if not (path.parent / target).resolve().exists():
                missing.append({"file": path.relative_to(ROOT).as_posix(), "target": target})
    return {"checked": checked, "missing": missing}


def pdf_checks(path, thumbnail_prefix):
    document = pymupdf.open(path)
    page_info, bad_boxes, replacement_glyphs = [], [], 0
    for index, page in enumerate(document):
        text = page.get_text()
        replacement_glyphs += text.count("\ufffd")
        page_info.append({"page": index + 1, "text_chars": len(text), "images": len(page.get_images())})
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                        bad_boxes.append({"page": index + 1, "text": span["text"][:60], "bbox": list(span["bbox"])})
    # Render every page to contact sheets, twelve pages per sheet.
    for first in range(0, len(document), 12):
        images = []
        for index in range(first, min(first + 12, len(document))):
            pixmap = document[index].get_pixmap(matrix=pymupdf.Matrix(.48, .48), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image = ImageOps.expand(image, border=(4, 22, 4, 4), fill="#e2e8f0")
            ImageDraw.Draw(image).text((8, 4), f"Page {index+1}", fill="black")
            images.append(image)
        columns = 3 if document[0].rect.width > document[0].rect.height else 4
        w, h = max(i.width for i in images), max(i.height for i in images)
        sheet = Image.new("RGB", (columns*w, ((len(images)+columns-1)//columns)*h), "#cbd5e1")
        for index, image in enumerate(images): sheet.paste(image, ((index % columns)*w, (index // columns)*h))
        sheet.save(PREVIEW / f"{thumbnail_prefix}_{first//12+1}.png")
    full_text = "\n".join(page.get_text() for page in document)
    return {"pages": len(document), "page_info": page_info, "out_of_page_text": bad_boxes,
            "replacement_glyphs": replacement_glyphs, "contains_primary_f1": "0,8146" in full_text,
            "contains_seed_mean": "0,8182" in full_text}


def docx_checks(stem):
    md = FINAL / f"{stem}.md" if stem != "BERT_comparison_report" else ANALYSIS / "analysis_report.md"
    expected = [[[plain(c) for c in row] for row in value] for kind, value in blocks(md) if kind == "table"]
    document = Document(FINAL / f"{stem}.docx")
    actual = [[[cell.text for cell in row.cells] for row in table.rows] for table in document.tables]
    return {"markdown_tables": len(expected), "docx_tables": len(actual), "tables_exact": expected == actual,
            "images": len(document.inline_shapes)}


def package():
    filename = FINAL / "NUCLE_delivery_20261028.zip"
    files = []
    for path in FINAL.iterdir():
        if path.is_file() and path.suffix in (".md", ".docx", ".pdf", ".pptx", ".json"):
            files.append(path)
    files += list((FINAL / "figures").glob("*.png"))
    files += [p for p in ANALYSIS.iterdir() if p.is_file() and p.suffix in (".md", ".json", ".csv", ".xml")]
    files += list((ANALYSIS / "figures").glob("*.png"))
    files += [p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts]
    files += list((ROOT / "configs").glob("*.json"))
    files += list((ROOT / "tests").glob("test_bert*.py"))
    files += [ROOT / p for p in ("README.md", "requirements-reporting.txt", "requirements-bert.txt", "pyproject.toml", "uv.lock",
                                "docs/bert_small_training_guide.md", "docs/status.md", "docs/reproduction.md")]
    # Keep public navigation useful offline; exclude unrelated/unreviewed user files.
    files += list((ROOT / "docs").rglob("*.md"))
    files += [ROOT / p for p in ("note.md", "reference.md", "tests/README.md")]
    files += list((ROOT / "tests").glob("*.py"))
    public = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    for name in public:
        path = ROOT / name
        if path.is_file() and name.startswith("reports/") and not any(
                part in {"private", "documentation_archive", "before_layout"} for part in path.parts) and path.suffix in (".md", ".docx", ".pdf", ".pptx", ".png", ".json", ".csv", ".xml"):
            files.append(path)
    files.append(ROOT / "reports/final_evaluation/reproducibility/code_protocol_bundle.zip")
    files = sorted(set(files))
    if any("private" in p.parts or "data_set" in p.parts or p.suffix in (".pt", ".safetensors") for p in files):
        raise ValueError("Private corpus or checkpoint entered public handover")
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    with ZipFile(filename, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files: archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("README_BUNDLE.md", "# NUCLE handover — 28/10/2026\n\nRead reports/summaries/final/handover_checklist.md and the PDF/slides.\nThis bundle contains code and reviewed aggregates, no corpus, private audit text, checkpoints or per-sample predictions. Full training reproduction requires licensed local inputs. Historical source_manifest.json is a snapshot; use delivery_manifest.json for the updated report. Names/contributions/AI declarations need the group's confirmation; no submission is implied.\n")
    with ZipFile(filename) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt handover ZIP")
        for path, expected in manifest.items():
            if hashlib.sha256(archive.read(path)).hexdigest() != expected:
                raise ValueError("ZIP manifest mismatch")
    (FINAL / "NUCLE_delivery_20261028.sha256").write_text(sha(filename) + "\n", encoding="utf-8")
    return {"file": filename.relative_to(ROOT).as_posix(), "files": len(files), "sha256": sha(filename), "no_private_corpus": True}


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    source = read(ANALYSIS / "source_manifest.json")
    protected = {path: (ROOT / path).exists() and sha(ROOT / path) == expected for path, expected in source["inputs"].items()}
    if not all(protected.values()):
        raise ValueError("An analysis input changed while building delivery")
    docx = {stem: docx_checks(stem) for stem in ("bao_cao_tong_ket_NUCLE", "BERT_comparison_report")}
    pdf = {stem: pdf_checks(FINAL / f"{stem}.pdf", stem) for stem in ("bao_cao_tong_ket_NUCLE", "BERT_comparison_report", "NUCLE_presentation")}
    pptx = Presentation(FINAL / "NUCLE_presentation.pptx")
    slide_spec = read(FINAL / "presentation_content.json")
    slide_checks = {"slides": len(pptx.slides), "expected_slides": len(slide_spec),
                    "titles_present": all(spec["title"] in "\n".join(shape.text for shape in slide.shapes if shape.has_text_frame)
                                          for spec, slide in zip(slide_spec, pptx.slides, strict=True)),
                    "notes_present": all(slide.notes_slide.notes_text_frame.text.strip() for slide in pptx.slides)}
    links = check_links()
    test_suite = ET.parse(ANALYSIS / "tests.xml").getroot()
    tests = [{key: suite.attrib.get(key) for key in ("tests", "failures", "errors", "skipped")}
             for suite in test_suite.iter("testsuite")]
    verification = {"date": "2026-10-28", "protected_inputs_unchanged": all(protected.values()),
                    "protected_input_count": len(protected), "docx": docx, "pdf": pdf,
                    "slides": slide_checks, "links": links, "tests": tests,
                    "manual_pending": ["Names/MSSV and actual contribution confirmation", "AI declaration sign-off", "22 independent expert reviews", "Course rubric/submission confirmation"],
                    "rendering_scope": "Every PDF page rendered; DOCX/PPTX structure checked. PDF report from same Markdown, slide PDF from same layout spec; native Word/PowerPoint may paginate differently."}
    editorial_path = ROOT / "reports/maintenance/editorial_verification.json"
    if editorial_path.exists():
        editorial = read(editorial_path)
        verification["editorial_review"] = {key: editorial[key] for key in (
            "all_passed", "protected_evidence_files", "edited_markdown_files", "edited_word_files")}
        if not editorial["all_passed"]:
            raise ValueError("Language edit has unresolved integrity checks")
    if not all(v["tables_exact"] for v in docx.values()) or links["missing"]:
        dump(FINAL / "delivery_verification.json", verification)
        raise ValueError("Document tables or links failed verification")
    if any(v["out_of_page_text"] or v["replacement_glyphs"] for v in pdf.values()):
        dump(FINAL / "delivery_verification.json", verification)
        raise ValueError("PDF text overflow or replacement glyph detected")
    if slide_checks["slides"] != 12 or not slide_checks["titles_present"] or not slide_checks["notes_present"]:
        raise ValueError("Slides incomplete")
    verification["all_automated_checks_pass"] = True
    dump(FINAL / "delivery_verification.json", verification)
    outputs = [p for p in FINAL.rglob("*") if p.is_file() and "documentation_archive" not in p.parts and
               p.name not in ("delivery_manifest.json", "NUCLE_delivery_20261028.zip", "NUCLE_delivery_20261028.sha256")]
    outputs += [p for p in ANALYSIS.rglob("*") if p.is_file() and p.name != "delivery_manifest.json"]
    outputs += [p for p in (ROOT / "reports").rglob("*") if p.is_file() and p.suffix in (".md", ".docx") and
                not any(part in {"documentation_archive", "before_layout", "private"} for part in p.parts)]
    if editorial_path.exists():
        outputs.append(editorial_path)
    inputs = [ROOT / p for p in ("src/reporting/document_export.py", "src/reporting/build_bert_delivery.py",
                                 "src/reporting/build_nucle_slides.py", "src/evaluation/bert_final_analysis.py",
                                 "src/evaluation/bert_audit_review.py", "src/evaluation/verify_bert_delivery.py",
                                 "src/reporting/refine_report_language.py", "src/evaluation/verify_report_language.py", "requirements-reporting.txt")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "report_date": "2026-10-28", "scope": "Current delivery; old source_manifest and locked inputs retained as historical snapshots",
                "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
                "outputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
                "training_input_hashes": source["inputs"], "personal_information_confirmed": False}
    dump(FINAL / "delivery_manifest.json", manifest)
    dump(ANALYSIS / "delivery_manifest.json", manifest)
    bundle = package()
    dump(PREVIEW / "bundle_check.json", bundle)
    print(json.dumps({"all_automated_checks_pass": True, "tables": docx,
                      "pdf_pages": {name: value["pages"] for name, value in pdf.items()},
                      "internal_links": links["checked"], "bundle": bundle}, indent=2))


if __name__ == "__main__":
    main()
