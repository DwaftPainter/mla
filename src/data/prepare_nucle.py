"""Build known-span samples without correction features or dataset splitting.

Run: python src/data/prepare_nucle.py
Policy: docs/preparation.md (v2 by default; --policy v1 reproduces old boundaries).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import pandas as pd
import spacy

try:  # Support both `python src/data/prepare_nucle.py` and package execution.
    from .constants import ANNOTATIONS_FILE, DATA_ROOT, PROJECT_ROOT, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from .boundary_challenges import BOUNDARY_CHALLENGES
    from .data_quality import annotation_review, compare_csv
    from .nucle_sgml import load_nucle
    from .parse_nucle import documents_to_dataframe
    from .sentence_boundaries import repaired_starts, rule_metadata
except ImportError:  # Direct-file execution sets sys.path to src/data.
    from constants import ANNOTATIONS_FILE, DATA_ROOT, PROJECT_ROOT, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from boundary_challenges import BOUNDARY_CHALLENGES
    from data_quality import annotation_review, compare_csv
    from nucle_sgml import load_nucle
    from parse_nucle import documents_to_dataframe
    from sentence_boundaries import repaired_starts, rule_metadata

POLICY = "nucle-preparation-v2"
SPACY_VERSION = "3.8.7"
SOURCE_SHA256 = "8441a60060730bdc6af1e79a064c68441ecb24a66da91ea2aab0061a64a130ac"
PIPELINE_CONFIG = {"punct_chars": [".", "!", "?", "…"], "overwrite": True}
V1_OUTPUT = DATA_ROOT / "processed" / "nucle_preparation_v1"
DEFAULT_OUTPUT = DATA_ROOT / "processed" / "nucle_preparation_v2"
DEFAULT_AUDIT = REPORTS_DIR / "nucle_annotation_audit.csv"
DEFAULT_REVIEW = DATA_ROOT / "interim" / "nucle_overlap_review_log_all.csv"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_nlp():
    require(spacy.__version__ == SPACY_VERSION, f"Required spaCy {SPACY_VERSION}, got {spacy.__version__}")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer", config=PIPELINE_CONFIG)
    return nlp


def segment_paragraph(paragraph, nlp, policy=POLICY):
    """Partition source text exactly; sentence cuts may divide a glued token."""
    nlp.max_length = max(1_000_000, len(paragraph) + 1)
    require(policy in {"nucle-preparation-v1", POLICY}, "Unknown preparation policy")
    doc = nlp(paragraph)
    starts = [next((t.idx for t in sent if not t.is_space), None) for sent in doc.sents]
    starts = [start for start in starts if start is not None]
    if not starts:
        require(not paragraph.strip(), "Tokenizer lost non-whitespace text")
        return []
    boundaries = ([0, *starts[1:]] if policy == "nucle-preparation-v1" else repaired_starts(doc)) + [len(paragraph)]
    tokens = [t for t in doc if not t.is_space]
    result, cursor = [], 0
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        require(start < end, "Empty sentence interval")
        sentence = paragraph[start:end]
        aligned = []
        while cursor < len(tokens) and tokens[cursor].idx < end:
            token = tokens[cursor]
            token_end = token.idx + len(token.text)
            if token_end <= start:
                cursor += 1
                continue
            fragment_start, fragment_end = max(start, token.idx), min(end, token_end)
            a, b = fragment_start - start, fragment_end - start
            fragment = paragraph[fragment_start:fragment_end]
            require(0 <= a < b <= len(sentence) and sentence[a:b] == fragment,
                    "Token fragment differs from source")
            aligned.append({"index": len(aligned), "text": fragment, "start": a, "end": b})
            if token_end <= end:
                cursor += 1
            else:
                break
        require(bool(aligned), "Sentence has no non-whitespace token")
        result.append({"sentence_index": index, "sentence_start_par": start,
                       "sentence_end_par": end, "sentence": sentence, "tokens": aligned})
    require(cursor == len(tokens), "Unassigned tokens")
    require("".join(s["sentence"] for s in result) == paragraph, "Sentence round-trip failed")
    return result


def construct_input(paragraph, start, end, sentences):
    """Return (input, assigned sentence, rejection reason), independent of labels."""
    require(0 <= start <= end <= len(paragraph), "Invalid offsets passed to construct_input")
    if not sentences:
        return None, None, "empty_paragraph"
    insertion = start == end
    if insertion and start == len(paragraph):
        sent = sentences[-1]
    else:
        sent = next((s for s in sentences if s["sentence_start_par"] <= start < s["sentence_end_par"]
                     and end <= s["sentence_end_par"]), None)
    if sent is None:
        return None, None, "cross_sentence"
    u, v = start - sent["sentence_start_par"], end - sent["sentence_start_par"]
    text, tokens = sent["sentence"], sent["tokens"]
    require(0 <= u <= v <= len(text) and text[u:v] == paragraph[start:end], "Span round-trip failed")
    cuts_start = any(t["start"] < u < t["end"] for t in tokens)
    cuts_end = any(t["start"] < v < t["end"] for t in tokens)
    indices = [] if insertion else [t["index"] for t in tokens if t["start"] < v and u < t["end"]]
    if insertion:
        if cuts_start:
            return None, sent, "insertion_inside_token"
        alignment = "insertion_boundary"
    elif not indices:
        return None, sent, "no_token_overlap"
    elif cuts_start or cuts_end:
        alignment = "partial_token"
    elif u == tokens[indices[0]]["start"] and v == tokens[indices[-1]]["end"]:
        alignment = "exact"
    else:
        alignment = "whitespace_edge"
    left = [t["index"] for t in tokens if t["end"] <= u][-2:]
    right = [t["index"] for t in tokens if t["start"] >= v][:2]
    require(not (set(indices) & (set(left) | set(right))) and not set(left) & set(right),
            "Context and span overlap")
    return {"sentence": text, "error_start": u, "error_end": v, "error_span": text[u:v],
            "span_kind": "insertion" if insertion else "non_empty", "alignment": alignment,
            "cuts_start_token": cuts_start, "cuts_end_token": cuts_end,
            "span_token_indices": indices, "left_token_indices": left, "right_token_indices": right,
            "input_sha256": digest(canonical([text, u, v]).encode("utf-8"))}, sent, None


def read_metadata(frame, audit_path, review_path, source_hash):
    """Recompute audit to reject stale metadata; validate review identity and source."""
    expected, flags = annotation_review(frame)
    require(compare_csv(expected, audit_path)["status"] == "match", "Audit CSV differs from current source audit")
    audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    require(not audit.annotation_id.duplicated().any(), "Duplicate audit annotation IDs")
    by_id = {int(r.annotation_id): sorted(f for f in flags if getattr(r, f) == "True")
             for r in audit.itertuples()}
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    required = {"annotation_id", "essay_id", "error_type", "source_sha256", "decision", "data_action"}
    require(required <= set(review.columns), "Missing review columns")
    require(not review.annotation_id.duplicated().any(), "Duplicate review annotation IDs")
    rows = frame.set_index("annotation_id")
    decisions = {}
    for row in review.itertuples():
        aid = int(row.annotation_id)
        require(aid not in decisions and aid in rows.index, "Unknown or duplicate review ID")
        source = rows.loc[aid]
        require(row.source_sha256 == source_hash, "Review source checksum mismatch")
        require(row.essay_id == source.essay_id and row.error_type == source.error_type, "Review identity mismatch")
        require(row.decision in {"retain_original", "unresolved"}, "Unsupported review decision")
        require(row.data_action == "keep_original_annotation_unchanged", "Review requests source changes; policy must be updated")
        decisions[aid] = row.decision
    return by_id, decisions


def build_dataset(documents, frame, quality_flags, decisions, source_hash, nlp, policy=POLICY):
    require(not frame.annotation_id.duplicated().any(), "Duplicate input annotation IDs")
    require(set(quality_flags) == set(frame.annotation_id), "Incomplete quality metadata")
    samples, provenance, sentences, boundary_challenges = [], [], {}, []
    cache = {}
    for row in frame.itertuples(index=False):
        target = row.error_type in TARGET_LABELS
        record = {name: getattr(row, name) for name in
                  ("annotation_id", "essay_id", "essay_index", "mistake_index", "start_par", "end_par", "start_off", "end_off")}
        record.update(source_sha256=source_hash, policy_version=policy, original_label=row.error_type,
                      mapping="identity" if target else None, status="not_constructible", reason=None,
                      sample_id=None, sentence_id=None, sentence_start_par=None, sentence_end_par=None,
                      quality_flags=quality_flags[row.annotation_id], review_status=decisions.get(row.annotation_id, "not_reviewed"))
        if policy == POLICY:
            challenge = BOUNDARY_CHALLENGES.get(row.annotation_id)
            record.update(boundary_flags=[challenge["flag"]] if challenge else [],
                          boundary_review_status=challenge["review_status"] if challenge else "not_flagged")
        item, sent = None, None
        if not target:
            record.update(status="out_of_scope", reason="label_out_of_scope")
        elif not row.valid_offset:
            record["reason"] = "invalid_offset"
        elif row.cross_paragraph:
            record["reason"] = "cross_paragraph"
        else:
            key = (row.essay_index, row.start_par)
            paragraph = documents[row.essay_index].paragraphs[row.start_par]
            require(paragraph == row.paragraph, "Frame paragraph differs from source")
            if key not in cache:
                cache[key] = segment_paragraph(paragraph, nlp, policy)
            item, sent, record["reason"] = construct_input(paragraph, row.start_off, row.end_off, cache[key])
        if sent is not None:
            sid = f"{source_hash}:{row.essay_index}:{row.start_par}:{sent['sentence_index']}"
            record.update(sentence_id=sid, sentence_start_par=sent["sentence_start_par"],
                          sentence_end_par=sent["sentence_end_par"])
        if item is not None:
            require(item["error_span"] == row.error_span, "Sample span differs from Understanding")
            record.update(status="retained", sample_id=f"{source_hash}:{row.annotation_id}")
            item.update(sample_id=record["sample_id"], annotation_id=row.annotation_id,
                        essay_id=row.essay_id, essay_index=row.essay_index, sentence_id=sid, error_type=row.error_type)
            if policy == POLICY:
                challenge = BOUNDARY_CHALLENGES.get(row.annotation_id)
                item.update(boundary_flags=[challenge["flag"]] if challenge else [],
                            boundary_review_status=challenge["review_status"] if challenge else "not_flagged")
                if challenge:
                    require(challenge["review_status"] == "unresolved", "Unsupported challenge status")
                    paragraph = documents[row.essay_index].paragraphs[row.start_par]
                    paragraph_sentences = cache[(row.essay_index, row.start_par)]
                    boundary_challenges.append({
                        "challenge_id": f"nucle-boundary-{row.annotation_id}",
                        "annotation_id": row.annotation_id,
                        "essay_id": row.essay_id,
                        "essay_index": row.essay_index,
                        "paragraph_index": row.start_par,
                        "source_sha256": source_hash,
                        "boundary_flags": [challenge["flag"]],
                        "boundary_review_status": challenge["review_status"],
                        "expectation": challenge["expectation"],
                        "annotation_start_par": row.start_off,
                        "annotation_end_par": row.end_off,
                        "annotation_span": row.error_span,
                        "paragraph": paragraph,
                        "expected_sentence_id": sid,
                        "expected_sentence_start_par": sent["sentence_start_par"],
                        "expected_sentence_end_par": sent["sentence_end_par"],
                        "expected_sentence": item["sentence"],
                        "expected_paragraph_sentences": [
                            {"start": s["sentence_start_par"], "end": s["sentence_end_par"], "text": s["sentence"]}
                            for s in paragraph_sentences
                        ],
                    })
            samples.append(item)
            sentences[sid] = {**sent, "sentence_id": sid, "essay_index": row.essay_index, "paragraph_index": row.start_par}
        provenance.append(record)
    sentence_rows = sorted(sentences.values(), key=lambda s: (s["essay_index"], s["paragraph_index"], s["sentence_index"]))
    summary, duplicate_groups = summarize(samples, provenance, policy)
    require(len(provenance) == len(frame), "Lost provenance rows")
    require(summary["counts"]["target"] == sum(frame.is_target_label), "Target accounting mismatch")
    require(len({s["sample_id"] for s in samples}) == len(samples), "Duplicate sample IDs")
    require(all(s["sentence_id"] in sentences for s in samples), "Missing sentence reference")
    if policy == POLICY:
        require(len({c["annotation_id"] for c in boundary_challenges}) == len(boundary_challenges),
                "Duplicate boundary challenge case")
    result = {"samples": samples, "sentences": sentence_rows, "provenance": provenance,
            "rejected": [r for r in provenance if r["status"] != "retained"],
            "input_groups": duplicate_groups, "summary": summary}
    if policy == POLICY:
        result["boundary_challenges"] = sorted(boundary_challenges, key=lambda c: c["annotation_id"])
    return result


def summarize(samples, provenance, policy=POLICY):
    counts = Counter(r["status"] for r in provenance)
    target = [r for r in provenance if r["status"] != "out_of_scope"]
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["input_sha256"]].append(sample)
    duplicates = []
    for key, members in sorted(groups.items()):
        require(len({(s["sentence"], s["error_start"], s["error_end"]) for s in members}) == 1,
                "Input hash collision")
        if len(members) > 1:
            labels = Counter(s["error_type"] for s in members)
            duplicates.append({"input_sha256": key, "annotation_ids": [s["annotation_id"] for s in members],
                               "essay_indices": sorted({s["essay_index"] for s in members}),
                               "label_counts": dict(sorted(labels.items())), "ambiguous": len(labels) > 1})
    per_label = {}
    for label in TARGET_LABELS:
        rows = [r for r in target if r["original_label"] == label]
        rejected = sum(r["status"] == "not_constructible" for r in rows)
        per_label[label] = {"target": len(rows), "retained": len(rows) - rejected,
                            "not_constructible": rejected, "rejection_rate": rejected / len(rows) if rows else 0,
                            "reasons": dict(Counter(r["reason"] for r in rows if r["reason"]))}
    summary = {"policy_version": policy,
               "counts": {"all": len(provenance), "target": len(target), "out_of_scope": counts["out_of_scope"],
                          "retained": counts["retained"], "not_constructible": counts["not_constructible"]},
               "per_label": per_label,
               "reasons": dict(Counter(r["reason"] for r in target if r["reason"])),
               "alignment": dict(Counter(s["alignment"] for s in samples)),
               "span_kind": dict(Counter(s["span_kind"] for s in samples)),
               "quality_flags": {}, "review_status": {},
               "input_duplicates": {"groups": len(duplicates), "annotations": sum(len(g["annotation_ids"]) for g in duplicates),
                                    "ambiguous_groups": sum(g["ambiguous"] for g in duplicates),
                          "ambiguous_annotations": sum(len(g["annotation_ids"]) for g in duplicates if g["ambiguous"])}}
    if policy == POLICY:
        summary["boundary_flags"] = dict(Counter(flag for r in provenance for flag in r.get("boundary_flags", [])))
        summary["boundary_review_status"] = dict(Counter(
            r.get("boundary_review_status", "not_flagged") for r in provenance if r["status"] == "retained"))
    for name, rows in (("target", target), ("retained", [r for r in target if r["status"] == "retained"]),
                       ("not_constructible", [r for r in target if r["status"] == "not_constructible"])):
        summary["quality_flags"][name] = dict(Counter(f for r in rows for f in r["quality_flags"]))
        summary["review_status"][name] = dict(Counter(r["review_status"] for r in rows))
    require(len(target) == len(samples) + counts["not_constructible"], "Sample accounting failed")
    return summary, duplicates


def write_json(path, value):
    path.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")


def export_dataset(output, dataset, manifest, input_hashes):
    """Publish a complete new directory; never replace an existing dataset."""
    output = Path(output).resolve()
    require(not output.exists(), f"Output already exists: {output}; use --output with a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Leave a failed staging directory for diagnosis, avoiding recursive cleanup.
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    for name, rows in dataset.items():
        if name == "summary":
            write_json(staging / "summary.json", rows)
        else:
            with (staging / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(canonical(row) + "\n")
    manifest = {**manifest, "outputs": {p.name: file_hash(p) for p in sorted(staging.iterdir())}}
    write_json(staging / "manifest.json", manifest)
    (staging / "manifest.sha256").write_text(digest(canonical(manifest).encode("utf-8")) + "\n", encoding="ascii")
    for path, expected in input_hashes.items():
        require(file_hash(path) == expected, f"Input changed during run: {path}")
    require(not output.exists(), "Output appeared during run")
    staging.rename(output)
    return manifest


def prepare(source=SGML_FILE, annotations=ANNOTATIONS_FILE, audit=DEFAULT_AUDIT,
            review=DEFAULT_REVIEW, output=None, expected_source_sha256=SOURCE_SHA256, policy=POLICY):
    require(policy in {"nucle-preparation-v1", POLICY}, "Unknown preparation policy")
    output = Path(output) if output is not None else (V1_OUTPUT if policy == "nucle-preparation-v1" else DEFAULT_OUTPUT)
    require(not Path(output).exists(), f"Output already exists: {output}; use --output with a new directory")
    inputs = {name: Path(path).resolve() for name, path in
              (("source", source), ("annotations", annotations), ("audit", audit), ("review", review),
               ("environment_lock", PROJECT_ROOT / "uv.lock"))}
    hashes = {path: file_hash(path) for path in inputs.values()}
    code_paths = sorted((PROJECT_ROOT / "src").rglob("*.py")) + [PROJECT_ROOT / "pyproject.toml"]
    code_hashes = {path: file_hash(path) for path in code_paths}
    source_hash = hashes[inputs["source"]]
    require(source_hash == expected_source_sha256, "Source checksum mismatch")
    nlp = make_nlp()
    tokenizer_hash = digest(nlp.tokenizer.to_bytes())
    documents = load_nucle(source)
    frame = documents_to_dataframe(documents)
    require(compare_csv(frame, annotations)["status"] == "match", "Understanding CSV differs from SGML")
    flags, decisions = read_metadata(frame, audit, review, source_hash)
    dataset = build_dataset(documents, frame, flags, decisions, source_hash, nlp, policy)
    if policy == POLICY and source_hash == SOURCE_SHA256:
        require({c["annotation_id"] for c in dataset["boundary_challenges"]} == set(BOUNDARY_CHALLENGES),
                "Required boundary challenge case missing or not constructible")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"policy_version": policy, "schema_version": 2 if policy == POLICY else 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "inputs": {name: {"path": str(path), "sha256": hashes[path]} for name, path in inputs.items()},
                "code_commit": commit, "code_sha256": {str(p.relative_to(PROJECT_ROOT)): h for p, h in code_hashes.items()},
                "python_version": platform.python_version(), "spacy_version": spacy.__version__,
                "pipeline": {"language": "en", "components": ["sentencizer"], "config": PIPELINE_CONFIG,
                             "tokenizer_sha256": tokenizer_hash,
                             "boundary_rules": rule_metadata() if policy == POLICY else None},
                "usage_authorization": "user_confirmed", "authorization_date": "2026-10-12",
                "counts": dataset["summary"]["counts"]}
    export_dataset(output, dataset, manifest, {**hashes, **code_hashes})
    return dataset["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("source", SGML_FILE), ("annotations", ANNOTATIONS_FILE),
                          ("audit", DEFAULT_AUDIT), ("review", DEFAULT_REVIEW), ("output", None)):
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--expected-source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--policy", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    args.policy = f"nucle-preparation-{args.policy}"
    args.output = args.output or (V1_OUTPUT if args.policy.endswith("v1") else DEFAULT_OUTPUT)
    summary = prepare(**vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
