"""Join frozen AI review notes to a private audit packet; never relabel gold."""
import argparse
from collections import Counter
from pathlib import Path

from src.evaluation.bert_final_analysis import ROOT, csv_write, dump, read, sha


def run(folder):
    private = folder / "private"
    packet = read(private / "audit_packet.json")
    notes = read(private / "ai_review.json")
    mapping = {row["annotation_id"]: row for row in notes["notes"]}
    if len(mapping) != len(packet) or set(mapping) != {row["annotation_id"] for row in packet}:
        raise ValueError("Review notes must cover each frozen audit ID exactly once")
    rows = []
    for record in packet:
        note = mapping[record["annotation_id"]]
        rows.append({**{k: record[k] for k in ("annotation_id", "sample_id", "stratum", "sentence", "span", "correction", "gold", "BERT", "LR")},
                     "ai_category": note["category"], "ai_review": note["note"],
                     "ai_contribution": note["contribution"], "ai_confidence": note["confidence"],
                     "review": "", "contribution": "", "human_reviewer": ""})
    if (private / "error_review_with_ai.csv").exists():
        raise FileExistsError("Do not overwrite a possibly edited review CSV")
    csv_write(private / "error_review_with_ai.csv", rows)
    summary = read(folder / "audit_summary.json")
    summary.update(ai_review_completed=True, ai_review_scope=notes["scope"], human_review_status="pending",
                   categories=dict(Counter(row["category"] for row in notes["notes"])),
                   notes_sha256=sha(private / "ai_review.json"),
                   expert_priority_ids=["23037", "34539", "16809", "30366", "32801", "32874"],
                   selection_limit="Round-robin over sorted gold/pred buckets; both-wrong subset contains 9 ArtOrDet and 1 Nn; not a representative prevalence sample")
    dump(folder / "audit_summary.json", summary)
    print("Joined 40 AI-assisted reviews; manual review/contribution columns remain blank.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "reports/bert_final_analysis")
    run(parser.parse_args().analysis.resolve())
