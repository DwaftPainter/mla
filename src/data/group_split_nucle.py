"""Leakage-aware grouping and train/validation/test split for NUCLE v2.

This module stores identifiers and hashes only; it never writes source text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
import unicodedata

try:  # Support both `python src/data/group_split_nucle.py` and package execution.
    from .constants import DATA_ROOT, SGML_FILE, TARGET_LABELS
    from .nucle_sgml import load_nucle
    from .prepare_nucle import canonical, file_hash, require
except ImportError:
    from constants import DATA_ROOT, SGML_FILE, TARGET_LABELS
    from nucle_sgml import load_nucle
    from prepare_nucle import canonical, file_hash, require

PREPARED = DATA_ROOT / "processed" / "nucle_preparation_v2"
DUPLICATES = DATA_ROOT.parent / "reports" / "nucle_duplicate_content.json"
OUTPUT = DATA_ROOT / "split" / "nucle_group_split"
SEED = 20261013
FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
BOILERPLATE_MIN_ESSAYS = 10
BOILERPLATE_MAX_WORDS = 10
BOILERPLATE_MAX_CHARS = 80


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1

    def components(self):
        result = defaultdict(list)
        for item in self.parent:
            result[self.find(item)].append(item)
        return [sorted(members) for members in result.values()]


def content_groups(documents, scope: str, kind: str) -> list[dict]:
    buckets = defaultdict(list)
    for essay_index, document in enumerate(documents):
        if scope == "essay":
            units = [("\n".join(document.paragraphs), -1)]
        else:
            units = [(paragraph, paragraph_index)
                     for paragraph_index, paragraph in enumerate(document.paragraphs)]
        for text, paragraph_index in units:
            if not text.strip():
                continue
            normalized = normalize_text(text)
            value = text if kind == "exact" else normalized
            buckets[sha256_text(value)].append({
                "essay_index": essay_index,
                "essay_id": document.essay_id,
                "paragraph_index": paragraph_index,
            })
    groups = []
    for digest, locations in sorted(buckets.items()):
        essay_indices = sorted({row["essay_index"] for row in locations})
        if len(essay_indices) > 1:
            groups.append({"scope": scope, "kind": kind, "sha256": digest,
                           "essay_indices": essay_indices, "locations": locations})
    return groups


def build_grouping(documents, samples, provenance, source_hash, prep_hash):
    essay_indices = list(range(len(documents)))
    uf = UnionFind(essay_indices)
    decisions = []
    target_paragraphs = {
        (row["essay_index"], row["start_par"])
        for row in provenance if row["status"] == "retained"
    }

    # Exact and normalized complete-essay duplicates are always one split group.
    for kind in ("exact", "normalized"):
        for group in content_groups(documents, "essay", kind):
            members = group["essay_indices"]
            for member in members[1:]:
                uf.union(members[0], member)
            decisions.append({**group, "decision": "group_together",
                              "rule": f"duplicate_{kind}_complete_essay"})

    boilerplate_groups = []
    # Normalized paragraph equality covers exact repeats as well. It also
    # treats NFC, case and whitespace-only variants as the same candidate.
    for group in content_groups(documents, "paragraph", "normalized"):
        locations = group["locations"]
        representative = documents[locations[0]["essay_index"]].paragraphs[locations[0]["paragraph_index"]]
        normalized = normalize_text(representative)
        has_target_annotation = any(
            (row["essay_index"], row["paragraph_index"]) in target_paragraphs
            for row in locations
        )
        frequent_short_boilerplate = (
            len(group["essay_indices"]) >= BOILERPLATE_MIN_ESSAYS
            and word_count(normalized) <= BOILERPLATE_MAX_WORDS
            and len(normalized) <= BOILERPLATE_MAX_CHARS
            and not has_target_annotation
        )
        if frequent_short_boilerplate:
            decision = {**group, "decision": "exclude_as_common_boilerplate",
                        "rule": "shared_by_10_plus_essays_at_most_10_words_80_chars_no_target_annotation",
                        "word_count": word_count(normalized), "normalized_char_count": len(normalized),
                        "target_annotation_present": False}
            boilerplate_groups.append(decision)
            decisions.append(decision)
            continue
        members = group["essay_indices"]
        for member in members[1:]:
            uf.union(members[0], member)
        decisions.append({**group, "decision": "group_together",
                          "rule": "repeated_normalized_paragraph",
                          "word_count": word_count(normalized),
                          "normalized_char_count": len(normalized),
                          "target_annotation_present": has_target_annotation})

    # Exact model input collisions are grouped even if paragraph text falls
    # under the boilerplate exemption. Hashes are verified against tuples.
    input_groups = defaultdict(list)
    for sample in samples:
        identity = (sample["sentence"], sample["error_start"], sample["error_end"])
        input_groups[sample["input_sha256"]].append((sample, identity))
    duplicate_inputs = []
    for digest, members in sorted(input_groups.items()):
        identities = {member[1] for member in members}
        require(len(identities) == 1, "Input hash collision in prepared samples")
        if len(members) > 1:
            essays = sorted({member[0]["essay_index"] for member in members})
            for member in essays[1:]:
                uf.union(essays[0], member)
            duplicate_inputs.append({"scope": "model_input", "sha256": digest,
                                     "essay_indices": essays,
                                     "sample_ids": sorted(member[0]["sample_id"] for member in members),
                                     "decision": "group_together",
                                     "rule": "identical_sentence_and_span"})
            decisions.append(duplicate_inputs[-1])

    components = uf.components()
    essay_to_group = {}
    group_rows = []
    for members in components:
        group_id = "g-" + hashlib.sha256(canonical(members).encode("utf-8")).hexdigest()[:16]
        essay_ids = [documents[index].essay_id for index in members]
        for index in members:
            essay_to_group[index] = group_id
        group_rows.append({"group_id": group_id, "essay_indices": members,
                           "essay_ids": essay_ids, "essay_count": len(members)})
    group_rows.sort(key=lambda row: row["group_id"])

    sample_counts = defaultdict(Counter)
    for sample in samples:
        sample_counts[essay_to_group[sample["essay_index"]]][sample["error_type"]] += 1
    component_rows = []
    for group in group_rows:
        counts = sample_counts[group["group_id"]]
        component_rows.append({**group, "sample_count": sum(counts.values()),
                               "label_counts": {label: counts[label] for label in TARGET_LABELS}})

    policy = {
        "version": "nucle-grouping-v1",
        "source_sha256": source_hash,
        "preparation_manifest_sha256": prep_hash,
        "normalization": "Unicode NFC, casefold, collapse all whitespace runs; no punctuation removal or stemming",
        "essay_rule": "identical full essay text (exact or normalized) connects essays",
        "paragraph_rule": "identical normalized paragraph connects essays unless it meets the boilerplate exemption",
        "boilerplate_exemption": {
            "distinct_essays_at_least": BOILERPLATE_MIN_ESSAYS,
            "normalized_word_count_at_most": BOILERPLATE_MAX_WORDS,
            "normalized_character_count_at_most": BOILERPLATE_MAX_CHARS,
            "target_annotation_in_any_occurrence": "then do not exempt; group together",
            "rationale": "avoid giant components caused by short non-informative repeated text",
        },
        "input_rule": "identical full sentence plus error_start/error_end always connects essays",
        "near_duplicate_rule": "not detected automatically; no semantic paraphrase matching",
        "split_unit": "connected component of essay graph",
        "fractions": FRACTIONS,
        "seed": SEED,
    }
    evidence = {
        "complete_essay_exact_groups": sum(d["rule"] == "duplicate_exact_complete_essay" for d in decisions),
        "complete_essay_normalized_groups": sum(d["rule"] == "duplicate_normalized_complete_essay" for d in decisions),
        "normalized_paragraph_groups": sum(d["scope"] == "paragraph" and d["kind"] == "normalized" for d in decisions),
        "boilerplate_exempt_groups": len(boilerplate_groups),
        "boilerplate_groups_with_target_annotations": sum(d.get("target_annotation_present", False) for d in boilerplate_groups),
        "identical_model_input_groups": len(duplicate_inputs),
        "essays": len(documents),
        "connected_groups": len(group_rows),
        "largest_group_essays": max((row["essay_count"] for row in group_rows), default=0),
        "largest_group_samples": max((row["sample_count"] for row in component_rows), default=0),
    }
    require(evidence["boilerplate_groups_with_target_annotations"] == 0,
            "Boilerplate exemption unexpectedly includes a target-annotated paragraph")
    return policy, decisions, group_rows, component_rows, essay_to_group, evidence, boilerplate_groups


def assignment_objective(assignments, ordered_groups, totals):
    fractions = FRACTIONS
    objective = 0.0
    for partition, fraction in fractions.items():
        count = sum(ordered_groups[g]["sample_count"] for g in ordered_groups
                    if assignments.get(g) == partition)
        objective += ((count - totals["all"] * fraction) / max(totals["all"] * fraction, 1)) ** 2
        for label in TARGET_LABELS:
            target = totals["labels"][label] * fraction
            actual = sum(ordered_groups[g]["label_counts"][label] for g in ordered_groups
                         if assignments.get(g) == partition)
            objective += 2.0 * ((actual - target) / max(target, 1)) ** 2
            if actual == 0 and target > 0:
                objective += 100.0
    return objective


def stratified_group_assignment(component_rows, seed=SEED, restarts=128):
    groups = {row["group_id"]: row for row in component_rows if row["sample_count"] > 0}
    totals = {"all": sum(row["sample_count"] for row in groups.values()),
              "labels": {label: sum(row["label_counts"][label] for row in groups.values())
                         for label in TARGET_LABELS}}
    require(totals["all"] > 0, "No target samples to split")
    rng = random.Random(seed)
    best = None
    for _ in range(restarts):
        jitter = {group_id: rng.random() for group_id in groups}
        def priority(group_id):
            row = groups[group_id]
            present = [totals["labels"][label] / row["label_counts"][label]
                       for label in TARGET_LABELS if row["label_counts"][label] > 0]
            rarity = min(present) if present else float("inf")
            return (rarity, -row["sample_count"], jitter[group_id])
        order = sorted(groups, key=priority)
        assignments = {}
        counts = {part: Counter() for part in FRACTIONS}
        for group_id in order:
            row = groups[group_id]
            choices = list(FRACTIONS)
            rng.shuffle(choices)
            best_part, best_cost = None, None
            for part in choices:
                fraction = FRACTIONS[part]
                proposed = counts[part].copy()
                proposed["all"] += row["sample_count"]
                for label in TARGET_LABELS:
                    proposed[label] += row["label_counts"][label]
                def partition_cost(state):
                    result = ((state["all"] - totals["all"] * fraction)
                              / max(totals["all"] * fraction, 1)) ** 2
                    result += sum(2.0 * ((state[label] - totals["labels"][label] * fraction)
                                         / max(totals["labels"][label] * fraction, 1)) ** 2
                                  for label in TARGET_LABELS)
                    return result
                # Use the change in this partition's loss, so filling an empty
                # validation/test target is rewarded instead of always favoring train.
                cost = partition_cost(proposed) - partition_cost(counts[part])
                if best_cost is None or cost < best_cost:
                    best_part, best_cost = part, cost
            assignments[group_id] = best_part
            counts[best_part]["all"] += row["sample_count"]
            for label in TARGET_LABELS:
                counts[best_part][label] += row["label_counts"][label]
        objective = assignment_objective(assignments, groups, totals)
        missing = sum(counts[part][label] == 0 for part in FRACTIONS for label in TARGET_LABELS)
        candidate = (missing, objective, assignments.copy())
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    require(best is not None, "No split assignment found")
    assignments = best[2]
    # Components without target annotations carry no samples; keep them in train.
    for row in component_rows:
        if row["sample_count"] == 0:
            assignments[row["group_id"]] = "train"
    return assignments, totals, {"restarts": restarts, "missing_class_partitions": best[0],
                                 "objective": best[1]}


def write_json(path: Path, value):
    path.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")


def create_split(source=SGML_FILE, prepared=PREPARED, duplicate_report=DUPLICATES,
                 output=OUTPUT, seed=SEED):
    source, prepared, duplicate_report, output = map(Path, (source, prepared, duplicate_report, output))
    require(not output.exists(), f"Output already exists: {output}; choose a new output directory")
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prep_hash = (prepared / "manifest.sha256").read_text(encoding="ascii").strip()
    require(prep_hash, "Missing preparation manifest checksum")
    source_hash = file_hash(source)
    require(source_hash == manifest["inputs"]["source"]["sha256"], "Source checksum differs from preparation")
    require(duplicate_report.is_file(), f"Duplicate report not found: {duplicate_report}")
    docs = load_nucle(source)
    reported = json.loads(duplicate_report.read_text(encoding="utf-8"))
    expected_duplicate_groups = {
        (row["kind"], row["sha256"], tuple(sorted({loc["essay_index"] for loc in row["locations"]})))
        for row in reported if row["scope"] == "paragraph" and row["kind"] == "normalized"
    }
    actual_duplicate_groups = {
        ("normalized", row["sha256"], tuple(row["essay_indices"]))
        for row in content_groups(docs, "paragraph", "normalized")
    }
    require(actual_duplicate_groups == expected_duplicate_groups,
            "Duplicate report does not match the current source snapshot")
    samples = read_jsonl(prepared / "samples.jsonl")
    provenance = read_jsonl(prepared / "provenance.jsonl")
    require(len(samples) == manifest["counts"]["retained"], "Sample count differs from preparation manifest")
    require(len(provenance) == manifest["counts"]["all"], "Provenance count differs from preparation manifest")
    require(len(docs) == 1397, "Essay inventory differs from expected source snapshot")
    require(all(sample["error_type"] in TARGET_LABELS for sample in samples), "Unexpected target label")
    require(len({sample["sample_id"] for sample in samples}) == len(samples), "Duplicate sample IDs")

    policy, decisions, essay_groups, component_rows, essay_to_group, evidence, boilerplate = build_grouping(
        docs, samples, provenance, source_hash, prep_hash)
    policy["seed"] = seed
    assignments, totals, split_quality = stratified_group_assignment(component_rows, seed=seed)

    essay_partitions = {}
    for row in essay_groups:
        part = assignments[row["group_id"]]
        for essay_index in row["essay_indices"]:
            essay_partitions[essay_index] = part
    sample_rows = []
    for sample in samples:
        sample_rows.append({"sample_id": sample["sample_id"],
                            "annotation_id": sample["annotation_id"],
                            "essay_id": sample["essay_id"],
                            "essay_index": sample["essay_index"],
                            "group_id": essay_to_group[sample["essay_index"]],
                            "partition": essay_partitions[sample["essay_index"]],
                            "error_type": sample["error_type"]})
    require(len(sample_rows) == len(samples), "Lost sample during split assignment")
    require(len({row["sample_id"] for row in sample_rows}) == len(sample_rows), "Sample assigned more than once")
    require(len({row["annotation_id"] for row in sample_rows}) == len(sample_rows), "Annotation assigned more than once")
    require(all(row["partition"] in FRACTIONS for row in sample_rows), "Invalid partition")
    group_partitions = defaultdict(set)
    for row in sample_rows:
        group_partitions[row["group_id"]].add(row["partition"])
    require(all(len(parts) == 1 for parts in group_partitions.values()), "A group crosses partitions")

    by_partition = {part: [row for row in sample_rows if row["partition"] == part] for part in FRACTIONS}
    summary = {"policy_version": policy["version"], "seed": seed,
               "source_sha256": source_hash, "preparation_manifest_sha256": prep_hash,
               "grouping_evidence": evidence, "split_quality": split_quality,
               "target_samples": len(sample_rows), "partitions": {},
               "target_fractions": FRACTIONS,
               "boilerplate_rule_applied_to": len(boilerplate)}
    for part, rows in by_partition.items():
        counts = Counter(row["error_type"] for row in rows)
        summary["partitions"][part] = {
            "samples": len(rows), "fraction": len(rows) / len(sample_rows),
            "essays": len({row["essay_index"] for row in rows}),
            "groups": len({row["group_id"] for row in rows}),
            "label_counts": {label: counts[label] for label in TARGET_LABELS},
        }
    require(0.65 <= summary["partitions"]["train"]["fraction"] <= 0.75,
            "Train fraction outside the 65–75% balance tolerance; split not locked")
    require(all(0.12 <= summary["partitions"][part]["fraction"] <= 0.18
                for part in ("validation", "test")),
            "Validation/test fraction outside the 12–18% balance tolerance; split not locked")
    require(all(summary["partitions"][part]["label_counts"][label] > 0
                for part in FRACTIONS for label in TARGET_LABELS),
            "At least one partition lacks a target class; do not lock this split")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    write_json(staging / "grouping_policy.json", policy)
    write_json(staging / "grouping_summary.json", {**evidence, "boilerplate_groups": boilerplate,
                                                   "split_quality": split_quality})
    with (staging / "grouping_decisions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in decisions:
            stream.write(canonical(row) + "\n")
    with (staging / "essay_groups.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("essay_id,essay_index,group_id,partition\n")
        for essay in range(len(docs)):
            group_id = essay_to_group[essay]
            essay_id = docs[essay].essay_id.replace('"', '""')
            stream.write(f'"{essay_id}",{essay},"{group_id}",{essay_partitions[essay]}\n')
    with (staging / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("sample_id,annotation_id,essay_id,essay_index,group_id,partition,error_type\n")
        for row in sorted(sample_rows, key=lambda x: x["annotation_id"]):
            essay_id = row["essay_id"].replace('"', '""')
            stream.write(f'"{row["sample_id"]}",{row["annotation_id"]},"{essay_id}",{row["essay_index"]},"{row["group_id"]}",{row["partition"]},{row["error_type"]}\n')
    write_json(staging / "split_summary.json", summary)
    hashes = {path.name: file_hash(path) for path in sorted(staging.iterdir())}
    write_json(staging / "manifest.json", {"policy_version": policy["version"], "created_from": {
        "preparation_manifest_sha256": prep_hash, "source_sha256": source_hash,
        "duplicate_report_sha256": file_hash(duplicate_report)},
        "seed": seed, "output_sha256": hashes})
    output = Path(output).resolve()
    require(not output.exists(), "Output directory appeared during split generation")
    staging.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SGML_FILE)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--duplicate-report", type=Path, default=DUPLICATES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(create_split(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
