"""Reproduce the evaluated pipeline in a separate directory, without changing the lock.

Run using .cache/final_replay_env/Scripts/python.exe after installing the recorded
runtime versions. Historical annotation-review decisions and selected configs are
fixed inputs, not recreated judgments. This does not rerun the historical BERT or
six-variant hyperparameter searches; it reruns the 52 fits of the evaluation batch.
"""
from pathlib import Path
import csv
import importlib.metadata as metadata
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.data.constants import SGML_FILE, ANNOTATIONS_FILE
from src.data.nucle_sgml import load_nucle
from src.data.parse_nucle import documents_to_dataframe
from src.data.data_quality import audit_quality, compare_csv
from src.data.prepare_nucle import prepare, DEFAULT_AUDIT, DEFAULT_REVIEW
from src.data.group_split_nucle import create_split
from src.features import nucle_features
from src.features.nucle_pos import load_tagger, align_span
from src.training import nucle_final_selection as selection
from src.evaluation import nucle_final_evaluation as evaluation

BASE = ROOT / '.cache/evaluation_pipeline_reproduction'
EVIDENCE = ROOT / 'reports/evaluation_followup'
ORIGINAL = ROOT / 'reports/final_evaluation'
read, dump, sha = selection.read, selection.dump, selection.sha


def exact(new, old):
    same = sha(new) == sha(old)
    if not same:
        raise AssertionError(f'Content differs: {new} vs {old}')
    return {'sha256': sha(new), 'byte_identical': True}


def main():
    lock = evaluation.verify_lock()
    assert Path(sys.prefix).resolve() == (ROOT / '.cache/final_replay_env').resolve()
    assert not BASE.exists(), f'Use a new output directory; already exists: {BASE}'
    BASE.mkdir(parents=True)
    EVIDENCE.mkdir(exist_ok=True)
    started = time.perf_counter()
    result = {'started_at': selection.now(), 'original_lock_sha256': sha(ORIGINAL / 'lock.json'),
              'python': platform.python_version(), 'python_executable': sys.executable,
              'scope': 'Raw SGML -> Understanding/audit -> preparation v2 -> grouping/split -> train-only features -> 52 evaluation-batch fits -> test prediction replay',
              'not_reproduced': ['Independent human annotation adjudication', 'Historical six-variant grid search', 'BERT experiments'],
              'fixed_historical_inputs': ['Annotation overlap review log', 'Historical selected configs and validation predictions'],
              'versions': {d.metadata['Name']: d.version for d in metadata.distributions()}, 'stages': {}}
    dump(EVIDENCE / 'pipeline_reproduction.json', result)

    def record(name, details):
        result['stages'][name] = details
        result['elapsed_seconds'] = time.perf_counter() - started
        dump(EVIDENCE / 'pipeline_reproduction.json', result)
        print('PASS', name, json.dumps(details, ensure_ascii=False), flush=True)

    print('Reconstructing Understanding and quality audit from raw SGML', flush=True)
    docs = load_nucle(SGML_FILE)
    frame = documents_to_dataframe(docs)
    annotations = BASE / 'nucle_errors.csv'
    frame.to_csv(annotations, index=False)
    check = compare_csv(frame, ANNOTATIONS_FILE)
    assert check['status'] == 'match'
    summary, audit, duplicates = audit_quality(docs, frame, annotations)
    audit_path = BASE / 'nucle_annotation_audit.csv'
    audit.to_csv(audit_path, index=False)
    duplicate_path = BASE / 'nucle_duplicate_content.json'
    duplicate_path.write_text(json.dumps(duplicates, indent=2), encoding='utf-8')
    assert read(duplicate_path) == read(ROOT / 'reports/nucle_duplicate_content.json')
    record('source_and_audit', {'annotations': len(frame), 'essays': len(docs),
                              'csv_comparison': check, 'audit': exact(audit_path, DEFAULT_AUDIT),
                              'duplicate_report_semantically_identical': True,
                              'review_log_sha256': sha(DEFAULT_REVIEW)})

    prep = BASE / 'preparation_v2'
    print('Rebuilding preparation v2', flush=True)
    prepare(annotations=annotations, audit=audit_path, output=prep)
    prep_files = {}
    for name in read(selection.PREP / 'manifest.json')['outputs']:
        prep_files[name] = exact(prep / name, selection.PREP / name)
    record('preparation_v2', {'data_outputs': prep_files,
                            'manifest_differs_as_expected': 'New time, code inventory and local input paths; source/data outputs compared directly'})

    split = BASE / 'group_split'
    print('Recomputing connected groups and split with seed 20261013', flush=True)
    split_summary = create_split(prepared=prep, duplicate_report=duplicate_path, output=split, seed=20261013)
    record('group_split', {'samples': split_summary['target_samples'],
                          'split_manifest': exact(split / 'split_manifest.csv', selection.SPLIT / 'split_manifest.csv'),
                          'essay_groups': exact(split / 'essay_groups.csv', selection.SPLIT / 'essay_groups.csv'),
                          'partitions': split_summary['partitions']})

    feat = BASE / 'features'
    print('Reparsing all prepared sentences and fitting train-only feature encoder', flush=True)
    fm = nucle_features.run(prepared_dir=prep, split_dir=split, output_dir=feat)
    record('features', {'samples': fm['sample_count'], 'dimensions': fm['feature_dimension'],
                        'sentences_parsed': fm['unique_sentences_parsed'],
                        'files': {n: exact(feat / n, selection.FEAT / n) for n in fm['outputs']}})

    # Only path globals change in memory. Frozen module files stay byte-identical.
    selection.PREP, selection.SPLIT, selection.FEAT = prep, split, feat
    selection.OUT = BASE / 'selection'
    print('Reproducing exactly 40 new-grid fits + 12 historical selected-config refits', flush=True)
    selection.run()
    original_selection = read(ORIGINAL / 'selection_summary.json')
    new_selection = read(selection.OUT / 'selection_summary.json')
    checks = {}
    for old, new in zip(original_selection['selected'], new_selection['selected'], strict=True):
        for key in ['id', 'variant', 'configuration', 'feature_count', 'keep', 'validation_macro_f1', 'validation_accuracy']:
            assert new[key] == old[key], (new['id'], key)
        a = torch.load(ROOT / old['checkpoint'], map_location='cpu', weights_only=True)
        b = torch.load(ROOT / new['checkpoint'], map_location='cpu', weights_only=True)
        same = all(torch.equal(a[k], b[k]) for k in a) if new['model'] == 'LR' else a == b
        assert same, new['id']
        checks[new['id']] = {'selected_config_and_validation_identical': True, 'checkpoint_content_identical': True}
    original_fits = read(ORIGINAL / 'fit_progress.json')
    new_fits = read(selection.OUT / 'fit_progress.json')
    assert len(original_fits) == len(new_fits) == 52
    for old, new in zip(original_fits, new_fits, strict=True):
        for key in ['variant', 'model', 'configuration', 'macro_f1', 'accuracy', 'convergence']:
            assert old[key] == new[key], (new['variant'], new['model'], key)
    record('training', {'fits': 52, 'new_grid': 40, 'historical_refits': 12,
                        'validation_predictions': exact(selection.OUT / 'validation_predictions.csv', ORIGINAL / 'validation_predictions.csv'),
                        'all_fit_metrics_and_termination_identical': True, 'selected': checks})

    print('Replaying test predictions using newly trained models and regenerated features', flush=True)
    test_meta, saved_predictions = evaluation.load_predictions()
    ids = [r['sample_id'] for r in test_meta]
    stored = {}
    with (feat / 'features.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['partition'] == 'test':
                stored[row['sample_id']] = {int(k): v for k, v in row['x'].items()}
    wanted = set(ids)
    samples = {r['sample_id']: r for r in map(json.loads, (prep / 'samples.jsonl').read_text(encoding='utf-8').splitlines()) if r['sample_id'] in wanted}
    texts = {r['sentence_id']: r['sentence'] for r in samples.values()}
    nlp = load_tagger()
    parsed = dict(zip(texts, nlp.pipe(texts.values(), batch_size=256), strict=True))
    mask = selection.load_encoder(selection.OUT / 'masked_encoder.json')
    pos = read(selection.OUT / 'pos_rule.json')
    full_rows = [stored[s] for s in ids]
    masked_rows, pos_predictions = [], []
    for sid in ids:
        s = samples[sid]
        doc = parsed[s['sentence_id']]
        masked_rows.append(mask.transform_one(selection.masked_features(s, doc)))
        sig = align_span(doc, s['error_start'], s['error_end'])['pos_signature']
        pos_predictions.append(pos['fixed'].get(sig, pos['lookup'].get(sig, pos['majority'])))
    replay = {'Majority': [pos['majority']] * len(ids), 'POS': pos_predictions}
    for entry in new_selection['selected']:
        replay[entry['id']] = evaluation.predict(entry, masked_rows if entry['variant'] == selection.MASK else full_rows)
    disagreements = {name: sum(a != b for a, b in zip(values, saved_predictions[name], strict=True)) for name, values in replay.items()}
    assert not any(disagreements.values()), disagreements
    rebuilt_metrics, _ = evaluation.subset_metrics(test_meta, replay)
    assert rebuilt_metrics == read(ORIGINAL / 'test_metrics.json')
    record('test_replay', {'samples': len(ids), 'pipeline_names': len(replay),
                           'disagreements': disagreements, 'all_saved_slice_metrics_identical': True,
                           'interpretation': 'Reproduction only; not a new independent test or model selection'})
    evaluation.verify_lock()
    result['complete'] = True
    result['finished_at'] = selection.now()
    result['elapsed_seconds'] = time.perf_counter() - started
    dump(EVIDENCE / 'pipeline_reproduction.json', result)
    print('REPRODUCTION COMPLETE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
