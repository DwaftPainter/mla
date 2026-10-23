"""Independent arithmetic/report review of saved evaluation evidence; no model fitting."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/final_evaluation'
REVIEW = ROOT / 'reports/evaluation_followup'
LABELS = ['ArtOrDet', 'Prep', 'Nn', 'Vform', 'SVA']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12), (a, b)


def macro(c, labels):
    result = []
    for k in labels:
        denominator = c[..., k, :].sum(-1) + c[..., :, k].sum(-1)
        result.append(np.divide(2 * c[..., k, k], denominator,
                               out=np.zeros_like(denominator, dtype=float), where=denominator != 0))
    return np.mean(result, axis=0)


def main():
    REVIEW.mkdir(exist_ok=True)
    lock = read(OUT / 'lock.json')
    assert digest(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for name, expected in lock['hashes'].items():
        assert digest(ROOT / name) == expected, name
    immutable = {name: digest(OUT / name) for name in ['lock.json', 'lock.sha256', 'selection_summary.json',
                 'test_predictions.csv', 'test_metrics.json', 'bootstrap.json', 'manual_error_review.json', 'verification.json']}
    data = rows(OUT / 'test_predictions.csv')
    metrics, boot = read(OUT / 'test_metrics.json'), read(OUT / 'bootstrap.json')
    meta = ['sample_id', 'annotation_id', 'essay_id', 'error_type', 'group_id', 'span_tokens', 'span_bin', 'primary_upos', 'pos_signature', 'alignment']
    models = [k for k in data[0] if k not in meta]
    truth = np.array([LABELS.index(r['error_type']) for r in data])
    pred = {m: np.array([LABELS.index(r[m]) for r in data]) for m in models}
    split = rows(ROOT / 'data_set/split/nucle_group_split/split_manifest.csv')
    selected = {r['sample_id']: r for r in split if r['partition'] == 'test'}
    assert set(selected) == {r['sample_id'] for r in data} and len(data) == 2368
    for r in data:
        for key in ['annotation_id', 'essay_id', 'error_type', 'group_id']:
            assert selected[r['sample_id']][key] == r[key]
    for key in ['essay_index', 'group_id']:
        parts = defaultdict(set)
        for r in split:
            parts[r[key]].add(r['partition'])
        assert all(len(v) == 1 for v in parts.values())
    definitions = {'all': (list(range(len(data))), list(range(5)))}
    for name in ['empty', 'single_token', 'multi_token']:
        definitions[name] = ([i for i, r in enumerate(data) if r['span_bin'] == name], list(range(5)))
    for pair, labels in [('Vform/SVA', ['Vform', 'SVA']), ('Nn/ArtOrDet', ['Nn', 'ArtOrDet'])]:
        ids = [i for i, r in enumerate(data) if r['error_type'] in labels]
        li = [LABELS.index(l) for l in labels]
        definitions[pair] = (ids, li)
        for pos in ['VERB', 'AUX', 'NOUN', 'DET']:
            definitions[pair + '@' + pos] = ([i for i in ids if data[i]['primary_upos'] == pos], li)
    groups = sorted({r['group_id'] for r in data})
    gi = {g: i for i, g in enumerate(groups)}
    assert groups == boot['groups']
    weights = np.random.default_rng(20261023).multinomial(len(groups), [1 / len(groups)] * len(groups), size=1000)
    assert np.array_equal(weights, np.load(OUT / 'bootstrap_group_weights.npy'))
    distributions, points, matrices = {}, {}, {}
    metric_checks = ci_checks = 0
    for name, (ids, li) in definitions.items():
        given = metrics[name]
        assert given['support'] == len(ids)
        n_groups = len({data[i]['group_id'] for i in ids})
        assert given['groups'] == n_groups
        eligible = bool(ids) and (name == 'all' or (len(ids) >= 20 and n_groups >= 10 and all(k in truth[ids] for k in li)))
        assert boot['slices'][name]['eligible'] == eligible
        for model in models:
            group_c = np.zeros((len(groups), 5, 5), dtype=np.int64)
            for i in ids:
                group_c[gi[data[i]['group_id']], truth[i], pred[model][i]] += 1
            c = group_c.sum(axis=0)
            matrices[name, model] = c
            reported = given['models'][model]
            assert c.tolist() == reported['confusion']
            assert reported['outside_label_predictions'] == sum(pred[model][i] not in li for i in ids)
            if ids:
                points[name, model] = float(macro(c, li))
                close(reported['macro_f1'], points[name, model])
                close(reported['accuracy'], np.trace(c) / len(ids))
                for k, label in enumerate(LABELS):
                    true_n, pred_n, tp = int(c[k].sum()), int(c[:, k].sum()), int(c[k, k])
                    p = reported['per_label'][label]
                    assert p['support'] == true_n
                    close(p['precision'], tp / pred_n if pred_n else 0)
                    close(p['recall'], tp / true_n if true_n else 0)
                    close(p['f1'], 2 * tp / (true_n + pred_n) if true_n + pred_n else 0)
            else:
                assert reported['macro_f1'] is None and reported['accuracy'] is None
            metric_checks += 1
            if eligible and model in boot['slices'][name]['models']:
                draws = (weights @ group_c.reshape(len(groups), 25)).reshape(1000, 5, 5)
                nonempty = draws.sum(axis=(1, 2)) > 0
                values = macro(draws, li)
                distributions[name, model] = (values, nonempty)
                b = boot['slices'][name]['models'][model]
                close(b['point'], points[name, model])
                expected_ci = np.quantile(values[nonempty], [.025, .975])
                assert np.allclose(b['ci95'], expected_ci, rtol=0, atol=1e-12)
                assert b['valid_replicates'] == int(nonempty.sum())
                assert b['empty_replicates'] == int((~nonempty).sum())
                missing = (draws.sum(axis=2)[:, li] == 0).any(axis=1)
                assert b['missing_class_replicates'] == int((missing & nonempty).sum())
                ci_checks += 1
    for d in boot['differences']:
        a, valid_a = distributions[d['slice'], d['a']]
        b, valid_b = distributions[d['slice'], d['b']]
        valid = valid_a & valid_b
        close(d['difference'], points[d['slice'], d['a']] - points[d['slice'], d['b']])
        assert np.allclose(d['ci95'], np.quantile((a - b)[valid], [.025, .975]), rtol=0, atol=1e-12)
        assert d['valid_replicates'] == int(valid.sum())

    findings = []
    overall = rows(OUT / 'overall_results.csv')
    selected_models = {e['id']: e for e in lock['selected']}
    baseline = read(ROOT / 'reports/nucle_pos_baseline_v1/summary.json')
    for r in overall:
        m = r['pipeline']
        all_m = metrics['all']['models'][m]
        close(r['test_macro_f1'], all_m['macro_f1'])
        close(r['accuracy'], all_m['accuracy'])
        for metric in ['precision', 'recall']:
            close(r['macro_' + metric], np.mean([v[metric] for v in all_m['per_label'].values()]))
        for column, value in zip(['ci95_low', 'ci95_high'], boot['slices']['all']['models'][m]['ci95']):
            close(r[column], value)
        expected_val = selected_models[m]['validation_macro_f1'] if m in selected_models else baseline['majority_baseline_validation' if m == 'Majority' else 'pos_rule_validation']['macro_f1']
        if abs(float(r['validation_macro_f1']) - expected_val) > 1e-12:
            findings.append({'type': 'baseline_precision_only', 'pipeline': m, 'stored': float(r['validation_macro_f1']),
                             'verified': expected_val, 'absolute_difference': abs(float(r['validation_macro_f1']) - expected_val),
                             'four_decimal_report_unchanged': f'{float(r["validation_macro_f1"]):.4f}' == f'{expected_val:.4f}'})
            r['validation_macro_f1'] = str(expected_val)
    with (REVIEW / 'reviewed_overall_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(overall[0])); writer.writeheader(); writer.writerows(overall)
    for r in rows(OUT / 'per_class_results.csv'):
        for key in ['precision', 'recall', 'f1', 'support']:
            close(r[key], metrics['all']['models'][r['pipeline']]['per_label'][r['label']][key])
    for r in rows(OUT / 'per_class_deltas.csv'):
        base = r['pipeline'].split('_')[0] + '_03'
        close(r['delta_f1_vs_full'], metrics['all']['models'][r['pipeline']]['per_label'][r['label']]['f1'] - metrics['all']['models'][base]['per_label'][r['label']]['f1'])
    for r, d in zip(rows(OUT / 'paired_differences.csv'), boot['differences'], strict=True):
        for key in ['slice', 'contrast', 'a', 'b']:
            assert r[key] == d[key]
        close(r['difference'], d['difference'])
        assert np.allclose(json.loads(r['ci95']), d['ci95'], rtol=0, atol=1e-12)
    # Compare DOCX and Markdown tables, then key numeric tables against artifacts.
    md_tables, current = [], []
    for line in (ROOT / 'reports/summaries/evaluation/evaluation_tasks_report.md').read_text(encoding='utf-8').splitlines() + ['']:
        if line.startswith('|'):
            cells = [s.strip() for s in line.strip('|').split('|')]
            if not all(set(s) <= set('-: ') for s in cells):
                current.append(cells)
        elif current:
            md_tables.append(current); current = []
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(ROOT / 'reports/summaries/evaluation/evaluation_tasks_report.docx') as archive:
        doc = ET.fromstring(archive.read('word/document.xml'))
    word_tables = [[[''.join(t.text or '' for t in cell.findall('.//w:t', ns)) for cell in row.findall('w:tc', ns)]
                    for row in table.findall('w:tr', ns)] for table in doc.findall('.//w:tbl', ns)]
    assert word_tables == md_tables
    def table(header):
        matches = [t for t in md_tables if t[0] == header]
        assert len(matches) == 1, header
        return matches[0][1:]
    f = lambda value: 'N/A' if value is None else f'{value:.4f}'
    interval = lambda value: 'N/A' if value is None else f'[{value[0]:.4f}; {value[1]:.4f}]'
    primary = table(['Model', 'Val F1', 'Test F1', 'CI95%', 'Accuracy', 'Macro P', 'Macro R'])
    for row, m in zip(primary, ['Majority', 'POS', 'LR_03', 'Tree_03'], strict=True):
        v = metrics['all']['models'][m]
        assert row[2:] == [f(v['macro_f1']), interval(boot['slices']['all']['models'][m]['ci95']), f(v['accuracy']),
                          f(np.mean([p['precision'] for p in v['per_label'].values()])), f(np.mean([p['recall'] for p in v['per_label'].values()]))]
    for row in table(['Subset', 'n', 'Groups', 'POS F1', 'LR F1', 'Tree F1', 'Δ LR−POS CI95%']):
        name = row[0]; v = metrics[name]
        ci = next((d['ci95'] for d in boot['differences'] if d['slice'] == name and d['a'] == 'LR_03' and d['b'] == 'POS'), None)
        assert row[1:] == [str(v['support']), str(v['groups']), f(v['models']['POS']['macro_f1']), f(v['models']['LR_03']['macro_f1']), f(v['models']['Tree_03']['macro_f1']), interval(ci)]
    for row, entry in zip(table(['Pipeline', 'Số chiều', 'Config chọn', 'Val F1']), lock['selected'], strict=True):
        assert row[0] == entry['model'] + ' ' + entry['variant']
        assert row[1] == str(entry['feature_count'])
        assert json.loads(row[2]) == entry['configuration']
        assert row[3] == f(entry['validation_macro_f1'])
    for row, label in zip(table(['Nhãn', 'Support', 'LR P', 'LR R', 'LR F1', 'POS F1', 'Tree Full F1']), LABELS, strict=True):
        m = metrics['all']['models']
        v = m['LR_03']['per_label'][label]
        assert row == [label, str(v['support']), f(v['precision']), f(v['recall']), f(v['f1']), f(m['POS']['per_label'][label]['f1']), f(m['Tree_03']['per_label'][label]['f1'])]
    for row, entry in zip(table(['Variant', 'LR val', 'LR test', 'Tree val', 'Tree test']), [e for e in lock['selected'] if e['model'] == 'LR'], strict=True):
        tree_id = entry['id'].replace('LR', 'Tree')
        assert row == [entry['variant'], f(entry['validation_macro_f1']), f(metrics['all']['models'][entry['id']]['macro_f1']), f(selected_models[tree_id]['validation_macro_f1']), f(metrics['all']['models'][tree_id]['macro_f1'])]
    for header, contrasts in [(['So sánh', 'Chênh F1', 'CI95%'], boot['differences'][:4]), (['Đối chiếu', 'Chênh test F1', 'CI95%'], boot['differences'][4:20])]:
        assert table(header) == [[d['contrast'], f(d['difference']), interval(d['ci95'])] for d in contrasts]
    class_tables = [t[1:] for t in md_tables if t[0] == ['Variant', *LABELS]]
    assert len(class_tables) == 2
    for family, contents in zip(['LR', 'Tree'], class_tables, strict=True):
        expected = []
        for e in lock['selected']:
            if e['model'] != family:
                continue
            cells = [e['variant']]
            for label in LABELS:
                score = metrics['all']['models'][e['id']]['per_label'][label]['f1']
                base = metrics['all']['models'][family + '_03']['per_label'][label]['f1']
                cells.append(f(score) + ' (' + f'{score - base:+.4f}' + ')')
            expected.append(cells)
        assert contents == expected
    c = matrices['all', 'LR_03']
    pairs = sorted([(int(c[i, j]), LABELS[i], LABELS[j]) for i in range(5) for j in range(5) if i != j], reverse=True)
    assert table(['Gold → prediction', 'Số lỗi']) == [[a + ' → ' + b, str(n)] for n, a, b in pairs[:10]]
    assert table(['Subset', 'Eligible CI', 'Valid', 'Empty', 'Missing-class']) == [
        [name, str(v['eligible']), *[str(v['models'].get('LR_03', {}).get(k, 'N/A')) for k in ['valid_replicates', 'empty_replicates', 'missing_class_replicates']]] for name, v in boot['slices'].items()]
    # Audit record integrity: selection is reproducible and source/parser checks are separate.
    audit = read(OUT / 'audit_selection.json')['rows']
    notes = read(OUT / 'manual_error_review.json')['notes']
    assert len(audit) == len(notes) == len({r['annotation_id'] for r in audit}) == 40
    assert {r['annotation_id'] for r in audit} == {r['annotation_id'] for r in notes}
    membership = Counter('LR_error' if r['LR'] != r['gold'] else 'LR_correct_control' if r['kind'] == 'LR_correct_control' else 'Tree_error_LR_correct' for r in audit)
    assert membership == {'LR_error': 25, 'Tree_error_LR_correct': 5, 'LR_correct_control': 10}
    for name, expected in immutable.items():
        assert digest(OUT / name) == expected
    result = {'test_rows': len(data), 'pipelines': len(models), 'slices': len(definitions),
              'model_slice_metric_checks': metric_checks, 'model_slice_ci_checks': ci_checks,
              'paired_contrast_ci_checks': len(boot['differences']), 'bootstrap_draws': 1000,
              'all_metric_and_ci_checks_pass': True, 'groups_and_essays_disjoint': True,
              'aggregate_csv_rows_checked': {'overall': 20, 'per_class': 100, 'per_class_deltas': 90, 'paired_differences': 30},
              'report_docx_markdown_tables_identical': len(md_tables),
              'report_numeric_tables_match_artifacts': 11,
              'audit_membership': dict(membership), 'findings': findings,
              'original_artifacts_unchanged': immutable}
    (REVIEW / 'evidence_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'original_artifacts_unchanged'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
