"""Recheck source/offset/parser evidence for the existing 40 audit cases.

No adjudication, relabeling, case reselection or test-model tuning. Output contains
IDs and aggregate feature contributions, not source sentences or lexical values.
"""
from pathlib import Path
from collections import Counter
import csv
import difflib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.data.constants import SGML_FILE
from src.data.nucle_sgml import load_nucle
from src.data.parse_nucle import documents_to_dataframe
from src.features.nucle_pos import load_tagger, align_span
from src.features.nucle_features import extract_features
from src.training.nucle_final_selection import read, dump, sha, LABELS
from src.evaluation.nucle_final_evaluation import verify_lock, load_predictions
from src.evaluation.nucle_final_error_audit import choose

OUT = ROOT / 'reports/final_evaluation'
REVIEW = ROOT / 'reports/evaluation_followup'


def main():
    lock = verify_lock()
    packet = read(OUT / 'private_error_audit_packet.json')
    notes = {r['annotation_id']: r for r in read(OUT / 'manual_error_review.json')['notes']}
    meta, predictions = load_predictions()
    errors, controls = choose(meta, predictions)
    assert [meta[i]['annotation_id'] for i in errors + controls] == [r['annotation_id'] for r in packet]
    source = documents_to_dataframe(load_nucle(SGML_FILE)).set_index('annotation_id')
    wanted = {r['sample_id'] for r in packet}
    samples = {r['sample_id']: r for r in map(json.loads, (ROOT / 'data_set/processed/nucle_preparation_v2/samples.jsonl').read_text(encoding='utf-8').splitlines()) if r['sample_id'] in wanted}
    nlp = load_tagger()
    checks, private_parsed = [], {}
    for r in packet:
        sid, ident = r['sample_id'], r['annotation_id']
        raw, sample = source.loc[int(ident)], samples[sid]
        assert raw.error_type == r['gold'] == sample['error_type']
        assert raw.error_span == r['span'] == sample['error_span']
        assert raw.correction == r['correction']
        assert raw.paragraph[int(raw.start_off):int(raw.end_off)] == r['span']
        start, end = r['offsets']
        assert [sample['error_start'], sample['error_end']] == [start, end]
        assert r['sentence'][start:end] == r['span']
        assert sample['sentence'] == r['sentence']
        doc = nlp(sample['sentence'])
        private_parsed[ident] = doc
        alignment = align_span(doc, start, end)
        fresh_tokens = [{'i': t.i, 'text': t.text, 'pos': t.pos_, 'dep': t.dep_, 'head': t.head.i,
                         'morph': str(t.morph), 'target': t.i in alignment['span_token_indices']} for t in doc]
        assert fresh_tokens == r['tokens'], ident
        features = {k: v for k, v in extract_features(sample, doc).items() if k.startswith(('P|', 'M|', 'D|'))}
        assert features == r['features'], ident
        original = [t.casefold() for t in r['span'].split()]
        corrected = [t.casefold() for t in r['correction'].split()]
        ops = [tag for tag, *_ in difflib.SequenceMatcher(a=original, b=corrected, autojunk=False).get_opcodes() if tag != 'equal']
        insert_only = bool(ops) and set(ops) == {'insert'}
        checks.append({'annotation_id': ident, 'gold': r['gold'], 'LR': r['LR'],
                       'source_label_span_correction_exact': True, 'offset_reconstruction_exact': True,
                       'parser_and_PMD_features_exact': True, 'missing_correction': bool(raw.missing_correction),
                       'empty_correction': raw.correction == '', 'nonempty_span': start < end,
                       'alignment': alignment['span_alignment'], 'span_pos': alignment['pos_signature'],
                       'token_edit_operations_casefold_whitespace': ops,
                       'annotation_adds_words_to_nonempty_span': insert_only and start < end,
                       'original_review_category': notes[ident]['category']})
    by_id = {r['annotation_id']: r for r in packet}
    confirmed_subject = []
    for ident in ['36821', '17311', '18531']:
        doc = private_parsed[ident]
        targets = [t['i'] for t in by_id[ident]['tokens'] if t['target']]
        direct = [child.i for i in targets for child in doc[i].children if child.dep_.lower() in {'nsubj', 'nsubjpass', 'csubj', 'csubjpass'}]
        head_children = [child.i for i in targets for child in doc[i].head.children if child.dep_.lower() in {'nsubj', 'nsubjpass', 'csubj', 'csubjpass'}]
        assert not direct and by_id[ident]['features']['D|has_subject'] == 0
        confirmed_subject.append({'annotation_id': ident, 'direct_subjects': direct,
                                   'subjects_attached_to_target_head': sorted(set(head_children)),
                                   'interpretation': 'Extractor records only direct subject children; head/conj propagation is absent. Parser links are not expert gold.'})
    # Exact additive margin decomposition for existing LR decisions; no intervention.
    entry = next(e for e in lock['selected'] if e['id'] == 'LR_03')
    cp = torch.load(ROOT / entry['checkpoint'], map_location='cpu', weights_only=True)
    names = read(OUT / 'full_encoder.json')['feature_names']
    encoded = torch.load(OUT / 'test_encoded.pt', map_location='cpu', weights_only=True)
    index = {sid: i for i, sid in enumerate(encoded['ids'])}
    margins = []
    a, b = LABELS.index('ArtOrDet'), LABELS.index('Nn')
    for ident in ['11916', '44353', '6079', '44388']:
        r = by_id[ident]
        x = encoded['full'][index[r['sample_id']]]
        sums = {group: 0.0 for group in 'SLPMD'}
        for j, value in x.items():
            group = names[j].split('|')[1] if names[j].startswith(('N|', 'C|')) else 'L'
            sums[group] += value * (float(cp['w'][j, a]) - float(cp['w'][j, b]))
        intercept = float(cp['b'][a]) - float(cp['b'][b])
        margin = sum(sums.values()) + intercept
        logits = cp['b'].double().clone()
        for j, value in x.items():
            logits += value * cp['w'][j].double()
        assert abs(float(logits[a] - logits[b]) - margin) < 1e-9
        assert LABELS[int(logits.argmax())] == r['LR']
        margins.append({'annotation_id': ident, 'gold': r['gold'], 'LR': r['LR'], 'POS': r['POS'],
                        'contrast': 'logit(ArtOrDet) - logit(Nn)', 'contributions_by_group': sums,
                        'intercept': intercept, 'margin': margin,
                        'scope': 'Post-hoc algebraic explanation of saved logits; not causal feature importance or new evaluation'})
    questions = {
        '34539': 'Nhãn ArtOrDet có phù hợp với thao tác thêm particle trong annotation này không?',
        '12324': 'Nhãn SVA liên quan phần nào của câu khi correction không trực tiếp đổi agreement?',
        '26635': 'Correction xóa giới từ nhưng nhãn SVA: cần đọc các annotation liên quan để giải thích.',
        '32705': 'Correction thay giới từ nhưng nhãn ArtOrDet: kiểm taxonomy/annotation gốc.',
        '37909': 'Correction xóa determiner nhưng nhãn Nn: đây là convention hay nhãn cần xem lại?',
        '18501': 'Correction đảo auxiliary trong mệnh đề gián tiếp: kiểm cả tính phù hợp của edit và nhãn ArtOrDet.',
        '201': 'Cấu trúc pseudo-cleft này có bắt buộc thêm dấu infinitive không?',
        '18531': 'Động từ thuộc phạm vi phối hợp nào; gold SVA có phù hợp cách đọc toàn câu?',
        '37631': 'Phạm vi danh sách/mệnh đề và cách đọc noun/verb có thống nhất với correction không?',
        '11916': 'Determiner được thêm có bắt buộc trong ngữ cảnh này hay là lựa chọn diễn đạt?',
        '6079': 'Determiner được thêm có bắt buộc với cách đọc danh từ trong ngữ cảnh này không?',
    }
    for ident in ['24526', '9351', '41235', '35455', '211', '37624']:
        questions[ident] = 'Xác nhận POS/dependency phù hợp bằng cách đọc toàn câu; parser tự động chưa phải gold.'
    REVIEW.mkdir(exist_ok=True)
    with (REVIEW / 'expert_review_queue.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['annotation_id', 'gold', 'current_category', 'question', 'status', 'expert_decision'])
        writer.writeheader()
        for ident, question in questions.items():
            writer.writerow({'annotation_id': ident, 'gold': by_id[ident]['gold'], 'current_category': notes[ident]['category'],
                             'question': question, 'status': 'pending_independent_human_review', 'expert_decision': ''})
    result = {'review_type': 'Second AI-assisted evidence review, not independent linguistic adjudication',
              'selection_reproduced': True, 'cases_checked': len(checks),
              'all_source_offsets_parser_and_PMD_checks_pass': True,
              'insertion_only_annotations_on_nonempty_spans': [r['annotation_id'] for r in checks if r['annotation_adds_words_to_nonempty_span']],
              'empty_correction_ids': [r['annotation_id'] for r in checks if r['empty_correction']],
              'missing_correction_ids': [r['annotation_id'] for r in checks if r['missing_correction']],
              'subject_scope_checks': confirmed_subject, 'lr_margin_decompositions': margins,
              'expert_review_pending': len(questions),
              'interpretation_limits': ['Insertion edits in annotations are not proof the addition is grammatically obligatory.',
                                       'Parser discrepancies remain suspected until expert confirmation.',
                                       'Same target word in two sentences is not a controlled intervention on context length.',
                                       'Do not estimate corpus error prevalence from these purposively selected cases.',
                                       'Gold labels, splits and model parameters remain unchanged.'],
              'cases': checks, 'source_packet_sha256': sha(OUT / 'private_error_audit_packet.json')}
    dump(REVIEW / 'audit_followup.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
