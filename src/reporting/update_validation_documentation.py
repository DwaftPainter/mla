"""Update current documentation only after complete week-1 artifacts exist."""
from pathlib import Path
import json, hashlib, zipfile
from docx import Document
ROOT=Path(__file__).resolve().parents[2]
def run():
    out=ROOT/'reports/contextual_model_validation'
    selection=json.loads((out/'bert/selection.json').read_text(encoding='utf-8'))
    replay=json.loads((out/'bert/checkpoint_replay.json').read_text(encoding='utf-8'))
    assert len(selection['runs'])==4 and len(replay)==3
    assert all(r['prediction_disagreements']==0 for r in replay)
    for name in ['contextual_model_validation_report.md','contextual_model_validation_report.docx']:
        assert (ROOT/'docs'/name).exists()
    p=ROOT/'note.md'; text=p.read_text(encoding='utf-8')
    text=text.replace('Các mục dưới đây là việc dự kiến, không phải kết quả đã hoàn thành.','Bảng dưới lưu kế hoạch gốc. Tuần 1 đã được thực hiện trong phiên ngày 23/10/2026; bảng hoàn thành và report bên dưới ghi trạng thái thực tế. Tuần 2 chưa thực hiện.')
    text=text.replace('- BERT chưa chạy; test chưa chấm.', '- Tại thời điểm lập kế hoạch, BERT chưa chạy. Sau thực hiện tuần 1, BERT-small đã chạy hai learning rate và ba seed ở learning rate chọn; test vẫn chưa chấm.')
    section='''
### Kết quả thực hiện tuần 1 — 23/10/2026

Đã thực hiện toàn bộ nhóm việc tuần 1, không thực hiện tuần 2. Kết quả chi tiết: [report DOCX mới](docs/contextual_model_validation_report.docx), [bản Markdown](docs/contextual_model_validation_report.md).

| Task | Trạng thái và bằng chứng |
|---|---|
| Related work | Hoàn thành bảng và tổng hợp: docs/contextual_models_related_work.md |
| Audit validation và code–protocol | Kiểm offset/token alignment, review hỗ trợ bằng AI 18 lỗi theo ID; ghi nhận khác biệt vocabulary/masking và hạn chế: docs/implementation_data_audit.md |
| Sensitivity v1/v2 | Đã chạy LR/POS trên cùng split; chẩn đoán refit LR theo số threads: reports/contextual_model_validation/audit/ |
| Protocol và pilot BERT | Đã chốt trước fine-tuning: docs/bert_experiment_protocol.md |
| BERT | Hoàn thành bốn lượt fine-tune BERT nhỏ trên CPU: hai learning rate ở seed đầu, thêm hai seed cho learning rate chọn; có checkpoint/history/predictions |
| So sánh validation | Có toàn tập, từng lớp, span, cặp nhãn và POS slices: reports/contextual_model_validation/comparison.csv |
| Report tuần 1 | Hoàn thành report mới, có hạn chế, công thức, code và kết quả âm; không ghi đè báo cáo lịch sử |

18 kiểm tra chức năng đã qua; ba checkpoint ở learning rate chọn tái tạo đúng prediction validation. Không đồng nghĩa đã tái lập môi trường sạch hoặc adjudication ngôn ngữ độc lập. Chỉ thử BERT-small, chưa thử BERT-base. Test/CI và khóa mô hình cuối thuộc tuần 2, chưa làm.

'''
    if '### Kết quả thực hiện tuần 1' not in text: text=text.replace('### Tuần 2 —',section+'### Tuần 2 —')
    p.write_text(text,encoding='utf-8')
    (ROOT/'docs/status.md').write_text('''# Trạng thái dự án

Cập nhật sau thực hiện tuần 1, 23/10/2026. Report hiện hành: [contextual_model_validation_report.docx](contextual_model_validation_report.docx), [Markdown](contextual_model_validation_report.md). Lịch và task: [note.md](../note.md).

## Đã hoàn thành

- Preparation, split và baseline/ablation/masking validation trước đây, giữ nguyên artifact.
- Related work cho tuần 1; không sửa Excel/PRISMA.
- Audit offset/alignment train+validation, review hỗ trợ bằng AI 18 lỗi validation và đối chiếu code–protocol.
- Sensitivity v1/v2 với LR/POS trên cùng split; xác minh ảnh hưởng số threads đến refit LR.
- Protocol, pilot và bốn lượt BERT-small: hai learning rate ở seed đầu, tổng ba seed cho learning rate chọn. Không phải BERT-base.
- So sánh toàn validation, từng lớp, span, cặp nhãn và POS; mean/std theo seed, confusion matrix, report mới.
- 18 kiểm tra chức năng đã qua; ba checkpoint chọn theo learning rate tái tạo prediction validation không bất đồng.

## Chưa thực hiện — thuộc tuần 2

- Khóa mô hình cuối và đánh giá test. Test chưa chấm.
- Bootstrap theo group, phân tích lỗi test, cập nhật report cuối/slide.
- Kiểm tra tái lập môi trường sạch và khôi phục gói tái lập.

Không diễn giải kết quả BERT-small thành kết luận về mọi mô hình BERT. Audit ngôn ngữ không thay đánh giá độc lập của chuyên gia. Các hạn chế được ghi trong report tuần 1.
''',encoding='utf-8')
    p=ROOT/'docs/README.md'; text=p.read_text(encoding='utf-8').replace('- Ngày 23/10/2026: BERT chưa chạy, test chưa chấm. Kế hoạch không phải kết quả.','- Sau tuần 1 ngày 23/10/2026: BERT-small đã chạy; test chưa chấm. Báo cáo mới: [contextual_model_validation_report.docx](contextual_model_validation_report.docx), [Markdown](contextual_model_validation_report.md). Hai DOCX cũ giữ kết quả mốc 20/10; không dùng trạng thái lịch sử trong đó thay cho status.md.')
    text+='\n## Tài liệu phát sinh tuần 1\n\n- contextual_models_related_work.md: bảng literature và tổng hợp.\n- implementation_data_audit.md: audit implementation và ngôn ngữ.\n- bert_experiment_protocol.md: cấu hình đóng dấu trước fine-tuning; không sửa ngược theo kết quả.\n- contextual_model_validation_report.md/.docx: kết quả mới; thay các mô tả tiến độ trước khi chạy tuần 1.\n'
    p.write_text(text,encoding='utf-8')
    p=ROOT/'docs/experiment_protocol.md'; text=p.read_text(encoding='utf-8').replace('BERT và test chưa chạy.','BERT-small đã chạy trong tuần 1; test chưa chấm. Kết quả ở contextual_model_validation_report.md.').replace('## 8. Bổ sung phạm vi BERT — 23/10/2026','## 8. Bổ sung phạm vi BERT — ghi trước chạy ngày 23/10/2026 (lịch sử)')
    text+='\n\n## 10. Hoàn tất tuần 1\n\nBERT-small đã thực hiện theo bert_experiment_protocol.md, không sửa cấu hình đăng ký sau kết quả. Report: contextual_model_validation_report.md. Số liệu và trạng thái trong mục 8 là mốc trước chạy; test, bootstrap và khóa mô hình cuối chưa làm. Sensitivity LR dùng 4 threads; chạy lại baseline đã lưu với 10 threads tái tạo prediction cũ. Ghi rõ threads khi tái lập LBFGS.\n'
    p.write_text(text,encoding='utf-8')
    # Preserve old reports as dated snapshots; add explicit pointer to the new report.
    for name in ['data_preprocessing_report.docx','modeling_ablation_results_report.updated.docx']:
        path=ROOT/'docs'/name; doc=Document(path)
        message='Ghi chú sau tuần 1 (23/10/2026): tài liệu này giữ kết quả và trạng thái lịch sử mốc 20/10. Kết quả audit, sensitivity và BERT-small mới xem contextual_model_validation_report.docx; trạng thái hiện hành xem status.md. Test chưa chấm.'
        if not any(p.text==message for p in doc.paragraphs): doc.paragraphs[0].insert_paragraph_before(message)
        doc.save(path)
    p=ROOT/'reference.md'; text=p.read_text(encoding='utf-8'); text=text.replace('## A. Các bài đã dùng','Bổ sung sau tuần 1: REF07, REF11–REF15 và REF20 đã được vận dụng trong related work/protocol/report mới. Trạng thái dự kiến ở các mục dưới mô tả thời điểm lập danh mục; kết quả thực thi xem docs/contextual_model_validation_report.md.\n\n## A. Các bài đã dùng',1); p.write_text(text,encoding='utf-8')
    # Artifact inventory excludes the private packet and checkpoint weights from content output.
    paths=[p for p in out.rglob('*') if p.is_file() and p.name!='artifact_hashes.json']
    inventory={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'artifact_hashes.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print('Week 1 docs finalized; test remains unscored.')
if __name__=='__main__': run()
