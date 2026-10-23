# Mô hình cổ điển và validation

[modeling_ablation_results_report.docx](modeling_ablation_results_report.docx) giải thích features, LR/Tree, công thức và ablation/masking **validation**. Tên file đã bỏ hậu tố `.updated`; DOCX giữ nội dung lịch sử.

## Bổ sung khi đọc DOCX

- Những câu “BERT chưa chạy”, “test chưa chấm” và “chưa có CI” thuộc mốc báo cáo. Đánh giá test/CI và thử BERT hiện đã có; xem [status](../../../docs/status.md).
- Giới hạn 500 là kênh pooled n-gram, không phải toàn bộ lexical vocabulary hoặc 500 chiều input. Full có 16.180 chiều; categorical slots học vocabulary riêng; numeric không z-score.
- Masking là xóa lexical identity của span trước khi fit encoder masked, giữ parse câu gốc. Không parse lại câu có marker để làm ablation này.
- LR là PyTorch LBFGS; Tree là implementation Gini trong project. Không coi kết quả tương đương sklearn, đặc biệt class weighting của Tree.
- LR Full và Tree Full là benchmark cùng biểu diễn. Tree S+L+P+M tốt hơn Tree Full trên validation được báo riêng, không thay model trong bảng đối chứng Full.

Artifact: [benchmark](../../nucle_feature_benchmark_solver_v2/README.md), [sáu variants](../../nucle_feature_ablation_v1/README.md), [POS](../../nucle_pos_baseline_v1/README.md), [span diagnostics](../../nucle_span_diagnostics_v2/README.md). Phần khác thiết kế ban đầu xem [implementation audit](../../../docs/methodology/implementation_data_audit.md).

Kết quả test, Full−P/M/D và uncertainty nằm ở [nhóm đánh giá](../evaluation/README.md). Không thay bảng validation cũ bằng điểm test.
