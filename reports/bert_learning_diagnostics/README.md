# Chẩn đoán khả năng học BERT

Các tệp trong thư mục này giải thích vì sao BERT có điểm validation thấp ở đợt đầu: đầu vào có giữ đúng span không, gradient có đi qua các thành phần cần học không và mô hình có thể học thuộc một tập nhỏ hay không. Chúng chỉ thuộc bước chẩn đoán trên train, không phải kết quả validation hoặc test của đợt mở rộng.

| File | Vai trò |
|---|---|
| [summary.json](summary.json) | Kết quả học thuộc và quyết định |
| [gradient_checks.json](gradient_checks.json) | Gradient |
| [input_audit.json](input_audit.json) | Span/tokenization |
| [same_sentence_pairs.json](same_sentence_pairs.json) | Hai span cùng câu |
| [marker_sensitivity.json](marker_sensitivity.json) | Độ nhạy marker |
| [manifest.json](manifest.json) | Cấu hình/hashes trước chạy |

Các thư mục cls_* và span_mean_* chứa history, predictions và checkpoint train nhỏ. Giữ local; không gọi kết quả này là test BERT.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

Ba lượt học trên 100 train cân bằng kiểm khả năng nhớ nhãn, gradient, optimizer và span representation. Probe 50 cặp khác span cùng câu là stress test có chủ đích.

Đạt accuracy train không đồng nghĩa tốt trên validation/test. Đây là chẩn đoán train nhỏ cũ. Đợt mở rộng đã hoàn tất ngày 28/10/2026: [báo cáo mới](../bert_final_analysis/analysis_report.md), primary test F1 0,8146; mean ba seed 0,8182 ± 0,0038. Các file trong thư mục này vẫn giữ đúng phạm vi chẩn đoán lịch sử.

Entry point: [src/evaluation/bert_learning_diagnostics.py](../../src/evaluation/bert_learning_diagnostics.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
