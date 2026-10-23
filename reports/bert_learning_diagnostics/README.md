# Chẩn đoán khả năng học BERT

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

Đạt accuracy train không đồng nghĩa tốt trên validation/test. Đề xuất thử tiếp trong report lịch sử đã chuyển sang future work; quyết định hiện hành là dừng BERT.

Entry point: [src/evaluation/bert_learning_diagnostics.py](../../src/evaluation/bert_learning_diagnostics.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
