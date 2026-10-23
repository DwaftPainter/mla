# Thử BERT-small và audit validation

| File | Vai trò |
|---|---|
| [comparison.csv](comparison.csv) | Bảng so sánh validation |
| [comparison.json](comparison.json) | Kết quả và metadata |
| [per_class_metrics.csv](per_class_metrics.csv) | Metric từng lớp |
| [bert/selection.json](bert/selection.json) | Learning rate/seed đã chọn |
| [audit/summary.json](audit/summary.json) | Sensitivity preparation v1/v2 |
| [verification.json](verification.json) | Kiểm chứng trước đó |

bert/ giữ lịch sử fine-tuning/checkpoint; audit/ có packet chứa câu nguồn. BERT chưa chấm test. Báo cáo DOCX chuyển sang reports/summaries/contextual.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

Benchmark BERT-small có bốn run: hai learning rate ở seed đầu, thêm hai seed ở learning rate chọn. Cấu hình chọn theo toàn validation, báo mean/std ba seed, không chọn seed cao nhất.

Đây không phải BERT-base. Test BERT chưa chấm; dừng mở rộng. Audit có văn bản nguồn/checkpoints local; Git chỉ có bảng và ảnh đã rà soát.

Entry point: [src/training/bert_span_classifier.py](../../src/training/bert_span_classifier.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
