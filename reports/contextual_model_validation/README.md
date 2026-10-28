# Thử BERT-small và audit validation

Thư mục này lưu đợt thử BERT-small ban đầu trên validation, cùng phép kiểm độ nhạy với tiền xử lý và phần rà soát 18 lỗi. Đây là mốc trước khi BERT được mở rộng và chấm test. Kết quả mới được trình bày riêng tại [bert_final_analysis](../bert_final_analysis/analysis_report.md), để người đọc có thể đối chiếu cả hai giai đoạn.

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

Đây là đợt cũ, không phải BERT-base và không có test của đợt này. Đợt mở rộng ở reports/bert_small_experiment đã hoàn tất ngày 28/10/2026; [báo cáo kết quả mới](../bert_final_analysis/analysis_report.md) có test/CI và phân tích lỗi. Không đổi artifact validation lịch sử. Audit/checkpoints chứa nguồn giữ local.

Entry point: [src/training/bert_span_classifier.py](../../src/training/bert_span_classifier.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
