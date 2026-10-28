# Span diagnostics v1 — lịch sử

Đây là phần phân tích span ở giai đoạn đầu. Các nhận xét được giữ để theo dõi quá trình phát triển, nhưng không dùng thay cho kết quả bổ sung ở v2 hoặc kết quả test cuối cùng. Khi đọc, cần phân biệt thống kê đầu vào với điểm đánh giá mô hình.

| File | Vai trò |
|---|---|
| [summary.json](summary.json) | Tổng hợp tập con ban đầu |
| [metrics.csv](metrics.csv) | Metric theo tập con |

Giữ snapshot cũ. Bản bổ sung có POS slices nằm ở nucle_span_diagnostics_v2.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

Snapshot phân tích span/cặp nhãn/POS ban đầu. Đọc v2 cho báo cáo validation đã bổ sung; giữ snapshot v1 để truy vết.

Không dùng folder này làm bảng test hoặc trạng thái BERT hiện hành.

Entry point: [src/evaluation/nucle_span_analysis.py](../../src/evaluation/nucle_span_analysis.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
