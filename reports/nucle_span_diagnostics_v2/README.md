# Span diagnostics v2 — validation

| File | Vai trò |
|---|---|
| [span_diagnostics_report.md](span_diagnostics_report.md) | Báo cáo đọc |
| [summary.json](summary.json) | Contract và tổng hợp |
| [metrics.csv](metrics.csv) | Bảng metric |

Phân biệt gold-label pairs với POS slices. Kết quả test lấy từ final_evaluation/test_metrics.json.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

Phân tích validation theo span rỗng/một/nhiều token, gold-label pairs và POS slices. Giữ predictions năm lớp; prediction ngoài cặp vẫn sai.

span_diagnostics_report.md là mốc trước thử BERT/test. Các mục “chưa chạy” hoặc “tuần sau” trong đó đã được kế hoạch và trạng thái hiện hành thay thế.

Entry point: [src/evaluation/nucle_span_analysis.py](../../src/evaluation/nucle_span_analysis.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
