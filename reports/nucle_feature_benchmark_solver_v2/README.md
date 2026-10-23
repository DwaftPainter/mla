# Benchmark Full LR/Tree — validation

| File | Vai trò |
|---|---|
| [summary.json](summary.json) | Grid, solver, thời gian và cấu hình chọn |
| `validation_predictions.csv` (local) | Dự đoán validation |

Giữ nguyên artifact và solver của đợt chạy. Không chọn cấu hình từ test.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

Benchmark Full gồm grid LR 8 cấu hình và Tree 12 cấu hình, encoder train-only, chọn macro-F1 validation. Solver v2 là mốc cấu hình LR, không phải preparation v2.

Summary là kết quả validation; checkpoints cuối được refit/khóa trong final_evaluation.

Entry point: [src/training/nucle_benchmark.py](../../src/training/nucle_benchmark.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
