# Ablation sáu variant — validation

| File | Vai trò |
|---|---|
| [summary.json](summary.json) | Grid và kết quả sáu variant |
| `validation_predictions.csv` (local) | Dự đoán theo variant/model |

Đầu vào lịch sử cho các refit trong đợt đánh giá. Full−P/M và alias Full−D được bổ sung ở final_evaluation.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

Sáu variants × (8 LR + 12 Tree) = 120 fit. Mỗi variant chọn hyperparameters riêng bằng validation; masking fit encoder riêng sau khi bỏ lexical span, giữ parse câu gốc.

Full−P/M và alias Full−D thuộc đợt final sau đó. Folder v1 là phiên bản ablation, không chỉ preparation v1.

Entry point: [src/training/nucle_ablation.py](../../src/training/nucle_ablation.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
