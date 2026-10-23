# Đánh giá test đã khóa

| File | Vai trò |
|---|---|
| [lock.json](lock.json) | Cấu hình/input/code/checkpoint hashes trước test |
| [selection_summary.json](selection_summary.json) | 52 fit và 18 tên LR/Tree gồm aliases |
| [test_metrics.json](test_metrics.json) | Metric test theo model và tập con |
| `test_predictions.csv` (local) | Dự đoán từng sample |
| [bootstrap.json](bootstrap.json) | CI và paired differences |
| `manual_error_review.json` (local) | 40 ghi chú audit |
| [reproducibility/README.md](reproducibility/README.md) | Gói bàn giao và phạm vi kiểm chứng tại thời điểm đóng gói |

checkpoints/ và private_error_audit_packet.json giữ local. documentation_archive/ chứa bản nháp đã bỏ, không phải báo cáo hiện hành. Không sửa lock hoặc gọi lại score lên thư mục này. Kết quả rà soát/tái lập mới hơn nằm ở evaluation_followup.

[Trở về mục lục reports](../README.md).

## Điều kiện đọc và tái lập

Đợt này có 40 grid fit Full−P/M và 12 refit cũ. Hai tên Full−D reuse S+L+P+M, nên 18 tên LR/Tree chỉ tương ứng 16 checkpoint riêng. Majority/POS đưa tổng tên pipeline lên 20. Model chỉ fit train, chọn validation; test không dùng tuning.

Git chứa lock và aggregate results; một clone không có đủ mọi file mà verify_lock cần. `score` có guard chống chấm lại; `bootstrap`/`replay` vẫn ghi output kiểm chứng. Điều kiện cụ thể: [runbook hiện hành](../../docs/reproduction.md). [reproducibility/README.md](reproducibility/README.md) mô tả gói của mốc bàn giao cũ; bằng chứng mới hơn ở [followup](../evaluation_followup/README.md).
