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
