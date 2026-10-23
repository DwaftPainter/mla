# Test, audit và tái lập

| Đọc theo mốc | Tài liệu |
|---|---|
| Các task ablation bổ sung, test, bootstrap và audit | [evaluation_tasks_report.docx](evaluation_tasks_report.docx), [Markdown](evaluation_tasks_report.md) |
| Rà soát và tái lập mới nhất | [evaluation_followup/review_report.md](../../evaluation_followup/review_report.md) |
| 17 ca cần người chuyên môn | [expert_review_queue.csv](../../evaluation_followup/expert_review_queue.csv) |

Báo cáo task đầu tiên là snapshot trước khi tái lập raw → features → 52 fit → test replay; giới hạn môi trường sạch ghi trong đó đã được bổ sung ở report rà soát. Đây đều là báo cáo task; final report dự án và slide tổng kết để sau.

Artifact: [final_evaluation](../../final_evaluation/README.md), [evaluation_followup](../../evaluation_followup/README.md). Protocol đã khóa: [final_evaluation_protocol.md](../../../docs/final_evaluation_protocol.md). Bảng chính xác đến đầy đủ chữ số dùng [reviewed_overall_results.csv](../../evaluation_followup/reviewed_overall_results.csv); bảng lịch sử giữ số đã xuất ở thời điểm chạy.

Các trường “chưa tái chạy từ raw” trong DOCX/Markdown task đầu phản ánh mốc trước followup. Tái lập mới đã bao phủ 52 fit của đợt final, chưa bao phủ mọi grid/BERT lịch sử hoặc adjudication con người. 13 tests trong followup khác bộ 99 tests của lần bàn giao code; [tests/README.md](../../../tests/README.md) giải thích phạm vi.

20 tên pipeline gồm 16 checkpoint LR/Tree riêng, 2 aliases Full−D và 2 baseline. Không gọi đây là 20 model đã huấn luyện độc lập. CI là pointwise paired group-bootstrap, không phải nhiều-seed training CI hoặc kiểm định đồng thời đã hiệu chỉnh.
