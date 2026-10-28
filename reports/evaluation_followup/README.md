# Rà soát bằng chứng và tái lập

Báo cáo trong thư mục này kiểm tra lại các kết quả cổ điển đã lưu: số liệu có tính đúng từ dự đoán hay không, các bảng có nhất quán không và quy trình tái lập được đến đâu. Đây là bằng chứng của đợt rà soát ngày 23/10/2026, không phải một lần chấm test độc lập mới.

| File | Vai trò |
|---|---|
| [review_report.md](review_report.md) | Báo cáo task mới nhất |
| [pipeline_reproduction.json](pipeline_reproduction.json) | Tái lập từ raw đến test replay |
| [evidence_review.json](evidence_review.json) | Metric, CI, CSV và report checks |
| [audit_followup.json](audit_followup.json) | Kiểm lại 40 ca |
| [reviewed_overall_results.csv](reviewed_overall_results.csv) | Bảng baseline đầy đủ chữ số |
| [expert_review_queue.csv](expert_review_queue.csv) | 17 ca chờ chuyên gia |
| [tests.xml](tests.xml) | 13 tests của tập con trong môi trường tái lập; không phải toàn suite |

Các file là bằng chứng của đợt rà soát. Artifact hash cũ ghi trạng thái code/tài liệu trước lần tổ chức này; ánh xạ và kiểm tra tổ chức nằm ở reports/organization. Không tự sửa manifest lịch sử.

[Trở về mục lục reports](../README.md).

`reviewed_overall_results.csv` bổ sung đủ chữ số thập phân cho điểm validation của baseline so với bảng lịch sử, không thay test predictions/metric. `expert_review_queue.csv` chỉ giữ IDs/câu hỏi/trạng thái, 17 ca chưa có kết luận chuyên gia; không diễn giải thành 17 nhãn sai.

Hướng dẫn chạy các reviewer và điều kiện input local: [reproduction.md](../../docs/reproduction.md). Những câu “không push trong lượt này” trong review_report.md mô tả phiên tạo báo cáo; các report tổng hợp đã được cho phép công bố ở lượt sau.
