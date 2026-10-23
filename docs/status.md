# Trạng thái dự án

Cập nhật sau đánh giá cuối ngày 23/10/2026. Nguồn kết quả hiện hành: [evaluation_tasks_report.docx](../reports/summaries/evaluation/evaluation_tasks_report.docx), [Markdown](../reports/summaries/evaluation/evaluation_tasks_report.md). Lịch và bằng chứng: [note.md](../note.md). Báo cáo validation/BERT cũ là snapshot lịch sử.

## Đã thực hiện

- Preparation v2, connected-group split, train-only features, baseline/ablation/masking validation.
- BERT-small validation và diagnostics train-only; dừng thử thêm, không chấm BERT test.
- 40 fit grid Full−P/M;12 refit cũ khớp validation predictions. Full−D alias của S+L+P+M.
- Khóa protocol/checkpoints/encoders/code/data hashes trước test: reports/final_evaluation/lock.json.
- Chấm test20 tên pipeline:18 LR/Tree gồm2 aliases, Majority/POS. Không chọn lại model từ test.
- 1.000 paired bootstrap theo97 connected groups; CI cho RQ1/RQ2 và subsets khó.
- Audit40ca theo ID sau prediction, hỗ trợ AI; không sửa gold hoặc suy prevalence.
- Report task DOCX/Markdown và gói hỗ trợ tái lập local; final report/slide tổng kết chưa chốt.
- Tái lập trong môi trường riêng từ SGML qua preparation v2/grouping/features đến 52 fit của đợt đánh giá và test replay: dữ liệu, checkpoint, predictions khớp.
- Rà soát độc lập 280 model–slice metrics, 40 CI F1, 30 CI chênh lệch; 13 tests pass trong môi trường riêng.
- Kiểm lại 40 ca audit; 17 ca chờ chuyên gia. Chi tiết tại [report task rà soát](../reports/evaluation_followup/review_report.md), không xuất DOCX mới.

## Giới hạn / việc không tuyên bố đã làm

- Đã tái lập chuỗi raw → test cho đợt 52 fit; chưa rerun toàn grid lịch sử của sáu variant và BERT. Review log/cấu hình lịch sử là input cố định; chưa phải tái lập mọi quyết định nghiên cứu.
- Audit chưa được adjudication độc lập của chuyên gia.
- Không có test BERT hoặc bằng chứng mọi mô hình ngữ cảnh đều kém LR.
- Test hiện đã mở; phát hiện mới chỉ dùng cho future work với đánh giá mới.

Artifact: reports/final_evaluation/. Protocol trước test: [final_evaluation_protocol.md](final_evaluation_protocol.md). Các báo cáo cũ không được diễn giải thành điểm test.

## Phạm vi báo cáo đã điều chỉnh

Theo yêu cầu mới, chỉ bàn giao report task. Final report dự án và slide tổng kết chưa thực hiện/chưa chốt; bản nháp tạo trước đó được lưu lịch sử. Không diễn giải việc hoàn thành thí nghiệm là đã hoàn thành báo cáo nộp cuối.
