# Tài liệu thiết kế và phương pháp

`docs` quản lý đề cương, phương pháp, protocol và related work. Báo cáo kết quả để đọc nằm ở [reports/summaries](../reports/summaries/README.md); dữ liệu kết quả từng đợt được chỉ dẫn ở [reports/README.md](../reports/README.md).

| Đọc gì | Tài liệu chính |
|---|---|
| Trạng thái hiện tại | [status.md](status.md) |
| Kế hoạch còn lại | [note.md](../note.md) |
| Mapping toàn dự án, preparation v1/v2 | [note_project_guide.md](note_project_guide.md) |
| Đề cương và câu hỏi nghiên cứu ban đầu | [proposal.md](proposal.md) |
| Schema, segmentation, offset, alignment | [methodology/preparation.md](methodology/preparation.md) |
| Split, đặc trưng, tuning và metric | [methodology/experiment_protocol.md](methodology/experiment_protocol.md) |
| Đối chiếu code với mô tả thiết kế | [methodology/implementation_data_audit.md](methodology/implementation_data_audit.md) |
| Tổng hợp bài liên quan | [literature/README.md](literature/README.md) |
| Danh mục nguồn học thuật | [reference.md](../reference.md) |

## Protocol đã ghi trước thí nghiệm

| File giữ nguyên | Phạm vi |
|---|---|
| [final_evaluation_protocol.md](final_evaluation_protocol.md) | Ablation bổ sung, khóa cấu hình, test, bootstrap, audit |
| [bert_experiment_protocol.md](bert_experiment_protocol.md) | Thử BERT-small trên validation |
| [bert_learning_diagnostics_protocol.md](bert_learning_diagnostics_protocol.md) | Kiểm học thuộc, gradient và marker |
| [bert_target_pair_probe_protocol.md](bert_target_pair_probe_protocol.md) | Kiểm hai span khác nhãn trong cùng câu |

Bốn protocol này giữ nguyên nội dung và đường dẫn vì manifest/hash và code lịch sử tham chiếu trực tiếp. Chúng ghi thiết kế tại thời điểm trước chạy; trạng thái thực tế luôn xem status.md. Đề cương cũng giữ vai trò lịch sử, không thay kế hoạch hiện hành.

## Quy tắc tổ chức

- Phương pháp ở docs; báo cáo kết quả ở reports/summaries; artifact máy chạy ở thư mục thí nghiệm tương ứng.
- Mỗi báo cáo chỉ có một bản hiện hành; mục lục dẫn liên kết, không sao chép bảng kết quả.
- Tiến độ quản lý tại status.md, lịch tại note.md, thư mục bài báo tại reference.md.
- Không đặt tên file theo tuần hoặc thêm hậu tố `.updated`; version thí nghiệm đã ghi trong artifact vẫn giữ nguyên.
- Tài liệu validation/BERT và báo cáo đánh giá trước lượt rà soát là snapshot. Báo cáo rà soát mới nhất nằm ở [evaluation_followup/review_report.md](../reports/evaluation_followup/review_report.md).
- Final report dự án và slide tổng kết chưa chốt. Các bản nháp đã bỏ được giữ riêng trong documentation_archive.

Ánh xạ trước/sau và bản sao phục hồi: [reports/organization](../reports/organization/README.md). Theo yêu cầu công bố code và docs, toàn bộ tài liệu hiện hành, note/reference và báo cáo để đọc được quản lý bằng Git. Corpus và artifact riêng tư vẫn giữ local; phạm vi cụ thể ở reports/README.md. Quy tắc chỉ push proposal trong ghi chép cũ đã được thay thế bởi yêu cầu này; các protocol lịch sử giữ nguyên nội dung.
