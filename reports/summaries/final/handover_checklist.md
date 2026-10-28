# Bàn giao, xác nhận đóng góp và khai báo AI

Cập nhật ngày 28/10/2026. Bộ tài liệu đã có kết quả huấn luyện, lựa chọn cấu hình, test, kiểm chứng dự đoán và bootstrap. Tên/MSSV, đóng góp thực tế và xác nhận nộp bài vẫn cần nhóm bổ sung. Những thông tin này không thể suy ra từ danh sách module hoặc nhật ký hỗ trợ AI.

## File sử dụng

- [Báo cáo Markdown](bao_cao_tong_ket_NUCLE.md), [DOCX](bao_cao_tong_ket_NUCLE.docx), [PDF](bao_cao_tong_ket_NUCLE.pdf).
- [Slide PowerPoint](NUCLE_presentation.pptx), [slide PDF](NUCLE_presentation.pdf), [ghi chú thuyết trình](speaker_notes.md).
- [Bảng/hình và kiểm chứng BERT](../../bert_final_analysis/analysis_report.md).
- Các ca có câu gốc nằm trong reports/bert_final_analysis/private/error_review_with_ai.csv. Cột review và contribution dành cho nhóm vẫn trống. Hàng đợi hợp nhất có 22 ID cần người đánh giá độc lập; các ghi chú AI chỉ là phần hỗ trợ để bắt đầu đọc.

## Xác nhận thành viên và đóng góp thực tế

| Thành viên | Họ tên/MSSV | Phần việc thực tế | Bằng chứng code/log/review | Tự kiểm tra và xác nhận |
|---|---|---|---|---|
| 1 | CHƯA CUNG CẤP | Cần điền | Cần dẫn file/ID cụ thể | Chưa xác nhận |
| 2 | CHƯA CUNG CẤP | Cần điền | Cần dẫn file/ID cụ thể | Chưa xác nhận |
| 3 | CHƯA CUNG CẤP | Cần điền | Cần dẫn file/ID cụ thể | Chưa xác nhận |

Ba phần nội dung đã được thống nhất là: dữ liệu và đặc trưng; mô hình và huấn luyện; đánh giá và phân tích. Nhóm có thể dùng cách chia này để trình bày, nhưng chỉ nên điền tên thành viên sau khi đối chiếu với công việc thực tế.

## Khai báo hỗ trợ AI để nhóm rà soát

Codex (OpenAI) đã hỗ trợ giải thích phương pháp, viết và sửa mã, tạo kiểm thử, tính toán từ dự đoán đã lưu, dựng bảng và hình, ghi chú audit và biên tập báo cáo. Người dùng tự thực hiện các lệnh train và test BERT. Phần đóng góp của từng thành viên cần được ghi riêng, không đồng nhất với công việc do công cụ hỗ trợ thực hiện.

Trước khi nộp, nhóm cần bổ sung các công cụ hoặc mô hình khác đã sử dụng, cho biết phần nội dung nào được giữ lại và cách kiểm tra lại kết quả. Khai báo cuối cùng cần phù hợp quy định môn học và được các thành viên xác nhận. Riêng các nhận xét về câu và bản sửa trong audit vẫn cần đọc lại; nhãn gốc được giữ nguyên.

## Kiểm trước nộp

- [x] Khóa cấu hình/primary seed trước BERT test; cùng split, không refit train+validation.
- [x] Giữ metric primary riêng với mean/std ba seed và bootstrap CI.
- [x] Giữ kết quả âm BERT đợt đầu cùng kết quả mở rộng; nêu test cổ điển đã được xem trước mở rộng.
- [x] Có source/output hashes, script kiểm chứng, bảng/hình và các tệp kết quả review.
- [ ] Nhóm điền tên/MSSV, đóng góp thực tế và chấp nhận khai báo AI.
- [ ] Người chuyên môn hoàn tất các review còn pending hoặc nhóm giữ rõ giới hạn.
- [ ] Đối chiếu mẫu/rubric, số trang, font, quy định slide và hạn nộp trên LMS.
- [ ] Kiểm PDF/slide trên máy dùng trình bày và thực hiện nộp bài; chưa được thực hiện thay người dùng.

Để chuẩn bị thuyết trình, mỗi thành viên nên chọn một mẫu có ID rõ ràng và giải thích được từ câu đầu vào đến đặc trưng hoặc encoding, dự đoán và nhãn gốc. Nhóm cũng cần nắm cách chia theo nhóm để hạn chế rò rỉ, lý do dùng macro-F1, sự khác nhau giữa độ lệch chuẩn theo seed và CI bootstrap, cùng một ca dự đoán sai đã tự đọc. Nếu tiếp tục phát triển sau báo cáo, nên dùng dữ liệu đánh giá mới thay vì tinh chỉnh theo test đã xem.
