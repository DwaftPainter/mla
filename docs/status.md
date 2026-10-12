# Trạng thái dự án

Cập nhật: 12/10/2026.

## Đã hoàn thành

- Preparation v1 được giữ làm baseline tái lập; v2 sửa quy tắc tách câu và giữ nguyên toàn bộ 15.788 mẫu thuộc năm nhãn mục tiêu.
- So sánh v1/v2: 544 câu đầu vào và 533 sentence ID thay đổi; 30 mẫu đổi token context; metadata nguồn, source offset và trạng thái giữ/loại không đổi.
- V2 có 14.447 span khớp trọn token, 1.130 span cắt token một phần, 211 span chạm whitespace; không có mẫu mục tiêu không dựng được.
- Hai boundary challenge 31297 và 39538 vẫn trong dữ liệu chính, được gắn cờ unresolved và dùng làm fixture kiểm thử.
- Hash đầu ra chuẩn: v1 `f9597cd007aef5614f208c5db76889f8b867f5265478707e93b24711fac67f59`; v2 `9f9fc5405492d03133bab2a89759af3504028d05568829d27a6cdc47838256a0`.
- 73 kiểm thử dữ liệu đạt. Full pytest chưa thu thập được kiểm thử model/training vì thiếu `src.data.dataset`.

## Câu hỏi nghiên cứu còn bỏ ngỏ

Phân bố nhãn đã biết: ArtOrDet 6.640 (42,06%), Nn 3.768 (23,87%), Prep 2.413 (15,28%), SVA 1.524 (9,65%), Vform 1.443 (9,14%). Đây mới là phân bố, chưa phải kết quả dự đoán.

Chưa biết cặp nhãn nào thường bị nhầm: chưa có split và dự đoán đánh giá; module confusion matrix chỉ là tiện ích tính toán, chưa có kết quả thực nghiệm.

Chưa biết POS của token lỗi có gần như quyết định nhãn hay không. POS mới là feature dự kiến, chưa có tagger/version cố định, POS × label thống kê hoặc baseline luật đã chạy. Vì vậy giả thuyết `ADP → Prep`, `DET → ArtOrDet` chưa được xác nhận; cần so với Majority trên cùng test set để đánh giá giá trị thực.

## Việc cần làm theo thứ tự

1. **Bước tiếp theo: rà soát và chốt chính sách nhóm duplicate/boilerplate.** Báo cáo chất lượng nêu 65 nhóm paragraph trùng chính xác và 62 nhóm trùng sau chuẩn hóa; đây là ứng viên cần kiểm tra, chưa phải số nhóm split đã xác nhận. Quyết định quy tắc cho nội dung boilerplate/ngắn và near-duplicate, rồi lưu grouping ID cùng lý do để tránh rò rỉ.
2. Tạo split theo essay hoặc nhóm nội dung đã chốt, kiểm tra không nhóm nào đi qua nhiều partition, rồi khóa seed và manifest.
3. Trước khi xem test, cố định POS tagger/version và cách gán POS cho span nhiều token, partial-token, POS thiếu hoặc xung đột. Định nghĩa POS-only rule (dự kiến ADP→Prep, DET→ArtOrDet, còn lại chọn lớp đa số train) cùng Majority.
4. Chạy Majority, POS rule, LR và Tree trên cùng split; báo cáo metric từng lớp và confusion matrix để xác định các cặp nhầm. Sau đó mới kết luận POS baseline có mạnh và mô hình có vượt baseline hay không.

## Hồ sơ chính

- [Quy trình preparation](preparation.md)
- [Protocol thí nghiệm](experiment_protocol.md)
- [Báo cáo chất lượng dữ liệu](../reports/nucle_data_quality.md)
- [Báo cáo so sánh preparation v1/v2](../reports/nucle_preparation_v2_comparison/comparison_summary.json)
