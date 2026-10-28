# Rà soát chất lượng dữ liệu NUCLE

Phép kiểm xem xét **44,912** annotation trong **1,397** bài viết, trong đó **15,788** annotation thuộc năm nhãn nghiên cứu. CSV và SGML khớp nhau. Có **455** cặp span chồng lấp; đây là số cặp, không phải số annotation bị gắn cờ. Kiểm tra nội dung tìm thấy 65 nhóm đoạn văn trùng hoàn toàn và 62 nhóm trùng sau chuẩn hóa, tương ứng các khóa paragraph_exact và paragraph_normalized.

Bảng dưới ghi số annotation có từng cờ trên toàn nguồn và trong tập nhãn mục tiêu. Một annotation có thể mang nhiều cờ, nên không cộng các cột để suy ra số mẫu cần xem lại.

| Quality flag | All annotations | Target annotations |
| --- | ---: | ---: |
| invalid_offset | 0 | 0 |
| cross_paragraph | 4 | 0 |
| missing_label | 0 | 0 |
| missing_correction | 0 | 0 |
| no_op | 30 | 6 |
| empty_edit | 0 | 0 |
| long_span | 387 | 2 |
| same_span_different_label | 47 | 11 |
| duplicate_candidate | 0 | 0 |
| overlapping_span | 531 | 170 |

Các cờ là điểm bắt đầu để rà soát, chưa phải kết luận annotation sai. Mỗi annotation vẫn được giữ riêng với nhãn và vị trí gốc. Phép kiểm không tự xóa hoặc gộp những span chồng lấp, cùng span khác nhãn hay nội dung nghi trùng.

Hai span chỉ chạm đầu mút không được tính là chồng lấp, theo quy ước khoảng ký tự nửa mở. Điểm chèn nằm bên trong một span hoặc hai điểm chèn cùng vị trí được gắn cờ để xem thêm. Văn bản và offset nguồn không bị chuẩn hóa; thứ tự annotation trong CSV được giữ nguyên.

Chi tiết theo ID nằm trong reports/nucle_annotation_audit.csv. Các nhóm nội dung trùng nằm trong reports/nucle_duplicate_content.json; chúng chưa phải nhóm dùng để chia tập. Nhãn thiếu, offset sai, bản sửa không làm thay đổi câu, boilerplate ngắn và các trường hợp gần trùng vẫn cần đọc nguồn hoặc ngữ cảnh để đánh giá.

Đây là bước kiểm tra dữ liệu trước mô hình. Nó chưa tạo đặc trưng, chạy POS baseline, tách câu hoặc chia train/validation/test.
