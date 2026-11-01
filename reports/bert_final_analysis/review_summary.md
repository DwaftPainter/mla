# Kết quả audit đã được nhóm rà soát và chốt

Cập nhật 01/11/2026. Nhóm đã rà soát và chốt toàn bộ 18 review validation, 40 review BERT/LR và 22 nhận định xử lý. Báo cáo này chỉ công bố số đếm, ID và kết luận tổng hợp; câu gốc, correction đầy đủ và hồ sơ từng ca được giữ nội bộ.

## 1. Phạm vi đã hoàn thành

| Phần việc | Kết quả | Trạng thái |
| --- | --- | --- |
| Audit LR validation | 18/18 review, chia Quân, Duy, Công mỗi người 6 ca | Nhóm đã hoàn tất rà soát |
| Đọc ca BERT/LR trên test | 40/40 review và nội dung đóng góp của từng ca | Nhóm đã hoàn tất rà soát |
| Hàng đợi cần xem lại | 22/22 nhận định xử lý có căn cứ | Nhóm đã hoàn tất rà soát |

18 ca validation được chọn từ 364 dự đoán sai trên 2.368 mẫu: ba ca đầu theo ID ở mỗi chiều Vform→SVA, SVA→Vform, Nn→ArtOrDet và ArtOrDet→Nn, sau đó thêm sáu ca sai có span nhiều token. Mẫu chiếm khoảng 4,95% số ca sai và được chọn có chủ đích, không dùng để ước lượng tỷ lệ nguyên nhân trên toàn validation.

## 2. Kết quả bộ 40 ca BERT/LR

| Nhóm theo dự đoán và gold hiện tại | Số ca |
| --- | --- |
| BERT sai, LR đúng | 12 |
| BERT đúng, LR sai | 12 |
| Cả hai cùng sai | 10 |
| Cả hai cùng đúng, làm đối chứng | 6 |

Các nhận xét đã đối chiếu thao tác sửa với nhãn, ranh giới span và annotation lân cận. Nhóm cùng sai chứa chín ArtOrDet và một Nn do cách chọn mẫu; phân bố này không đại diện cho toàn bộ test. Có ca cùng đúng nhưng correction vẫn cần kiểm tra, nên khớp gold không tự xác nhận chất lượng bản sửa.

## 3. Kết quả 22 nhận định xử lý

| Hướng xử lý đã chốt | Số ca | ID |
| --- | --- | --- |
| Có cơ sở giữ nhãn theo bản sửa và ngữ cảnh | 9 | 18531, 37631, 24526, 9351, 41235, 35455, 211, 37624, 32801 |
| Cần xem lại sự phù hợp của nhãn | 6 | 34539, 12324, 26635, 32705, 37909, 16809 |
| Nghi vấn correction hoặc phạm vi thay thế | 3 | 18501, 23037, 30366 |
| Chỉnh sửa có thể tùy chọn | 3 | 201, 11916, 6079 |
| Bản sửa gồm nhiều thao tác | 1 | 32874 |

Hàng đợi gộp 17 ID cũ và sáu ID ưu tiên từ audit BERT, có một ID trùng. Tám ID cũng nằm trong bộ 40 ca, nên lượt đối chiếu mở rộng bao phủ 54 annotation khác nhau.

Những điểm được làm rõ:

- 32801 nhất quán khi đọc cùng các annotation bỏ mạo từ và chuyển danh từ đầy đủ sang số nhiều.
- 18531 được giải thích bởi các bản sửa cấu trúc và động từ đứng trước; không nên nhận xét target tách rời nhóm sửa.
- 23037 có ý định sửa số danh từ khá rõ nhưng phạm vi span và replacement cần kiểm tra trước khi áp dụng máy móc.
- 18501 và 30366 có nghi vấn về cấu trúc sau khi áp correction.
- 24526, 9351, 41235 và 35455 có dấu hiệu POS hoặc dependency chưa phù hợp với cách đọc câu; chưa thực hiện can thiệp thay parse để xác định nguyên nhân của prediction.

## 4. Cách hiểu kết quả đã chốt

Việc rà soát chốt nội dung nhận xét và hướng xử lý của nhóm. Các ca được đánh dấu cần xem lại vẫn giữ nguyên trạng thái vấn đề được nêu trong nhận định; không tự chuyển thành annotation đã được sửa. Gold, offset nguồn, prediction, checkpoint và các bảng điểm giữ nguyên. Không huấn luyện lại hoặc chấm lại test trong đợt cập nhật này.

Các kết luận mô tả bằng chứng từ mẫu chọn có chủ đích. Chúng giúp giải thích giới hạn biểu diễn span, ranh giới nhãn và tín hiệu parser; chưa chứng minh quan hệ nhân quả hoặc mức phổ biến trên toàn corpus.
