# Ghi chú thuyết trình

12 slides, khoảng 9–10 phút; chưa có thông tin thời lượng từ người dùng. Tên/MSSV và đóng góp cần nhóm xác nhận.

## Slide 1. Phân loại lỗi ngữ pháp trên NUCLE — khoảng 35 giây

Bài trình bày tập trung vào một câu hỏi cụ thể: khi đã biết đoạn nào trong câu có lỗi, mô hình có thể phân loại lỗi đó tốt đến đâu? Đây chưa phải hệ tự tìm và sửa lỗi. Bộ slide dự kiến khoảng 10 phút. Nhóm cần bổ sung họ tên và MSSV trước khi sử dụng để nộp bài.

## Slide 2. Bài toán và ba câu hỏi nghiên cứu — khoảng 45 giây

Ba câu hỏi đi từ hiệu quả tổng thể đến vai trò của thông tin đầu vào, rồi đến các lỗi còn lại. Nhãn dùng để học, còn bản sửa chỉ được đọc khi rà soát sau dự đoán; chúng không nằm trong đầu vào dự đoán. BERT được bổ sung sau để kiểm tra một cách biểu diễn khác trên cùng bài toán.

## Slide 3. Dữ liệu và tránh rò rỉ — khoảng 50 giây

Một câu có thể tạo nhiều mẫu nếu chứa nhiều annotation. Nếu chia ngẫu nhiên từng dòng, các mẫu có liên hệ có thể xuất hiện ở cả train và test. Chia theo nhóm giúp hạn chế tình huống này. Cũng vì sự phụ thuộc đó, khoảng tin cậy được lấy mẫu lại theo nhóm bài viết.

## Slide 4. Từ dữ liệu đến kết quả: ba phần công việc — khoảng 45 giây

Đây là cách chia nội dung thành ba phần để giải thích quy trình. Phần tạo đặc trưng thuộc bước dữ liệu. Biểu diễn Full của LR và Tree có 16.180 chiều; BERT học biểu diễn từ câu có marker, không nối thêm vector S/L/P/M/D. Phân công theo tên cần được xác nhận từ công việc thực tế.

## Slide 5. Thiết kế mô hình và lựa chọn — khoảng 45 giây

Các biến thể đặc trưng được chọn tham số riêng, nên chênh lệch chưa phải hiệu ứng nhân quả của một nhóm thông tin ở cùng bộ tham số. POS rule có bảng tra học từ train. LR cũng có dependency từ cả câu, vì thế đối chiếu với BERT không đơn giản là mô hình có ngữ cảnh và mô hình không có ngữ cảnh.

## Slide 6. Từ BERT ban đầu đến đợt mở rộng — khoảng 50 giây

Thử 100 mẫu giúp kiểm tra khả năng tối ưu hóa, không cho biết mô hình sẽ làm tốt trên dữ liệu chưa học. Vì vậy, đợt mở rộng bắt đầu lại từ pretrained và đánh giá trên toàn validation. Khi giải thích cải thiện, cần nhớ ngoài readout còn có thời gian huấn luyện và learning rate của head thay đổi.

## Slide 7. Chọn BERT bằng validation, không chọn seed test — khoảng 50 giây

Cấu hình được chọn bằng kết quả trung bình của cùng ba seed trên validation. Seed chính đã cố định từ trước, nên không thay bằng seed 20261015 dù seed đó có điểm test cao nhất. Span mean lấy trung bình biểu diễn của các token trong đoạn lỗi sau khi encoder đã xử lý cùng ngữ cảnh.

## Slide 8. Kết quả test trên cùng split — khoảng 50 giây

Các điểm trên hình đều dùng cùng 2.368 mẫu test. Điểm BERT là của seed chính đã khóa, còn trung bình ba seed được báo riêng. Accuracy đếm mọi mẫu ngang nhau, trong khi macro-F1 cho mỗi lớp trọng số ngang nhau; vì thế hai chỉ số có thể dẫn đến thứ tự khác nhau.

## Slide 9. Chênh lệch có chắc chắn không? — khoảng 55 giây

Điểm cần nhìn là CI của chênh lệch, không phải hai CI riêng có chồng nhau hay không. Cùng một lượt lấy lại nhóm được dùng cho mọi mô hình. CI chứa 0 nghĩa là chưa đủ bằng chứng xếp hạng chắc chắn, không chứng minh hai mô hình bằng nhau. Phép tính cũng chưa bao quát biến thiên do split mới hoặc nhiều seed huấn luyện.

## Slide 10. Điểm mạnh và điểm yếu theo lớp — khoảng 45 giây

Biểu đồ theo lớp cho thấy hai mô hình có những đánh đổi khác nhau dù điểm trung bình gần nhau. SVA đáng được đọc kỹ vì chênh lệch lớn hơn các lớp còn lại. Tuy nhiên, chỉ nhìn dự đoán chưa đủ để kết luận attention hay parser là nguyên nhân gây sai.

## Slide 11. Audit lỗi và giới hạn suy luận — khoảng 50 giây

Có thể dùng CSV trên máy để giải thích một ca đã tự đọc. ID 38155 là mảnh từ nhưng cả hai mô hình vẫn đúng, giúp tránh suy rằng mọi partial-token đều gây lỗi. ID 34539 cho thấy cần xem quan hệ giữa nhãn và bản sửa. Các ca được chọn có chủ đích, không phải mẫu ngẫu nhiên đại diện corpus.

## Slide 12. Kết luận và bàn giao — khoảng 45 giây

Kết quả cho thấy cách biểu diễn và thiết lập huấn luyện đều đáng quan tâm, nhưng chưa cho phép tuyên bố mô hình hiểu ngữ pháp đầy đủ hoặc tốt trên mọi corpus. Đợt mở rộng dùng lại test đã biết từ các mô hình cổ điển, nên nghiên cứu tiếp nên có dữ liệu bên ngoài. Phần còn lại của nhóm là xác nhận thông tin và tự kiểm tra tài liệu trước khi nộp.
