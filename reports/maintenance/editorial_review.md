# Biên tập ngôn ngữ báo cáo

Ngày 28/10/2026, các báo cáo được biên tập theo hướng rõ ràng và dễ đọc hơn, giữ tiếng Việt làm ngôn ngữ chính. Phần thay đổi tập trung vào cách diễn đạt, không bổ sung thí nghiệm hoặc điều chỉnh kết luận thống kê.

Trong báo cáo tổng kết, các đoạn tóm tắt, phương pháp, diễn giải kết quả và kết luận được viết lại thành những đoạn có mạch nối. Các thuật ngữ được giải thích ở lần xuất hiện đầu, còn mã, công thức và tên tệp được giữ khi cần đối chiếu. Báo cáo BERT được tổ chức quanh câu hỏi nghiên cứu, cách chọn cấu hình, kết quả và các ca sai khác, thay cho cách liệt kê theo log thực hiện.

Các báo cáo theo giai đoạn cũng được chỉnh lời dẫn và những đoạn giải thích quan trọng. Mốc thực nghiệm của từng tài liệu vẫn được giữ: chẳng hạn, báo cáo validation ban đầu chưa được viết lại như thể đã có test lúc đó. Phần khai báo hỗ trợ AI và các ca cần đánh giá độc lập vẫn còn trong tài liệu.

Phạm vi biên tập gồm 32 tệp Markdown hiện có và bảy báo cáo Word. Phần trình bày và ghi chú của 12 slide được viết lại để thuận tiện khi nói. Các bản Word/PDF được xuất từ văn bản đã biên tập; các lần xuất sau mặc định giữ bản Markdown này, thay vì tạo lại phần diễn giải từ bản nháp kỹ thuật.

Việc kiểm tra tập trung vào những nội dung cần bảo toàn:

- Bảng số liệu trong Markdown và cả bảy Word giữ nguyên so với bản trước biên tập.
- Số hình và các ví dụ lệnh thực thi được đối chiếu với bản lưu trước sửa.
- 354 tệp bằng chứng, gồm kết quả, dữ liệu kiểm chứng, checkpoint và dự đoán, giữ nguyên hash.
- Các nhận xét AI trong audit, cột review của nhóm và nhãn gốc không được sửa trong đợt này.

Chi tiết đối chiếu nằm trong [editorial_verification.json](editorial_verification.json). Bản trước sửa được lưu riêng trong reports/documentation_archive/before_language_edit_20261028 để có thể xem lại cách diễn đạt ban đầu. Việc biên tập không xác nhận đóng góp cá nhân, hoàn tất review chuyên gia hoặc thực hiện nộp bài thay nhóm.
