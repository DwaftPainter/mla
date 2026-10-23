# Protocol chẩn đoán khả năng học và biểu diễn span

Ngày 23/10/2026; ghi trước chạy. Chỉ lấy mẫu từ train; không tính thêm metric validation/test, không thay baseline LR hoặc kết quả BERT trước đây.

## Câu hỏi và kiểm tra

1. Tokenizer có giữ đúng marker và toàn bộ span không? Kiểm toàn train, ID marker, padding, ánh xạ nhãn và collision input sau tokenization. Không đưa correction/ID/nhãn vào input.
2. Gradient có hữu hạn, khác 0 và đi đến classifier, pooler, các layer encoder, embedding marker/target không? Đo gradient trước clipping và thay đổi tham số sau một bước. So loss với cross-entropy tính tay và logits với forward gốc. Gradient trên **hidden state cuối tại target bằng 0** có thể là hệ quả trực tiếp của CLS readout, không tự chứng minh mất gradient encoder.
3. BERT có học thuộc 100 mẫu train cân bằng (20/lớp) không? Chọn theo seed 20261023, loại input tokenized trùng khỏi tập chẩn đoán. Chọn trước khi chạy, không dựa trên prediction. Tập này không đại diện phân bố dữ liệu thật.
4. Đổi vị trí marker trên cùng chuỗi token có thay đổi CLS/logits không? Dùng checkpoint đã huấn luyện và train subset. Đây là perturbation nhân tạo, không phải ground-truth để chấm accuracy. So cả pooled CLS và mean hidden của target qua cùng pooler/classifier để xem cơ chế đọc thông tin; không dùng điểm variant chưa huấn luyện làm benchmark.

## Thí nghiệm học thuộc

- Giữ BERT 2-layer/hidden128, checkpoint/revision và tokenizer cũ. Fine-tune toàn bộ tham số; batch20, CPU4threads, seed20261023; max400 update, đánh giá trên **chính 100 mẫu train** mỗi25 update.
- Bỏ dropout và weight decay; constant learning rate, không scheduler/warmup. Đây là điều kiện dễ để kiểm tra khả năng tối ưu hóa, không tái lập nguyên lịch huấn luyện cũ.
- Ba lượt theo thứ tự, đều khởi tạo từ cùng pretrained weights và cùng classifier/marker seed; cùng thứ tự minibatch:
  - CLS, learning rate 5e-5.
  - CLS, learning rate 1e-3.
  - Span mean, learning rate 1e-3: lấy trung bình hidden của token giữa marker, loại marker/padding, qua cùng dense+tanh pooler và classifier như CLS. Không thêm tham số.
- Đạt tiêu chí học thuộc khi accuracy=1,0 và cross-entropy≤0,05 ở hai lần kiểm liên tiếp; dừng sớm khi đạt. Nếu hết400 update chưa đạt, báo thất bại trong ngân sách này; không kết luận không thể học thuộc.
- Clip gradient norm1. Lưu lịch loss/accuracy/macro-F1, checkpoint, prediction train và thông tin gradients. Không chọn seed tốt nhất hoặc tìm cấu hình bằng test.

## Quy tắc quyết định

- Nếu token/label/gradient sai: sửa cơ chế trước, chưa chạy benchmark mới.
- Nếu learning rate cao giúp CLS học thuộc còn rate cũ không: có căn cứ thử lại cách tối ưu hóa, chưa chứng minh learning rate cao tổng quát hóa tốt.
- Nếu span mean giúp học thuộc nhanh/ổn định hơn CLS: có căn cứ thử readout theo span với protocol mới trên train/validation, chưa coi đây là kết quả vượt LR.
- Dù học thuộc thành công, kết quả trên100 mẫu không phải metric validation và không được so trực tiếp với LR0,8067. Giữ LR là ứng viên mạnh nhất theo bằng chứng hiện có. BERT trước đây là thử nghiệm chưa thành công, không phải bằng chứng ngữ cảnh không hữu ích.
