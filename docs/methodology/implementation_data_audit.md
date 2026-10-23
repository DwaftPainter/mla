# Audit code–protocol và dữ liệu tuần 1

Ngày 23/10/2026. Không thay baseline, nhãn, split hoặc test. Đối chiếu `nucle_features.py`, `nucle_ablation.py`, `nucle_logistic.py`, `nucle_benchmark.py` và protocol.

| Vấn đề | Bằng chứng và xử lý |
|---|---|
| Vocabulary 500 | Code giới hạn 500 unigram/bigram ở kênh pooled `L|span_ngrams`; bốn slot lexical trái/phải có categorical vocabulary riêng fit train. Khác thiết kế dự kiến một vocabulary chung. Báo rõ baseline đã chạy là thiết kế thực tế, không gọi mọi lexical feature bị giới hạn tổng 500. Không đổi feature rồi tái dùng điểm cũ. |
| N-gram theo vai trò | Kênh `span_ngrams` thực chứa candidate từ cả span và hai vùng context; không giữ chỉ báo vai trò cho từng n-gram. Các slot trái/phải vẫn có vai trò riêng. Tên key không có nghĩa chỉ span. |
| Scaling | Numeric feature giữ giá trị gốc; code không có z-score hoặc scaler đã fit. Không mô tả như đã chuẩn hóa. |
| Token alignment | Feature parse lại câu bằng spaCy; không dùng trực tiếp token list preparation. Kiểm tra mọi train/validation của v1/v2: offset khớp; số token giao span không có bất đồng với preparation. Đây không phải kiểm chứng POS/parse đúng ngữ nghĩa. |
| Masking | Xóa lexical span trước khi fit vocabulary mask riêng; giữ context và giữ parse câu gốc. Đây là feature masking, không phải thay câu rồi parse lại. Không tạo bigram nối trái–phải qua span bị che. |
| LR hội tụ | Dừng theo change tolerance trước giới hạn vẫn được protocol chấp nhận dù gradient norm chưa nhỏ hơn tolerance_grad. Không gọi là nghiệm tối ưu chính xác hoặc gradient-converged. |
| Sensitivity | Cùng IDs/split, train-only encoder riêng theo v1/v2, LR C=0.1 và class_weight=None, không retune. Khác input của 381 train và 96 validation. Không có exact input signature trùng qua train/validation ở cả hai phiên bản. |
| Refit v2 | Chẩn đoán trên chính feature matrix lưu: 4 threads tái tạo toàn bộ prediction sensitivity, 10 threads tái tạo toàn bộ prediction baseline cũ. LBFGS dừng theo change tolerance ở số iteration khác nhau; khác biệt 2/2.368 prediction liên quan cấu hình số học/threads. Giữ điểm gốc trong bảng BERT, ghi threads trong lần tái lập; không dùng chênh lệch nhỏ làm bằng chứng v2 tốt hơn. Bằng chứng: audit/thread_diagnostic.json. |
| BERT | Là BERT nhỏ 2-layer trên CPU, không phải BERT-base của bài Nagata. Protocol riêng ghi trước fine-tuning; nguồn checkpoint/revision và tokenizer đã lưu. |

## Diễn giải audit ngôn ngữ

Đã đọc 18 lỗi validation chọn có chủ đích (không ngẫu nhiên), đối chiếu câu gốc, span, parse và correction sau khi có prediction. Ghi chú bằng annotation ID ở `reports/contextual_model_validation/audit/manual_review.json`; câu gốc và parse chỉ nằm trong packet local bị Git ignore.

- Annotation 11501, 12050, 12301: lỗi thêm determiner được gán span danh từ **không rỗng**. Vì vậy 0 span rỗng không có nghĩa không có lỗi thiếu từ.
- Annotation 12314, 14939: ranh giới gold Vform/SVA không hoàn toàn trùng với phân loại ngữ pháp nhìn từ edit; không tự đổi nhãn.
- Annotation 12019: parse đã có chủ ngữ và số/ngôi phù hợp để phát hiện bất hòa, nhưng LR vẫn nhầm. Có tín hiệu không đồng nghĩa classifier sử dụng hiệu quả.
- Annotation 11696, 19068: nghi ngờ POS/morph tự động không phản ánh đúng dạng/ngữ nghĩa trong câu lỗi; cần người chuyên môn xác minh nếu muốn kết luận chắc chắn.
- Span rộng như 1269, 19065, 19069 chứa nhiều cấu trúc và chỉ một nhãn; không thể quy sự giảm điểm cho số token đơn thuần.

Đây là nhận xét hỗ trợ bằng máy, không phải adjudication độc lập của chuyên gia; không suy tỷ lệ các nguyên nhân trên toàn corpus và không coi phân loại nguyên nhân là chứng minh nhân quả.
