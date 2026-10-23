# Protocol BERT tuần 1 — ghi trước fine-tuning

Ngày 23/10/2026. Phạm vi chỉ train/validation; không khóa mô hình cuối, không chấm test hoặc chạy bootstrap tuần 2.

## Điều chỉnh theo tài nguyên

PyTorch 2.14.0+cpu không có CUDA khả dụng. Chọn pretrained **BERT nhỏ 2 layer, hidden 128, 2 attention heads**, không gọi là BERT-base. Model `google/bert_uncased_L-2_H-128_A-2`, revision `30b0a37ccaaa32f332884b96992754e246e48c5f`; khoảng 4,39 triệu tham số sau thêm classifier/marker. Model card: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2 . Cơ sở kiến trúc: REF14; động cơ thử ngữ cảnh: REF11.

Pilot 20 bước train, không đánh giá validation, mất 2,314 giây (~0,116 giây/bước). Dự kiến khoảng 40 giây/epoch chưa tính validation và ghi checkpoint. Ngân sách bốn lượt, năm epoch/lượt; giới hạn dự kiến 30 phút cho fine-tuning. Nếu vượt ngân sách, ghi lượt thiếu/thất bại, không chọn lại theo test. Không dùng trọng số pilot cho lượt chính.

## Input và split

- Dùng preparation v2 và split hiện hành: 11.052 train, 2.368 validation. Không giữ hoặc chấm test trong dataloader.
- Input: `[CLS] left [SPAN_START] span [SPAN_END] right [SEP]`. Nhãn, correction, ID không đưa vào tokenizer/model; ID chỉ nối prediction với mẫu.
- Kiểm tra bounds và `sentence[start:end] == error_span`. Tokenize riêng ba vùng để giữ trọn span, kể cả partial-token. Marker thêm vào tokenizer, resize embedding; embedding marker và classifier được khởi tạo rồi học cùng encoder.
- Max length 128 gồm bốn special tokens. Giữ toàn bộ subword của span; chia ngân sách còn lại cho ngữ cảnh trái/phải gần span và phân bổ lại phần dư. Nếu span không vừa thì dừng, không âm thầm bỏ token. Dynamic padding mỗi batch.
- Pilot ghi 5 train và 6 validation bị cắt ngữ cảnh; span dài nhất 28/30 subword, không span nào bị cắt. Không đồng nhất “đọc ngữ cảnh” với đọc đủ câu ở mọi mẫu.

## Huấn luyện và chọn cấu hình

- `BertForSequenceClassification`, pooled CLS → dropout → linear năm lớp; cross-entropy không class weight, fine-tune toàn bộ tham số.
- CPU, 4 threads; batch 32; 5 epoch cố định, không early stopping; gradient clipping norm 1.
- AdamW, learning rate grid theo thứ tự `[3e-5, 5e-5]`; weight decay 0,01 trừ bias/LayerNorm; epsilon 1e-8, betas mặc định (0,9; 0,999).
- Linear warmup 10% tổng bước, sau đó linear decay. Shuffle train có seed; validation không shuffle.
- Hai learning rate chạy seed 20261013; chọn checkpoint bằng macro-F1 toàn validation, hòa chọn epoch sớm nhất. Chọn LR bằng điểm checkpoint, hòa chọn LR xuất hiện trước.
- LR đã chọn chạy thêm seed 20261014 và 20261015. Báo cả ba seed, mean và sample standard deviation; không chọn seed cao nhất đại diện. Seed chính cố định 20261013.
- Đây là exploratory validation sau nhiều bước phát triển, không phải kết quả generalization test. So với LR cố định và POS; không kết luận BERT nói chung kém/tốt từ một kiến trúc nhỏ.

## Đánh giá và artifact

Giữ năm nhãn theo thứ tự ArtOrDet, Prep, Nn, Vform, SVA; zero division = 0. Báo accuracy, macro-F1 và precision/recall/F1/support từng lớp, confusion matrix. Theo span: rỗng/một/nhiều token. Theo gold-label pair: Vform/SVA, Nn/ArtOrDet; giữ prediction năm lớp, tính cả prediction ra ngoài cặp. POS slices giữ VERB/AUX/NOUN/DET đã dùng ở Tuần 9; không đổi nhóm sau khi thấy BERT.

Lưu model/tokenizer checkpoint, config, history từng epoch, runtime, predictions, code/protocol/data/split hash và phiên bản phần mềm trong `reports/contextual_model_validation/bert/`. Không đổi baseline đã công bố bằng các điểm refit sensitivity. Kết quả ba seed là dao động huấn luyện, không thay confidence interval theo group.

Lệnh: `.venv/Scripts/python.exe src/training/bert_span_classifier.py --train`.
