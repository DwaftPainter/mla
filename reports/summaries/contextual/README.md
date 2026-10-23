# BERT-small và chẩn đoán

| Báo cáo | Mốc / đọc để biết gì |
|---|---|
| [contextual_model_validation_report.docx](contextual_model_validation_report.docx) | Benchmark validation BERT-small, LR/POS, sensitivity preparation và audit. Nội dung ghi trước test cuối. |
| [bert_learning_diagnostics_report.docx](bert_learning_diagnostics_report.docx) | Train-only diagnostics: học thuộc 100 mẫu, gradient, span/marker và khuyến nghị tại thời điểm chẩn đoán. |

## Bổ sung khi đọc hai DOCX

Quyết định **hiện hành là dừng thử thêm BERT**, xem [status](../../../docs/status.md). Câu “nên thử cấu hình tối ưu hóa khác” trong DOCX diagnostics là đề xuất ở mốc trước, đã chuyển sang future work; DOCX không tự ghi quyết định dừng sau đó.

Benchmark dùng BERT-small 2 layers/hidden 128/2 heads, không phải BERT-base. Ba seed ở learning rate chọn cho validation macro-F1 khoảng 0,1984 ± 0,0030 (độ lệch chuẩn mẫu); không chọn riêng seed tốt nhất. Cả ba cấu hình học thuộc đạt 100% accuracy trên 100 train, nhưng CLS với LR 5e-5 chưa qua ngưỡng loss chặt. Hai kết quả thuộc hai thiết kế khác nhau.

Không có BERT test; kết quả chưa thành công không chứng minh ngữ cảnh vô ích. LR vốn dùng một phần dependency từ câu. Các Markdown tương ứng với hai DOCX không tồn tại trong workspace; không dẫn đến tên file dự kiến.

Artifact: [contextual validation](../../contextual_model_validation/README.md), [diagnostics](../../bert_learning_diagnostics/README.md). Protocol/literature: [docs](../../../docs/README.md). Cách đọc snapshot và kết quả mới hơn: [mục lục báo cáo](../README.md).
