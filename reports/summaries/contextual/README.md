# BERT-small và chẩn đoán

Hai báo cáo Word dưới đây ghi lại đợt BERT-small đầu tiên và các phép chẩn đoán tiếp theo. Chúng giải thích kết quả chưa thành công ban đầu và lý do thử lại cách lấy biểu diễn, nhưng chưa chứa test của đợt mở rộng. [Báo cáo mới](../../bert_final_analysis/analysis_report.md) trình bày kết quả ngày 28/10 trên cùng split.

| Báo cáo | Mốc / đọc để biết gì |
|---|---|
| [contextual_model_validation_report.docx](contextual_model_validation_report.docx) | Benchmark validation BERT-small, LR/POS, sensitivity preparation và audit. Nội dung ghi trước test cuối. |
| [bert_learning_diagnostics_report.docx](bert_learning_diagnostics_report.docx) | Train-only diagnostics: học thuộc 100 mẫu, gradient, span/marker và khuyến nghị tại thời điểm chẩn đoán. |

## Bổ sung khi đọc hai DOCX

Hai DOCX dưới đây giữ đợt BERT cũ. Đợt mở rộng đã hoàn tất ngày 28/10/2026: 12 run, chọn span mean/head LR 1e−4, test primary0,8146; mean ba seed 0,8182 ± 0,0038. Đọc [báo cáo mới](../../bert_final_analysis/analysis_report.md) và [hướng dẫn pipeline](../../../docs/bert_small_training_guide.md). Chưa đủ bằng chứng BERT tốt hơn LR theo paired CI.

Benchmark dùng BERT-small 2 layers/hidden 128/2 heads, không phải BERT-base. Ba seed ở learning rate chọn cho validation macro-F1 khoảng 0,1984 ± 0,0030 (độ lệch chuẩn mẫu); không chọn riêng seed tốt nhất. Cả ba cấu hình học thuộc đạt 100% accuracy trên 100 train, nhưng CLS với LR 5e-5 chưa qua ngưỡng loss chặt. Hai kết quả thuộc hai thiết kế khác nhau.

Đợt cũ không có BERT test; đợt mở rộng có test và bootstrap riêng. LR vốn dùng dependency từ câu. Các Markdown tương ứng với hai DOCX lịch sử không tồn tại trong workspace; không dẫn tới tên file dự kiến.

Artifact: [contextual validation](../../contextual_model_validation/README.md), [diagnostics](../../bert_learning_diagnostics/README.md). Protocol/literature: [docs](../../../docs/README.md). Cách đọc snapshot và kết quả mới hơn: [mục lục báo cáo](../README.md).
