# Chẩn đoán khả năng học BERT

| File | Vai trò |
|---|---|
| [summary.json](summary.json) | Kết quả học thuộc và quyết định |
| [gradient_checks.json](gradient_checks.json) | Gradient |
| [input_audit.json](input_audit.json) | Span/tokenization |
| [same_sentence_pairs.json](same_sentence_pairs.json) | Hai span cùng câu |
| [marker_sensitivity.json](marker_sensitivity.json) | Độ nhạy marker |
| [manifest.json](manifest.json) | Cấu hình/hashes trước chạy |

Các thư mục cls_* và span_mean_* chứa history, predictions và checkpoint train nhỏ. Giữ local; không gọi kết quả này là test BERT.

[Trở về mục lục reports](../README.md).
