# Mục lục báo cáo và artifact

Để đọc kết quả, bắt đầu từ [summaries/README.md](summaries/README.md). Báo cáo rà soát mới nhất là [evaluation_followup/review_report.md](evaluation_followup/review_report.md). Test đã chấm sau khóa; đã tái lập đợt 52 fit từ dữ liệu nguồn. Final report và slide tổng kết vẫn để sau.

## Báo cáo để đọc

| Nhóm | Điểm bắt đầu |
|---|---|
| Dữ liệu và preprocessing | [summaries/data](summaries/data/README.md) |
| LR/Tree và ablation validation | [summaries/classical](summaries/classical/README.md) |
| BERT-small và chẩn đoán học | [summaries/contextual](summaries/contextual/README.md) |
| Test, bootstrap, audit và tái lập | [summaries/evaluation](summaries/evaluation/README.md) |

## Artifact máy chạy — giữ đường dẫn ổn định

| Giai đoạn | Thư mục/file | Nội dung |
|---|---|---|
| Chất lượng nguồn | `nucle_annotation_audit.csv` (local), `nucle_data_quality.json` (local), `nucle_duplicate_content.json` (local) | Input audit/grouping có hash; không phải mẫu đã lọc |
| Preparation v1/v2 | [nucle_preparation_v2_comparison](nucle_preparation_v2_comparison/README.md) | So sánh dữ liệu, boundary cases, packet review local |
| Baseline từ loại | [nucle_pos_baseline_v1](nucle_pos_baseline_v1/README.md) | POS/Majority và predictions validation |
| Benchmark cổ điển | [nucle_feature_benchmark_solver_v2](nucle_feature_benchmark_solver_v2/README.md) | Grid Full LR/Tree và cấu hình chọn bằng validation |
| Ablation validation | [nucle_feature_ablation_v1](nucle_feature_ablation_v1/README.md) | Sáu variant lịch sử |
| Phân tích span ban đầu | [nucle_span_diagnostics_v1](nucle_span_diagnostics_v1/README.md) | Snapshot chẩn đoán cũ |
| Phân tích span bổ sung | [nucle_span_diagnostics_v2](nucle_span_diagnostics_v2/README.md) | Span/cặp nhãn/POS slices và report |
| Thử mô hình ngữ cảnh | [contextual_model_validation](contextual_model_validation/README.md) | BERT validation, sensitivity preprocessing và audit |
| Chẩn đoán BERT | [bert_learning_diagnostics](bert_learning_diagnostics/README.md) | Học thuộc train nhỏ, gradients, marker/span, checkpoints |
| Đánh giá test đã khóa | [final_evaluation](final_evaluation/README.md) | Lock, checkpoint, predictions, bootstrap, audit, gói tái lập |
| Rà soát và tái lập tiếp theo | [evaluation_followup](evaluation_followup/README.md) | Kiểm độc lập, tái lập raw → test, 17 ca chờ chuyên gia |
| Tổ chức tài liệu | [organization](organization/README.md) | Ánh xạ đường dẫn, backup và kiểm tra sau di chuyển |

Tên thư mục `final_evaluation` chỉ giai đoạn chấm test, không có nghĩa đã viết final report. Các artifact giữ nguyên tên và đường dẫn để code, checkpoint metadata và lock cũ tiếp tục hoạt động. Không sửa manifest cũ chỉ để làm đẹp cây thư mục.

Git chứa báo cáo Markdown/DOCX hiện hành, hình tổng hợp và các bảng JSON/CSV đã rà soát theo danh sách cho phép trong .gitignore. Checkpoint, encoder/vocab, predictions từng sample, packet chứa câu nguồn và các bản sao lịch sử vẫn giữ local. Đường dẫn ghi `(local)` chỉ có trong workspace nghiên cứu. Repository không bao gồm corpus hoặc toàn bộ gói tái lập; cần quyền sử dụng NUCLE và các input lịch sử được nêu trong protocol để tái lập đầy đủ. Code lịch sử đã khóa có thể xuất report ra vị trí cũ khi chạy lại riêng; dùng layout_manifest.json để đối chiếu, không chạy lại thí nghiệm chỉ để sắp xếp tài liệu.
