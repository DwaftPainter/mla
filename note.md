# Kế hoạch và nhật ký công việc

Cập nhật 23/10/2026. **Các task thực nghiệm của hai tuần đã hoàn thành trong các phiên ngày 23/10; các ngày 23/10–05/11 bên dưới là lịch dự kiến ban đầu, không phải ngày chạy thực tế.** Hiện tập trung bàn giao tài liệu; final report dự án và slide tổng kết để sau. Trạng thái chi tiết quản lý tại [docs/status.md](docs/status.md).

## 1. Việc còn lại và quyết định hiện hành

| Việc | Trạng thái / điều kiện |
|---|---|
| Review chuyên gia | 17 ca trong [expert_review_queue.csv](reports/evaluation_followup/expert_review_queue.csv) chờ người chuyên môn. Chưa đổi gold; nếu chưa có review, giữ giới hạn trong báo cáo. |
| Tài liệu và Git | Đã công bố code/docs ở 52349bd; lượt này làm rõ README, đường chạy, mốc báo cáo và tài liệu còn thiếu rồi push bản cập nhật. Không đưa corpus/private artifacts lên Git. |
| Code cleanup | Đã có [danh mục rà soát](reports/maintenance/code_redundancy_review.md); chưa xóa hoặc sửa code. Ưu tiên phần ngoài lock, bảo toàn snapshot đã chạy. |
| Sao lưu riêng | Chưa xác nhận nơi lưu độc lập cho corpus/private artifacts; không coi Git công khai hoặc bản restore cùng máy là thay thế. |
| Final report và slide | Chưa viết/chưa chốt theo yêu cầu hiện hành; không tự mở rộng task tài liệu thành báo cáo nộp cuối. |
| BERT / Excel | Dừng thử thêm BERT; không chạy BERT test/BERT-base. Không có task sửa workbook hoặc PRISMA. |

Sau khi xem test, ý tưởng mới chỉ ghi future work; không tune lại cấu hình hoặc sửa nhãn theo lỗi test. LR vẫn là mô hình chính đã chọn trên validation; BERT-small được báo như thử nghiệm chưa thành công, không kết luận ngữ cảnh vô ích.

## 2. Lịch hai tuần ban đầu và kết quả thực tế

### Tuần 1 — dự kiến 23–29/10/2026

| Ngày dự kiến | Task | Kết quả thực tế |
|---|---|---|
| 23/10 | Đọc bài, mapping input/output/metric và ứng dụng vào report | Đã tổng hợp [related work](docs/literature/contextual_models_related_work.md) và [reference](reference.md). Không sửa Excel. |
| 24–25/10 | Audit validation, offset/parser, sensitivity preparation v1/v2 | Đã có audit 18 ca, sensitivity cùng split và chẩn đoán threads; [implementation audit](docs/methodology/implementation_data_audit.md). |
| 25/10 | Ghi protocol BERT, pilot tài nguyên và marker/cropping | Đã ghi [protocol](docs/bert_experiment_protocol.md) trước fine-tuning. |
| 26–28/10 | Fine-tune BERT-small trên train, chọn validation | Đã chạy hai learning rate ở seed đầu, thêm hai seed cho learning rate chọn: bốn run, không chọn riêng seed tốt nhất. |
| 29/10 | So sánh toàn validation, từng lớp/span/cặp nhãn/POS | Đã có bảng, mean/std và report; [báo cáo contextual](reports/summaries/contextual/README.md). |

Các task trên thực hiện trong phiên 23/10, không giả định đã trôi qua bảy ngày. BERT dùng google/bert_uncased_L-2_H-128_A-2 (2 layers, hidden 128, 2 heads), không phải BERT-base.

Sau benchmark đã bổ sung diagnostics theo yêu cầu: ba cấu hình đều nhớ nhãn 100 mẫu train; hai cấu hình đạt ngưỡng cross-entropy chặt. Không thấy gradient bị đứt trong probes; checkpoint CLS cũ chỉ đúng cả hai target ở 2/50 cặp cùng câu có chủ đích. Đề xuất thử tiếp trong DOCX diagnostics thuộc mốc cũ; quyết định sau đó là dừng BERT trong phạm vi này.

### Tuần 2 — dự kiến 30/10–05/11/2026

| Bước / ngày dự kiến | Task đã chốt | Kết quả thực tế |
|---|---|---|
| 1 / 30–31/10 | Bổ sung Full−P/M/D cho LR/Tree; chốt trước test và khóa | 40 grid fits Full−P/M + 12 refit cũ; Full−D alias S+L+P+M nên không thêm 20 fit. Validation replay khớp; [protocol](docs/final_evaluation_protocol.md), [lock](reports/final_evaluation/lock.json). |
| 2 / 01/11 | Chấm test các pipeline đã chọn, gồm Majority/POS | 20 tên pipeline, 16 checkpoint LR/Tree riêng + hai baseline; test 2.368 mẫu. Không chấm BERT. |
| 3 / 02/11 | 1.000 paired connected-group bootstrap | CI F1 và 30 contrast/slice đủ điều kiện; seed 20261023, 97 test groups. Không hiệu chỉnh kiểm định đồng thời. |
| 4 / 03/11 | Error analysis và tổng hợp BERT đã có | Audit 25 lỗi LR, 5 lỗi Tree khi LR đúng, 10 controls; tổng 40 ca. BERT chỉ dùng kết quả trước đó. |
| 5 / 04–05/11 | Báo cáo và kiểm tái lập | Phạm vi đã đổi thành **report task**, không final report/slide. Đã có DOCX/Markdown, checkpoint replay và gói restore; sau đó bổ sung tái lập raw → 52 fit → test. |

Toàn bộ task thực nghiệm đã chạy ngày 23/10. Các con số dự kiến 60 fit bổ sung và tối đa 50 lỗi + 10 controls được thay bằng phạm vi thực chạy ở bảng trên; không diễn giải kế hoạch thành số đã chạy. Output: [báo cáo đánh giá](reports/summaries/evaluation/README.md).

## 3. Bằng chứng trả lời RQ1–RQ3

| Câu hỏi | Đầu ra đã có |
|---|---|
| RQ1: LR/Tree so với Majority | Benchmark Full trên test, per-class metric, CI, paired differences; POS là đối chứng bổ sung. |
| RQ2: Nhóm feature | Chín tên variant/model: S+L, S+L+P, S+L+P+M, Full, masked, non-lexical, Full−P/M/D; Full−D reuse. |
| RQ3: Nhóm nhầm và đặc điểm liên quan | Confusion, span bins, gold-label pairs, POS slices và audit theo ID. |

Bảng số và phạm vi kết luận nằm trong [report rà soát](reports/evaluation_followup/review_report.md); không sao chép thêm một bộ bảng kết quả vào kế hoạch.

## 4. Rà soát/tái lập sau đánh giá — đã làm

- Kiểm độc lập 280 model–slice metrics, 40 CI F1, 30 CI chênh lệch; đối chiếu CSV và 11 bảng định lượng trong report.
- Kiểm lại nguồn/offset/parser 40 ca; giữ 17 ca trong hàng chờ chuyên gia, không tự đổi nhãn.
- Tái lập SGML → preparation v2 → grouping/split → features → 52 fit → test replay trong môi trường riêng; outputs/checkpoints/predictions khớp. Chưa rerun toàn grid lịch sử hoặc BERT.
- Tổ chức docs/reports, kiểm 5 DOCX và sửa liên kết. Các DOCX/protocol lịch sử giữ nội dung mốc chạy; README từng nhóm giải thích phần cập nhật.
- Rà code thừa, sau đó rà tài liệu và bổ sung [hướng dẫn tái lập](docs/reproduction.md). Không sửa implementation trong lượt rà tài liệu.

## 5. Quy tắc diễn giải tiếp tục áp dụng

- Cặp Vform/SVA và Nn/ArtOrDet chọn bằng gold label nhưng giữ predictions năm lớp; dự đoán ngoài cặp vẫn sai. Không đồng nhất cặp gold với slice theo POS.
- Span rỗng có support 0: ghi N/A, không tự tạo sample. Lỗi thiếu từ vẫn có thể được annotation bằng span không rỗng.
- LR có dependency từ parse toàn câu; không mô tả nó là chỉ nhìn hai từ mỗi phía. Bỏ nhóm P/M không xóa mọi thông tin POS/morph có thể còn trong D.
- Tuning riêng theo variant đo hiệu quả pipeline dưới cùng ngân sách, không chứng minh quan hệ nhân quả của feature.
- Test đã mở; review ngôn ngữ là phân tích sau prediction. Các kết quả học thuộc train nhỏ không phải validation/test score.
