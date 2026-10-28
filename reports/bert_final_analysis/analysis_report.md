# BERT-small: kết quả mở rộng và so sánh với các mô hình đối chứng

Báo cáo ngày 28/10/2026 tổng hợp kết quả của đợt BERT-small mở rộng. Mục đích là xem cách lấy biểu diễn tại span và cấu hình huấn luyện có giúp cải thiện kết quả ban đầu hay không, đồng thời so sánh BERT với LR, Tree và hai đối chứng trên cùng tập test. Các phép tính dưới đây sử dụng dự đoán đã lưu; mô hình, nhãn và seed chính được giữ nguyên.

## Cách huấn luyện và chọn cấu hình

Đợt thử gồm 12 lượt: hai cách lấy biểu diễn CLS và span mean, hai head learning rate 1e−4 và 1e−3, lặp lại với ba seed. Encoder learning rate giữ ở 5e−5. Mỗi lượt chạy tối đa 20 epochs, có dừng sớm khi validation không cải thiện đủ mức trong bốn epochs liên tiếp.

Span mean với head learning rate 1e−4 được chọn nhờ macro-F1 validation trung bình cao nhất. Seed chính là 20261013, được xác định trước khi xem test. Các phép kiểm về lựa chọn, đầu vào, mã nguồn, checkpoint và dự đoán đã lưu đều khớp: 61 phép kiểm đã qua.

Các mô hình dùng cùng 2.368 mẫu test và 97 nhóm bài viết. Khoảng tin cậy được tính từ cùng 1.000 lượt bootstrap theo nhóm, nhờ đó chênh lệch giữa các mô hình được so sánh trên cùng mẫu lấy lại. Độ lệch chuẩn giữa seed được báo riêng vì nó trả lời một câu hỏi khác với CI bootstrap.

## Kết quả trên test

| Model | Val F1 | Test F1 | Accuracy | CI95% |
| --- | --- | --- | --- | --- |
| Majority | 0.1184 | 0.1184 | 0.4206 | [0.1104; 0.1279] |
| POS rule | 0.7048 | 0.7106 | 0.7441 | [0.6766; 0.7365] |
| LR Full | 0.8067 | 0.8163 | 0.8357 | [0.7876; 0.8382] |
| Tree Full | 0.7879 | 0.7996 | 0.8095 | [0.7693; 0.8246] |
| BERT-small primary | 0.8173 | 0.8146 | 0.8438 | [0.7919; 0.8350] |

BERT ở seed chính đạt macro-F1 0,8146 và accuracy 0,8438. Kết quả trung bình qua ba seed là 0,8182 ± 0,0038. Seed có điểm test cao nhất không được dùng để thay seed chính; nếu làm vậy, phép lựa chọn sẽ phụ thuộc vào test.

| Contrast | Δ F1 | CI95% |
| --- | --- | --- |
| BERT primary - LR Full | -0.0017 | [-0.0184; 0.0182] |
| BERT primary - Tree Full | 0.0149 | [-0.0049; 0.0378] |
| BERT primary - POS rule | 0.1040 | [0.0785; 0.1343] |
| BERT primary - Majority | 0.6962 | [0.6702; 0.7175] |

CI của BERT−LR và BERT−Tree đều chứa 0. Vì vậy, dù điểm số gần nhau, dữ liệu chưa đủ để kết luận BERT tốt hơn hay kém hơn hai mô hình này. Với POS, CI nằm hoàn toàn trên 0, cho thấy mức cải thiện rõ hơn trong phép so sánh đã thực hiện.

Các khoảng này được tính riêng cho từng phép so sánh, trên một split đã khóa. Ngoài ra, đợt BERT mở rộng diễn ra sau khi kết quả test cổ điển đã được biết. Đây là kết quả bổ sung trên cùng tập test, chưa thay thế một phép đánh giá độc lập trên dữ liệu bên ngoài.

![Macro-F1 test và khoảng tin cậy của các mô hình. BERT dùng seed chính đã khóa.](figures/benchmark_ci.png)

![F1 theo từng loại lỗi, giúp nhìn rõ những đánh đổi mà điểm trung bình có thể che khuất.](figures/per_class_f1.png)

## Những khác biệt đáng chú ý

Trên toàn bộ test, BERT và LR cùng đúng 1.860 ca. BERT đúng riêng 138 ca, LR đúng riêng 119 ca, còn 251 ca cả hai cùng sai. BERT làm tốt hơn LR ở bốn lớp nhưng yếu hơn ở SVA: F1 là 0,7392 so với 0,8106. Ngược lại, ở Nn, BERT đạt 0,8098 còn LR đạt 0,7781. Những khác biệt theo lớp giải thích vì sao accuracy của BERT cao hơn trong khi macro-F1 ở seed chính thấp hơn một chút.

Để đọc kỹ các sai khác này, phần audit chọn 40 ca: 12 ca BERT sai/LR đúng, 12 ca theo chiều ngược lại, 10 ca cả hai sai và sáu ca cả hai đúng để đối chiếu. Sáu ca gold SVA được ưu tiên trong nhóm đầu. ID được chọn trước khi đọc câu và bản sửa; nhãn cùng span của cả 40 ca đều khớp nguồn SGML.

Mỗi ca đã có ghi chú AI, còn cột review và contribution của nhóm vẫn để trống. Đây là mẫu chọn có chủ đích: nhóm cùng sai có 9 ArtOrDet và 1 Nn do thứ tự duyệt bucket. Không thể dùng tỷ lệ trong mẫu này để suy tỷ lệ toàn corpus. Các nhận xét cũng chưa xác định nguyên nhân nhân quả hoặc thay thế đánh giá độc lập của chuyên gia.

Một số ca cho thấy span chỉ là mảnh từ, hoặc lỗi thiếu từ được neo vào từ bên cạnh. Một số khác cần xem lại quan hệ giữa nhãn và bản sửa. Sáu ID 23037, 34539, 16809, 30366, 32801 và 32874 được ưu tiên cho chuyên gia. Ca partial-token được cả hai mô hình dự đoán đúng cũng được giữ làm đối chiếu, tránh diễn giải rằng mọi mảnh từ đều gây lỗi.

CSV có câu gốc và bản sửa nằm tại `private/error_review_with_ai.csv`; `error_review_template.csv` là bản để tự review. Các tệp chứa corpus được giữ trên máy nghiên cứu. Nhãn gốc không thay đổi sau test.

## Đọc và kiểm tra lại kết quả

Các tệp chính gồm:

- `overall_results.csv`, `per_class_metrics.csv`: kết quả tổng thể và từng lớp.
- `validation_grid.csv`, `seed_results.csv`: cách lựa chọn cấu hình và kết quả theo seed.
- `bootstrap.json`, `paired_differences.csv`: khoảng tin cậy và chênh lệch paired.
- `verification.json`: kết quả kiểm hash và tính lại metric.
- `audit_summary.json`: cách chọn ca và trạng thái review; phần có câu nguồn nằm trong `private/`.

Có thể tính lại phần phân tích vào thư mục mới bằng `python -m src.evaluation.bert_final_analysis --output reports/bert_final_analysis_recheck`. Script từ chối ghi đè để giữ bằng chứng đã lưu. Ghi chú ngôn ngữ là quyết định review, nên không thể được tự tái tạo giống một phép tính metric.

`source_manifest.json` ghi các tệp tại lúc tạo phân tích; `delivery_manifest.json` ghi nguồn và đầu ra của bộ báo cáo hiện tại. Với đợt cổ điển, workspace còn thiếu bốn source và hai CSV đã đổi định dạng. Chuẩn hóa CSV trong bộ nhớ tái tạo được hash gốc và metric/CI vẫn khớp, nhưng verify_lock hiện tại chưa qua. Các manifest lịch sử được giữ nguyên để phản ánh đúng mốc của chúng.
