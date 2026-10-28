# Kiểm tra reports trước đợt BERT-small mới

Báo cáo ngày 27/10/2026 kiểm tra các tệp hiện có, tính lại số liệu từ dự đoán đã lưu và đối chiếu bảng Word với Markdown. Ở mốc này, đợt BERT mới mới có mã và hướng dẫn để người dùng tự chạy; chưa có kết quả huấn luyện hoặc test mới. Việc rà soát không thay nhãn hoặc gọi lại mô hình để chấm test.

## Kết luận

Các số liệu đã lưu vẫn khớp khi tính lại. Tuy nhiên, thư mục làm việc đã thay đổi so với bản khóa: thiếu bốn source, hai CSV dự đoán đổi định dạng và một số manifest còn ghi các đường dẫn trước khi dọn tệp. Vì vậy, kết luận số liệu đúng cần được tách khỏi kết luận toàn bộ workspace còn nguyên byte.

Pipeline BERT mới đọc preparation v2, split và pretrained riêng; không phụ thuộc final lock cổ điển. Các phát hiện này không phải bằng chứng rằng BERT mới đã được train hoặc có điểm test.

## Kiểm tra số liệu và file

| Hạng mục | Kết quả |
|---|---|
| JSON hiện hành trong reports | 115 file đọc hợp lệ trước khi bổ sung báo cáo này; không tính documentation_archive/before_layout |
| Test predictions cổ điển | 2.368 IDs không trùng, 97 groups |
| Metric/confusion model–slice | 280 đối chiếu khớp, gồm slice rỗng giữ metric null |
| reviewed_overall_results.csv | 20 dòng khớp macro-F1, accuracy, macro-precision và macro-recall tính lại |
| Bootstrap | Weights tái tạo đúng từ seed; tính lại 1.000 replicates cho 40 CI model và 30 CI chênh lệch, không sai lệch quá 10⁻¹² |
| BERT validation cũ | Bốn run khớp predictions; ba seed tại LR chọn: macro-F1 0,1983711770 ± 0,0029738005 |
| BERT test | Chưa có; selection cũ vẫn test_scored=false |
| Audit validation | Packet, manual_review và CSV người dùng cùng đủ 18 IDs; review/contribution vẫn trống |
| Hàng chờ chuyên gia | 17 ca, không đồng nghĩa 17 gold sai |
| DOCX | Sáu file mở ZIP/XML hợp lệ: năm báo cáo giai đoạn và một tổng kết; chưa kiểm pagination bằng Word |
| Evaluation tasks | 16/16 bảng DOCX/Markdown khớp |
| Tổng kết | 32/32 bảng DOCX/Markdown khớp sau chuẩn hóa hiển thị liên kết; DOCX có tám hình |

Điểm test chính được xác nhận lại: LR Full 0,8162577160; Tree Full 0,7996486300; POS 0,7105899861; Majority 0,1184304400. Không thay số hoặc predictions trong báo cáo.

## Sai khác toàn vẹn

### Final lock: 69/74 file khớp

`lock.json` và `lock.sha256` vẫn khớp nhau. Năm entry không khớp workspace:

| Entry | Hiện trạng | Bằng chứng |
|---|---|---|
| src/reporting/build_ablation_notebook.py | Đã xóa | Bản trong zip đúng checksum lock |
| src/reporting/build_bert_diagnostics_report.py | Đã xóa | Bản trong zip đúng checksum lock |
| src/reporting/build_contextual_validation_report.py | Đã xóa | Bản trong zip đúng checksum lock |
| src/reporting/update_validation_documentation.py | Đã xóa | Bản trong zip đúng checksum lock |
| reports/final_evaluation/validation_predictions.csv | Đổi định dạng | 37.888 dòng; chuẩn hóa padding/CSV trong bộ nhớ tái tạo đúng hash gốc |

Ngoài lock, `test_predictions.csv` lệch hash trong `test_scoring_complete.json` và artifact_hashes: 2.368 dòng cũng tái tạo đúng hash gốc bằng chuẩn hóa trong bộ nhớ. Nội dung ô và thứ tự dòng được bảo toàn; khác biệt ở biểu diễn file. Loader lịch sử không tự strip padding, nên chỉ bỏ qua hash không phải cách phục hồi đúng.

Không ghi đè hai CSV hoặc phục hồi source đã bị xóa trong lượt kiểm này: đó là thay đổi đã có trước yêu cầu kiểm tra. Không sửa lock/scoring hash để làm kiểm tra pass. Nếu cần verify/replay lịch sử, tạo workspace phục hồi riêng: lấy bốn source đúng byte từ [code_protocol_bundle.zip](../final_evaluation/reproducibility/code_protocol_bundle.zip), dùng biểu diễn CSV có hash đúng, rồi đối chiếu đủ 74 entry và scoring hash. Không cần train hoặc chấm lại test.

### Manifest snapshot và nguồn báo cáo

- `contextual_model_validation/artifact_hashes.json`: 62 file khớp, comparison.json khác hash. Diff có trước lượt kiểm chỉ sửa input hash selection.json; số liệu BERT không đổi và đã tính lại. Không viết lại artifact_hashes lịch sử.
- `final_evaluation/artifact_hashes.json`: 65 file khớp, hai CSV predictions đổi định dạng như trên.
- `evaluation_followup/artifact_hashes.json`: 9 file khớp, source review_saved_evidence.py khác snapshot. Bằng chứng chạy cũ không tự xác nhận phiên bản code hiện tại đã chạy lại.
- `summaries/final/source_manifest.json`: trước sửa điều hướng lượt này, direct_sources có 84/94 entry khớp; source_inventory có 76/86 khớp. Mỗi danh sách thiếu chín đường dẫn và một entry docs/README đã thay đổi. Các mục thiếu gồm bảy generator reporting và hai Markdown diagnostics/contextual ở đường dẫn cũ. Đây là giới hạn truy vết nguồn hiện tại, không phải lỗi số học. README/guide còn có thể đổi sau snapshot; không cập nhật hash ngược vào manifest lịch sử.

Sau cập nhật điều hướng lượt này, hai README final_evaluation/reproducibility cũng khác artifact_hashes cũ: còn 63/67 entry khớp. docs/status được cập nhật trạng thái BERT mới, nên source_manifest tổng kết hiện có 83/94 direct_sources và 75/86 source_inventory khớp; chín đường dẫn thiếu không đổi. Các thay đổi tài liệu này được ghi nhận, không diễn giải thành thay đổi prediction hoặc metric.

## Điều hướng đã sửa

- Mục lục reports/summaries dẫn tới tổng kết Markdown/DOCX đã có; phân biệt báo cáo với PDF/slide và xác nhận nộp bài.
- README contextual/diagnostics phân biệt đợt cũ đã dừng với pipeline đợt mới chưa chạy; dẫn [hướng dẫn BERT-small](../../docs/bert_small_training_guide.md).
- README final_evaluation/reproducibility ghi tình trạng lock hiện tại để người đọc không coi kiểm chứng cũ là kết quả verify của workspace này.
- Hình 8 tổng kết Markdown dùng đường dẫn tương đối, thay đường dẫn ổ C; hình nhúng trong Word giữ nguyên.
- Trước sửa có 15 hyperlink thiếu đích tới sáu source reporting đã xóa trong code_redundancy_review. Giữ nhận xét snapshot, chuyển các hyperlink này thành đường dẫn chữ và thêm ghi chú lịch sử; không khôi phục quyết định xóa source.

Sau sửa, kiểm 28 Markdown hiện hành trong reports: 267 liên kết file nội bộ đều có đích, không còn liên kết thiếu. Đây là kiểm sự tồn tại file, không xác nhận mọi đích đều nằm trong Git hoặc website ngoài truy cập được. Các snapshot trong documentation_archive/before_layout không thuộc phạm vi sửa.

Nội dung định lượng, DOCX, protocol, checkpoint, lock và manifest lịch sử giữ nguyên trong lượt kiểm này. Không kiểm website nguồn trực tuyến hoặc chỉnh literature. Source và các bản archive không được di chuyển/xóa.

Chi tiết máy đọc được và script kiểm local: `.cache/reports_review/result.json`, `.cache/reports_review/check_reports.py`. Lượt này chỉ kiểm bằng chứng đã lưu; không phải tái train 52 fit hoặc một test mới độc lập.
