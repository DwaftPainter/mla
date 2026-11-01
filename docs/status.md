# Trạng thái dự án

Cập nhật ngày 01/11/2026. Đợt cổ điển/chẩn đoán ban đầu thuộc23/10; BERT mở rộng đã hoàn tất train, khóa cấu hình và chấm test ngày28/10. [Tổng kết Markdown](../reports/summaries/final/bao_cao_tong_ket_NUCLE.md), [Word](../reports/summaries/final/bao_cao_tong_ket_NUCLE.docx), [PDF](../reports/summaries/final/bao_cao_tong_ket_NUCLE.pdf) và [slide](../reports/summaries/final/NUCLE_presentation.pptx) đã được chuẩn bị. [Week 11](../reports/summaries/weekly/GROUP05_CLC02_WEEK11_REPORT.md) cập nhật kết quả và audit được nhóm duyệt. Phân công theo tên chỉ xác nhận cho bộ 18 annotation; chưa xác nhận nộp bài hoặc đóng góp toàn project.

## BERT mở rộng và bàn giao ngày 28/10

- Người dùng đã chạy đủ12 run CLS/span_mean × head LR1e−4/1e−3 × ba seed. Chọn span mean/head1e−4 bằng mean validation0,8183 ±0,0011; encoder LR5e−5; primary seed20261013 cố định. Đã khóa trước BERT test và chấm đủ ba seed; không refit train+validation.
- Primary BERT test macro-F1 0,8146, accuracy0,8438, CI95% [0,7919;0,8350]. Mean test ba seed0,8182 ±0,0038. BERT−LR −0,0017, CI [−0,0184;0,0182]; chưa đủ bằng chứng hơn/kém. Test cổ điển đã được xem trước đợt mở rộng; không gọi đây là external test độc lập.
- [Phân tích cuối](../reports/bert_final_analysis/analysis_report.md) lưu61 kiểm tra đã qua, paired bootstrap1.000×97 và bảng/hình. Nhóm đã rà soát và duyệt đủ40 review BERT/LR cùng22 nhận định xử lý ngày01/11; [kết quả audit](../reports/bert_final_analysis/review_summary.md) ghi phạm vi và hướng xử lý. 23 tests BERT/analysis tổng hợp đã qua ở lần kiểm trước; không đồng nhất với suite lịch sử.
- [Bộ bàn giao](../reports/summaries/final/handover_checklist.md) có DOCX/PDF tổng kết, báo cáo BERT riêng, PPTX/PDF slide12 trang và speaker notes khoảng10 phút. Thông tin cá nhân, đóng góp ngoài bộ18 ca audit và xác nhận nộp bài do nhóm hoàn thiện.
- Đã kiểm lại số liệu đã lưu: 280 model–slice metrics, 40 CI và 30 CI chênh lệch khớp; bốn run validation BERT và 18 IDs audit khớp. Đây là tính lại từ predictions, không gọi model scorer lên test.
- Workspace hiện tại chưa qua verify lock cũ: 69/74 hash khớp, thiếu bốn source reporting và một CSV validation đổi định dạng; CSV test cũng lệch scoring hash do đổi định dạng. Chuẩn hóa CSV trong bộ nhớ tái tạo đúng hash gốc; source có bản sao đúng hash trong zip. Xem [báo cáo kiểm tra](../reports/maintenance/reports_check.md). Kết quả tái lập cũ bên dưới là bằng chứng ở thời điểm chạy, không xác nhận workspace hiện tại còn nguyên byte.

## Đã hoàn thành

- Preparation, split và baseline/ablation/masking validation trước đây, giữ nguyên artifact.
- Related work cho tuần 1; không sửa Excel/PRISMA.
- Audit offset/alignment train+validation, review18 lỗi validation đã được nhóm duyệt và đối chiếu code–protocol.
- Sensitivity v1/v2 với LR/POS trên cùng split; xác minh ảnh hưởng số threads đến refit LR.
- Protocol, pilot và bốn lượt BERT-small: hai learning rate ở seed đầu, tổng ba seed cho learning rate chọn. Không phải BERT-base.
- So sánh toàn validation, từng lớp, span, cặp nhãn và POS; mean/std theo seed, confusion matrix, report mới.
- 18 kiểm tra chức năng đã qua; ba checkpoint chọn theo learning rate tái tạo prediction validation không bất đồng.

## Đã hoàn thành sau tuần 1

- Khóa cấu hình trên validation và đánh giá test cho các pipeline cổ điển; giữ mô hình fit train, không refit train+validation. BERT đã chấm test riêng ở đợt mở rộng28/10, không sửa kết quả cổ điển.
- Tính 1.000 paired group-bootstrap, phân tích lỗi và đối chiếu 40 ca audit với nguồn, offset, correction và đặc trưng.
- Tái lập SGML → preparation → grouping/split → NLP/features → 52 fit đợt đánh giá → test replay trong môi trường riêng; predictions test khớp. Kiểm số học độc lập 280 model–slice metrics và 70 CI. Đây là bằng chứng đã lưu, không phải kiểm thử chạy lại ngày cập nhật tài liệu.
- Tổng hợp báo cáo kết quả cuối cùng Markdown/DOCX, bảng/hình và manifest nguồn.

## Kết quả cuối cùng

| Mô hình | Macro-F1 validation | Macro-F1 test | CI95% test |
|---|---:|---:|---|
| Majority | 0,1184 | 0,1184 | [0,1104; 0,1279] |
| POS rule | 0,7048 | 0,7106 | [0,6766; 0,7365] |
| LR Full | 0,8067 | 0,8163 | [0,7876; 0,8382] |
| Tree Full | 0,7879 | 0,7996 | [0,7693; 0,8246] |
| BERT-small primary | 0,8173 | 0,8146 | [0,7919; 0,8350] |

Nguồn: [reviewed_overall_results.csv](../reports/evaluation_followup/reviewed_overall_results.csv) và [báo cáo kiểm chứng](../reports/evaluation_followup/review_report.md). LR Full có accuracy test 0,8357. Hiệu số macro-F1 LR−Tree khoảng 0,0166, CI95% [0,0018; 0,0301]; LR−POS khoảng 0,1057, CI95% [0,0829; 0,1288].

BERT-small mở rộng có macro-F1 validation trung bình0,8183 ±0,0011 và test0,8182 ±0,0038 qua ba seed; primary dùng trong paired comparison là0,8146. Đợt ban đầu đạt validation0,1984 ±0,0030 và giữ như snapshot kết quả âm. Dữ liệu15.788 annotation/11.272 câu, split11.052/2.368/2.368; encoder đặc trưng chỉ fit train.

## Giới hạn và việc còn lại

- Nhóm đã duyệt nhận định xử lý của17 ca trong [hàng đợi cũ](../reports/evaluation_followup/expert_review_queue.csv) và toàn bộ22 ID sau hợp nhất; gold giữ nguyên. Có9 ca được hỗ trợ giữ nhãn,6 ca xem lại nhãn,3 nghi vấn correction/span,3 chỉnh sửa có thể tùy chọn và1 bản sửa nhiều thành phần. Các vấn đề được đánh dấu vẫn được giữ rõ trong nhận định, chưa thay dữ liệu đã khóa.
- Phạm vi là phân loại tại vị trí lỗi đã biết, không đánh giá phát hiện lỗi hoặc sửa câu. Một grouped split và seed cổ điển; CI bootstrap không đo biến thiên do seed huấn luyện.
- Tái lập bao phủ 52 fit của đợt đánh giá, chưa chạy lại toàn bộ grid lịch sử hoặc BERT. Corpus và một số artifact giữ local; điều kiện tái lập ở [reproduction.md](reproduction.md).
- Trước bàn giao: nhóm xác nhận đóng góp ngoài phạm vi18 ca audit và chuẩn bị mẫu có ID, trường hợp lỗi cùng kiểm tra nhỏ cho interview. Không suy đóng góp cá nhân từ kết quả chung hoặc tạo nhật ký hồi tố.
- PDF/PPTX và slide PDF đã tạo; nhóm cần xác nhận thông tin cá nhân/rubric, kiểm trên máy trình bày và tự nộp bài. Lịch interview/hạn nộp theo LMS; không có xác nhận đã nộp.

Các tài liệu tuần 1 là snapshot lịch sử. Dòng “test chưa chấm” ở snapshot cũ và `test_scored=false` trong `selection_summary.json` phản ánh thời điểm trước scoring; kết quả hiện hành nằm trong `test_metrics.json` và artifact kiểm chứng sau đó.

Không diễn giải kết quả BERT-small thành kết luận về mọi mô hình BERT. Audit ngôn ngữ không thay đánh giá độc lập của chuyên gia. Các hạn chế được ghi trong report tuần 1.
