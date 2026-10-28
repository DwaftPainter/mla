# Báo cáo task rà soát bằng chứng và tái lập pipeline

Đợt rà soát ngày 23/10/2026 kiểm tra các kết quả đã lưu có nhất quán và tái lập được trong phạm vi nào. Báo cáo đánh giá ban đầu được giữ làm mốc đối chiếu; tài liệu này bổ sung kết quả kiểm số học, kiểm nguồn và tái lập quy trình. Đây chưa phải báo cáo tổng kết dự án tại thời điểm đó.

## 1. Phạm vi và kết quả kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Understanding từ SGML | 44.912 annotation / 1.397 bài; 0 ô khác CSV cũ |
| Audit dữ liệu và duplicate evidence | Audit CSV giống byte; danh mục duplicate giống nội dung |
| Preparation v2 | Cả 7 data output giống byte, gồm 15.788 sample và provenance |
| Grouping/split | split_manifest.csv và essay_groups.csv giống byte; không có essay/group đi qua nhiều split |
| NLP và đặc trưng | Parse lại 11.272 câu; 15.788 vector / 16.180 chiều; features, targets, train-only encoder giống byte |
| Huấn luyện | 52 fit: 40 grid Full−P/M và 12 refit cấu hình cũ; cấu hình, metric và termination khớp |
| Checkpoint | 18 tên LR/Tree gồm 2 aliases; nội dung 16 checkpoint riêng khớp tuyệt đối |
| Test replay | 20 tên pipeline × 2.368 mẫu, không prediction nào khác; toàn bộ metric/tập con giữ nguyên |
| Kiểm số học độc lập | 280 model–slice metric, 40 CI F1 và 30 CI chênh lệch khớp, tính lại đủ 1.000 bootstrap |
| Report và CSV | 16 bảng DOCX/Markdown giống nhau; 11 bảng định lượng khớp artifact; 240 dòng CSV được đối chiếu |
| Tests môi trường riêng | 13 passed, không lỗi; có warning thư viện không ảnh hưởng kết quả |
| Audit theo ID | 40/40 ca khớp nguồn, offset, correction, tokens và P/M/D; chọn lại bằng seed cho đúng cùng IDs |

Thời gian pipeline tái lập: 13.89 phút. Môi trường riêng `.cache/final_replay_env`, Python 3.13.7, NumPy 2.5.3, PyTorch 2.14.0+cpu, spaCy 3.8.7 và en_core_web_sm 3.8.0. Môi trường không dùng system-site-packages. Thư viện được cài đúng phiên bản đã ghi; môi trường `.venv` gốc không đổi.

Manifest của lần tái lập có thời gian, đường dẫn đầu vào và danh mục mã theo lúc chạy, nên không thể yêu cầu mọi trường đều giống bản cũ. Phần đối chiếu tập trung vào dữ liệu qua SHA256, trọng số hoặc cấu trúc checkpoint và dự đoán theo từng mẫu. Lock cùng kết quả gốc được giữ nguyên. Việc tái tạo dự đoán trên cùng test xác nhận khả năng tái lập thiết kế đã có, không tạo ra một test độc lập mới.

## 2. Phát hiện cần diễn giải chính xác

### 2.1 Hai giá trị baseline nhập gần đúng

`overall_results.csv` cũ lưu validation Majority=0.11843044470868014 và POS=0.7048153. Artifact gốc xác nhận lần lượt 0.11843043995243759 và 0.7048153331595242. Sai lệch tối đa khoảng 3,32×10⁻⁸, không đổi chữ số thứ tư trong report hoặc kết luận. Đã lưu bảng chính xác tại `reviewed_overall_results.csv` và sửa script dựng report để đọc summary baseline, không nhập số cố định. Không chạy lại script xuất DOCX, không ghi đè bảng lịch sử.

### 2.2 Span không rỗng vẫn có annotation thêm từ

16/40 ca trong mẫu audit có correction chỉ thêm từ khi đối chiếu token theo khoảng trắng và bỏ phân biệt hoa/thường, mặc dù span gốc không rỗng. Đây là thống kê của mẫu audit có chủ đích, không phải tỷ lệ toàn corpus. Thao tác thêm từ trong annotation cũng chưa chứng minh từ đó bắt buộc về ngữ pháp.

Hai ID 26635 và 37909 có correction rỗng thật, tương ứng thao tác xóa; không phải parser đọc mất trường correction. Cả 40 ca đều có trường correction trong nguồn. ID 17814 là span một phần token hợp lệ: biểu diễn token bao phủ có thể không phản ánh nhãn của edit ký tự.

### 2.3 Nhận định parser/gold còn cần người đánh giá

Đã xác minh được dữ kiện parser và thao tác edit; chưa thể gọi đó là kết luận chuyên môn độc lập. Lập hàng đợi 17 ca ở `expert_review_queue.csv`, gồm 7 ca nhãn/edit mơ hồ nổi bật, 2 ca phạm vi cấu trúc, 2 ca determiner có thể phụ thuộc cách đọc và 6 ca nghi ngờ parser. Đây là danh sách xem lại, không khẳng định cả 17 annotation sai. Gold và phép chấm giữ nguyên.

Những mô tả như “ngữ cảnh dài hơn” giúp nhận diện khác biệt giữa hai câu, nhưng chưa đủ để giải thích nguyên nhân dự đoán khác nhau. Tương tự, việc LR khớp nhãn gốc chỉ xác nhận điểm chấm theo annotation; nó chưa chứng minh bản sửa là lựa chọn duy nhất hoặc bắt buộc về ngữ pháp.

### 2.4 Dependency không trực tiếp lấy được mọi chủ ngữ

Ở ID 36821 và 17311, target là auxiliary: D|has_subject=0 nhưng parse có subject gắn vào head của target. Extractor chỉ đọc subject child trực tiếp và không truyền qua head/conj. ID 18531 không có subject child trực tiếp hoặc ở head ngay bên cạnh; phạm vi phối hợp cần đọc toàn câu. Đây là giới hạn biểu diễn hiện tại; không sửa extractor sau khi đã xem test.

### 2.5 Giải thích cụ thể một quyết định LR

Phân rã logit(ArtOrDet) − logit(Nn) tại ID 6079: S=−0,0422; L=+1,6728; P=−0,8141; M=+0,3155; D=−0,1921; intercept=+0,1070. Tổng +1,0470 phù hợp quyết định ArtOrDet của LR, trong khi luật POS dự đoán Nn. Cách biểu diễn dùng từ/cụm từ và POS của lân cận, hình thái, dependency cùng vị trí/độ dài span; không có đặc trưng quan sát trực tiếp từ còn thiếu.

Đây là phân rã đại số trên checkpoint đã có, không phải ablation mới, độ quan trọng nhân quả hay bằng chứng model hiểu ngữ pháp. Ba ca LR đúng và một ca sai được lưu đầy đủ dưới dạng đóng góp theo nhóm, không chứa câu nguồn.

## 3. Rà soát phạm vi kết luận RQ1–RQ3

| RQ | Bằng chứng đủ để phát biểu trong phạm vi nghiên cứu | Giới hạn cần giữ |
|---|---|---|
| RQ1 | Test macro-F1: LR Full 0,8163; Tree Full 0,7996; POS 0,7106; Majority 0,1184. LR−Tree +0,0166, CI [0,0018;0,0301]; LR−POS +0,1057, CI [0,0829;0,1288]. | Cùng grouped split và task known-span; không thay mô hình bằng variant thắng test. CI pointwise, một seed cổ điển. |
| RQ2 | Với LR, bỏ toàn L giảm 0,0281; bỏ nhóm P giảm 0,0181, CI đều dương cho chênh Full−variant. M có bằng chứng rõ hơn ở Tree. | CI của M và D ở LR chứa 0; không kết luận vô ích. Bỏ nhóm P/M không loại hết thông tin POS/morph còn có trong D. Full−D là alias; các variant tune riêng. |
| RQ3 | Nhầm lớn: Nn→ArtOrDet 134, chiều ngược 59; Vform→SVA 29, chiều ngược 18. LR tốt hơn POS trên hai cặp nhãn và các tập con Vform/SVA@VERB, Nn/ArtOrDet@NOUN. | Không cải thiện đồng đều trên mọi slice. Quan hệ với token neo, span rộng, parse và gold/edit là bằng chứng mô tả. Audit chủ đích có hỗ trợ AI; chưa xác minh nguyên nhân hay tỷ lệ corpus. |

BERT giữ nguyên kết quả validation và diagnostics cũ; không có BERT test mới. Thử nghiệm BERT chưa thành công không chứng minh ngữ cảnh không hữu ích. LR vốn đã sử dụng một phần thông tin dependency từ parse câu.

## 4. Phạm vi tái lập và phần còn lại

Đã mở rộng kiểm chứng từ replay encoded test sang chuỗi SGML → Understanding/audit → preparation v2 → grouping/split → NLP/features → 52 fit của đợt đánh giá → test prediction replay. Review log annotation lịch sử và cấu hình được chọn của sáu variant cũ vẫn là input cố định. Chưa chạy lại toàn bộ grid lịch sử của sáu variant hoặc thí nghiệm BERT; không tuyên bố tái lập toàn bộ lịch sử nghiên cứu.

Phần còn lại trước báo cáo tổng kết là nhóm/người chuyên môn xem 17 ca trong hàng đợi và xác nhận cách diễn giải. Nếu chưa thực hiện được, ghi rõ giới hạn audit. Final report dự án và slide vẫn để sau; đợt này chỉ bàn giao report task.

## 5. Code và artifact

- `src/evaluation/reproduce_evaluation_pipeline.py`: tái dựng và so sánh trong thư mục riêng; giữ nguyên code đã khóa. Script từ chối ghi đè output, không gọi lại scorer lên test gốc.
- `src/evaluation/review_saved_evidence.py`: tính độc lập confusion, metric và bootstrap; đối chiếu CSV, bảng DOCX/Markdown; lưu bảng baseline chính xác.
- `src/evaluation/review_audit_evidence.py`: kiểm lại nguồn/offset/parser, tái tạo IDs audit, phân rã logit và lập hàng đợi chuyên gia.
- `pipeline_reproduction.json`: môi trường, hash, kết quả từng stage và giới hạn phạm vi.
- `evidence_review.json`: 280 metric checks, 70 CI checks, bảng report và sai lệch nhập số rất nhỏ.
- `audit_followup.json`: kiểm 40 ca, đóng góp nhóm feature, giới hạn giải thích; không chứa câu/correction.
- `reviewed_overall_results.csv`: bảng tổng thể với baseline validation chính xác.
- `expert_review_queue.csv`: 17 ca chưa có kết luận người đánh giá độc lập.
- `tests.xml`: 13 tests pass trong môi trường riêng.

Raw/preparation/features/checkpoints tái dựng giữ local tại `.cache/evaluation_pipeline_reproduction`. Không upload, commit hoặc push trong lượt này.
