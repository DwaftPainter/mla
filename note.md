# Kế hoạch công việc

Đây là nơi quản lý lịch và đầu ra công việc còn lại. Trạng thái: [docs/status.md](docs/status.md). Quy tắc: [docs/methodology/experiment_protocol.md](docs/methodology/experiment_protocol.md). Bản đồ tài liệu: [docs/README.md](docs/README.md).

## Kế hoạch hai tuần còn lại — cập nhật 23/10/2026

Tuần 1 là 23–29/10/2026; tuần 2 là 30/10–05/11/2026. Bảng tuần 1 lưu kế hoạch gốc; công việc đã được thực hiện trong phiên ngày 23/10/2026, với trạng thái thực tế và report bên dưới. Kế hoạch tuần 2 bao phủ RQ1–RQ3: bổ sung ablation cho LR/Tree, khóa mô hình, đánh giá test cùng Majority/POS và hoàn thiện báo cáo. Dừng thử thêm BERT. Các task thí nghiệm tuần 2 đã chạy ngày 23/10/2026; bảng dưới giữ kế hoạch, phần cuối ghi bằng chứng thực hiện. Test đã chấm sau khóa cấu hình. Hiện chỉ bàn giao report task; final report dự án và slide tổng kết để sau.

### Mục tiêu và trạng thái xuất phát

- Đã có preprocessing, split theo essay/group, đặc trưng S/L/P/M/D, baseline, ablation và masking trên validation. Đã có bảng theo độ dài span, cặp nhãn khó và cùng POS; không cần chạy lại các phần này nếu dữ liệu/code không thay đổi.
- Đã rà soát sáu bài từ Excel. Ghi chú cũ tham chiếu `docs/literature_excel_review.md`, nhưng file này hiện không có trong workspace; xem [bảng related work hiện có](docs/literature/contextual_models_related_work.md) và reference.md. Theo cập nhật phạm vi, không còn nhiệm vụ sửa Excel hoặc cập nhật screening/PRISMA trong kế hoạch hai tuần.
- Tại thời điểm lập kế hoạch, BERT chưa chạy. Sau thực hiện tuần 1, BERT-small đã chạy hai learning rate và ba seed ở learning rate chọn; test vẫn chưa chấm. Câu hỏi trọng tâm: khi POS không đủ phân biệt loại lỗi, encoder đọc ngữ cảnh có cải thiện so với LR không, đặc biệt ở Vform/SVA và Nn/ArtOrDet?
- Giữ bài toán câu gốc + gold span → ArtOrDet, Prep, Nn, Vform, SVA. Không dùng câu sửa làm đầu vào; không biến thí nghiệm thành sinh câu sửa hoặc tìm vị trí lỗi.

### Tuần 1 — Hoàn thiện cơ sở nghiên cứu và thử BERT trên validation

| Thời gian | Công việc | Đầu ra và điều kiện hoàn thành |
|---|---|---|
| Ngày 1–2, 23–24/10 | Hoàn thiện related work từ các bài đã đọc: phân biệt dữ liệu thật/giả lập, classification/GEC và metric; chọn nội dung hỗ trợ thử BERT, đánh giá nhóm khó và giải thích đặc trưng. | Bảng literature phục vụ report gồm nguồn, input, output, dữ liệu, metric, điểm áp dụng và giới hạn so sánh; trích dẫn thống nhất với reference.md. |
| Ngày 2–3, 24–25/10 | Audit các lỗi validation tiêu biểu: Vform/SVA, Nn/ArtOrDet, span nhiều token; kiểm tra offset, token alignment, parser và khác biệt code–protocol. Hoàn tất sensitivity v1/v2 trên validation theo protocol, ghi rõ phần dữ liệu khác nhau. | Bảng audit bằng ID và nhận xét; phân biệt lỗi parser, thiếu ngữ cảnh, nhãn mơ hồ và lỗi mô hình. Không đưa văn bản corpus vào Git. |
| Ngày 3, 25/10 | Ghi bổ sung protocol BERT trước lượt chạy chính: câu có marker span, tokenization, cách giữ span khi câu dài, checkpoint, grid nhỏ, seed, ngân sách và quy tắc chọn checkpoint. Kiểm tra tài nguyên bằng lượt chạy ngắn. | Protocol có cấu hình cụ thể và giới hạn runtime theo tài nguyên thực tế; kiểm tra marker/offset, không để truncation làm mất span. Lưu revision mô hình, phiên bản thư viện và cấu hình. |
| Ngày 4–6, 26–28/10 | Fine-tune BERT phân loại năm lớp trên train; chọn checkpoint/cấu hình bằng macro-F1 validation. Dùng cùng split và nhãn với POS/LR. Dự kiến tối đa hai learning rate ở seed đầu; cấu hình chọn chạy tổng ba seed nếu runtime cho phép. | Checkpoint, config, seed, thời gian, log và prediction validation. Nếu giảm số lượt do tài nguyên, ghi rõ lý do; không chọn riêng seed có điểm cao nhất để đại diện BERT. Không mở test. |
| Ngày 7, 29/10 | So POS/LR/BERT trên toàn validation, từng lớp, span một/nhiều token, cặp nhãn khó và các nhóm POS đã định nghĩa. Giữ Majority/Tree trong bảng baseline tổng thể. | Bảng so sánh cùng metric/support và confusion matrix. Với nhiều seed báo trung bình và độ dao động. Ghi cả trường hợp BERT không cải thiện; lập danh sách cấu hình đủ điều kiện khóa. |

**Tài liệu cho tuần 1:** xem [reference.md](reference.md), đặc biệt REF07 và REF11–REF16. Nhận xét độ phù hợp của bài nằm ở [bảng related work hiện có](docs/literature/contextual_models_related_work.md); các phát hiện về Excel chỉ giữ làm ghi chú lịch sử.

**Mốc kết thúc tuần 1:** hoàn tất bảng literature cho report; có kết quả BERT validation nếu chạy thành công, bảng đối chiếu nhóm khó và protocol cập nhật. Nếu BERT bị giới hạn tài nguyên, lưu cấu hình đã thử và trạng thái thực tế; không viết thành kết quả hoàn thành, không để việc tuning kéo dài lấn phần test và báo cáo.


### Kết quả thực hiện tuần 1 — 23/10/2026

Đã thực hiện toàn bộ nhóm việc tuần 1, không thực hiện tuần 2. Kết quả chi tiết: [report DOCX mới](reports/summaries/contextual/contextual_model_validation_report.docx).

| Task | Trạng thái và bằng chứng |
|---|---|
| Related work | Hoàn thành bảng và tổng hợp: docs/literature/contextual_models_related_work.md |
| Audit validation và code–protocol | Kiểm offset/token alignment, review hỗ trợ bằng AI 18 lỗi theo ID; ghi nhận khác biệt vocabulary/masking và hạn chế: docs/methodology/implementation_data_audit.md |
| Sensitivity v1/v2 | Đã chạy LR/POS trên cùng split; chẩn đoán refit LR theo số threads: reports/contextual_model_validation/audit/ |
| Protocol và pilot BERT | Đã chốt trước fine-tuning: docs/bert_experiment_protocol.md |
| BERT | Hoàn thành bốn lượt fine-tune BERT nhỏ trên CPU: hai learning rate ở seed đầu, thêm hai seed cho learning rate chọn; có checkpoint/history/predictions |
| So sánh validation | Có toàn tập, từng lớp, span, cặp nhãn và POS slices: reports/contextual_model_validation/comparison.csv |
| Report tuần 1 | Hoàn thành report mới, có hạn chế, công thức, code và kết quả âm; không ghi đè báo cáo lịch sử |

18 kiểm tra chức năng đã qua; ba checkpoint ở learning rate chọn tái tạo đúng prediction validation. Không đồng nghĩa đã tái lập môi trường sạch hoặc adjudication ngôn ngữ độc lập. Chỉ thử BERT-small, chưa thử BERT-base. Test/CI và khóa mô hình cuối thuộc tuần 2, chưa làm.

### Chẩn đoán BERT bổ sung — đã thực hiện theo yêu cầu

- Đã kiểm input, gradient và khả năng học thuộc100 mẫu train cân bằng; cả ba cấu hình đạt100% accuracy trên chính tập này. CLS5e-5 chưa qua ngưỡng CE chặt trong400 update; CLS1e-3 và span mean1e-3 qua tại100/75 update.
- Không thấy đứt gradient trong probe; marker ảnh hưởng logits, nhưng checkpoint cũ còn yếu khi phân biệt hai gold span cùng câu (chỉ2/50 cặp đúng cả hai).
- Đề xuất thử lại tối ưu hóa và so CLS/span mean sau chẩn đoán đã được chuyển sang hướng phát triển tương lai. Theo quyết định cập nhật ngày 23/10/2026, dừng thử thêm BERT trong phạm vi báo cáo này; không chạy grid mới hoặc BERT-base. Giữ toàn bộ kết quả BERT và chẩn đoán như một thử nghiệm chưa thành công với cấu hình/ngân sách đã dùng. LR vẫn là ứng viên mạnh nhất; không dùng kết quả này để kết luận BERT không phù hợp hoặc ngữ cảnh vô ích.
- Report: [bert_learning_diagnostics_report.docx](reports/summaries/contextual/bert_learning_diagnostics_report.docx). Chỉ dùng train; test và các task tuần2 chưa thực hiện.

### Tuần 2 — Thực nghiệm đã chạy; final report và slide để sau

**Phạm vi cập nhật theo RQ1–RQ3:** thay thế kế hoạch chỉ chấm LR/POS trước đó. LR là ứng viên chính theo validation, nhưng benchmark cuối phải có Majority, LR và Decision Tree để trả lời RQ1; luật POS là đối chứng bổ sung. RQ2 cần chuỗi tăng dần, masking, non-lexical và bổ sung Full−P/M/D cho cả LR và Tree trước khi mở test. RQ3 cần confusion matrix và phân tích lỗi có truy vết. BERT chỉ được tổng hợp từ validation và chẩn đoán đã có, không chạy thêm hoặc đưa vào đợt test chính.

| Câu hỏi | Bằng chứng cần hoàn tất |
|---|---|
| RQ1 — Khả năng phân loại của LR và Tree so với Majority | Benchmark cùng split và biểu diễn Full; metric tổng thể/từng lớp, khoảng tin cậy và chênh lệch LR–Majority, Tree–Majority, LR–Tree. Báo thêm POS và LR–POS. |
| RQ2 — Đóng góp của các nhóm đặc trưng | Chín biến thể cho mỗi mô hình: S+L, S+L+P, S+L+P+M, Full=S+L+P+M+D, Full che lexical span, S+P+M+D, Full−P, Full−M, Full−D. Báo macro-F1 và F1 từng lớp; so các bước tăng dần và Full với các biến thể bỏ nhóm/masking. |
| RQ3 — Nhóm nhầm lẫn và đặc điểm liên quan | Confusion matrix, bảng span/cặp nhãn/POS, audit mẫu sai và mẫu đúng; phân biệt lỗi mô hình, parser và nhãn mơ hồ, nêu giới hạn của review có hỗ trợ AI. |

| Thời gian | Công việc | Đầu ra và điều kiện hoàn thành |
|---|---|---|
| Bước 1 — Ngày 1–2, 30–31/10 | Ghi protocol bổ sung trước khi chạy Full−P, Full−M, Full−D cho cả LR và Tree: giữ split, grid và ngân sách tương ứng như các biến thể trước, encoder chỉ fit train, chọn bằng macro-F1 validation. Dự kiến 60 lượt fit bổ sung theo grid hiện hành. Kiểm tra feature bị loại, convergence, số chiều và lưu checkpoint/predictions; tái sử dụng sáu biến thể đã chạy nếu artifact hợp lệ. | Bảng validation đủ chín biến thể × hai mô hình. Sau đó khóa toàn bộ cấu hình, preprocessing, encoder, checkpoint, seed, threads, phiên bản và hash. Chốt metric, tập con, các phép so sánh bootstrap và quy tắc thiếu lớp trước test. Giữ mô hình fit trên train; không tự refit train+validation. Đồng bộ protocol với phạm vi RQ này. |
| Bước 2 — Ngày 3, 01/11 | Chấm test trong một đợt cho 18 pipeline LR/Tree đã chọn theo validation, Majority và POS. Lưu predictions để tái sử dụng. Báo accuracy, macro-F1, F1 từng lớp và support cho mọi pipeline; confusion matrix và các bảng span rỗng/một từ/nhiều từ, Vform/SVA, Nn/ArtOrDet, POS slices cho benchmark chính. | Bảng RQ1 dùng LR Full và Tree Full trên cùng biểu diễn, cùng Majority/POS; RQ2 dùng toàn bộ chín biến thể của mỗi mô hình. Cấu hình tốt nhất validation nếu báo thêm phải ghi riêng, không thay benchmark cùng biểu diễn. Nhóm không có mẫu ghi N/A. Không dùng test để đổi mô hình, seed hoặc nhóm đánh giá. |
| Bước 3 — Ngày 4, 02/11 | Bootstrap 1.000 lần theo essay/group, dùng cùng resampled groups cho các pipeline. Tính CI 95% macro-F1 và chênh lệch cho các đối chiếu RQ1, các bước tăng dần và Full so với masking/non-lexical/Full−P/M/D của từng mô hình. Báo thêm LR–POS trên các tập con khó đủ dữ liệu. | Bảng CI, seed, support/số nhóm và số lượt thiếu lớp theo quy tắc đã khóa. Chênh lệch F1 từng lớp được báo trong RQ2; không coi các CI của nhiều phép so sánh là kiểm định đồng thời đã hiệu chỉnh. Ghi giới hạn tập con ít mẫu và phân biệt độ bất định do dữ liệu với seed huấn luyện. |
| Bước 4 — Ngày 5, 03/11 | Phân tích tối đa 50 lỗi và 10 mẫu đúng từ dự đoán test đã khóa; bao phủ LR/Tree, hai cặp khó, span nhiều token và bất đồng với POS. Đối chiếu parser và đặc trưng để giải thích RQ3. Tổng hợp riêng kết quả BERT và chẩn đoán đã có. | Bảng theo ID, nhãn/prediction, parser evidence, lý giải và mức chắc chắn; correction chỉ dùng sau dự đoán. Không diễn giải hệ số hay ablation thành quan hệ nhân quả. Nêu audit có hỗ trợ AI, chưa phải adjudication độc lập; BERT chưa thành công không chứng minh ngữ cảnh vô ích. |
| Bước 5 — Ngày 6–7, 04–05/11 | Hoàn thiện report có câu trả lời riêng cho RQ1/RQ2/RQ3, related work, công thức, giải thích code, test/CI, phân tích lỗi và bibliography; chuẩn bị sáu slide. Kiểm tra tái lập môi trường sạch, đối chiếu bảng với predictions; đóng gói code, config, dependency lock, manifest, checksum và hướng dẫn chạy. | Mỗi kết luận RQ dẫn đến bảng/artifact tương ứng, phân biệt validation và test. Tài liệu preprocessing giữ phần dữ liệu, tài liệu mô hình giữ phương pháp/kết quả và dẫn chiếu báo cáo BERT để tránh trùng lặp. Báo cáo lịch sử giữ nhãn thời điểm. Gói tái lập ghi đúng phần đã kiểm chứng; không công khai corpus hoặc tự upload dữ liệu. |

### Quy tắc đánh giá và ưu tiên khi thiếu thời gian

- Cặp Vform/SVA và Nn/ArtOrDet chọn theo gold label nhưng giữ prediction năm lớp; dự đoán ngoài cặp vẫn tính là lỗi. Không ép mô hình thành bộ phân loại nhị phân hoặc loại prediction ngoài cặp để tăng F1.
- Báo riêng tập con theo gold label và tập con theo POS. Không gọi mọi mẫu thuộc cặp nhãn là chắc chắn khó, hoặc tự định nghĩa “dễ/khó” dựa trên lỗi test.
- Span rỗng hiện có support bằng 0 trong dữ liệu đã phân tích: ghi N/A, không tự tạo mẫu để có kết quả. Bảng cuối cần lấy support thực tế của split đang báo cáo.
- Dependency features hiện có khai thác parse của câu; không mô tả LR là chỉ nhìn hai từ mỗi phía. BERT là phép thử cách biểu diễn ngữ cảnh khác, không phải bằng chứng mặc định rằng LR không có thông tin toàn câu.
- Không dành thêm thời gian tuning BERT trong tuần 2. Nếu thiếu thời gian, giảm biểu đồ phụ và số ví dụ phân tích thủ công trước; giữ benchmark RQ1, ablation/masking RQ2, khóa mô hình, test và độ bất định. Full−P/M/D nay là phần cần hoàn thành của kế hoạch; nếu không chạy đủ, phải ghi rõ thu hẹp phạm vi và giới hạn RQ2 trước khi mở test, không tuyên bố hoàn thành đầy đủ đề cương. Các hướng thử BERT tiếp theo chỉ ghi trong future work.
- Không triển khai thêm LET/XGEC, không đổi corpus/taxonomy, không mở rộng feature sau khi xem test. Những ý tưởng phát sinh từ test chỉ đưa vào future work.

**Sản phẩm cuối hai tuần:** bảng validation bổ sung Full−P/M/D; manifest khóa; bảng test Majority/POS và chín biến thể cho mỗi mô hình LR/Tree; CI/chênh lệch điểm và error analysis trả lời RQ1–RQ3; related work và tổng hợp BERT đã có với giới hạn rõ ràng; báo cáo cuối nhất quán, sáu slide và gói tái lập. Các task thực nghiệm đã có đầu ra. Bước viết final report và chốt slide chưa thực hiện theo yêu cầu mới; hiện chỉ bàn giao báo cáo task.

### Kết quả thực hiện đánh giá cuối — 23/10/2026

Báo cáo mới: [evaluation_tasks_report.docx](reports/summaries/evaluation/evaluation_tasks_report.docx), [Markdown](reports/summaries/evaluation/evaluation_tasks_report.md). Artifact: `reports/final_evaluation/`. Các câu “chưa chấm test” trong phần kết quả tuần 1/chẩn đoán phía trên là trạng thái lịch sử.

| Bước | Trạng thái và bằng chứng |
|---|---|
| Ablation và khóa | 40 fit Full−P/M và12 refit cấu hình cũ, validation predictions cũ khớp. Full−D trùng S+L+P+M nên reuse. Protocol: docs/final_evaluation_protocol.md; lock.json lưu hash trước test. |
| Test | 20 tên pipeline:18 LR/Tree gồm aliases, Majority/POS;2.368 mẫu/144 essay/97 groups. LR Full F1=0,8163; Tree Full=0,7996; POS=0,7106; Majority=0,1184. Không chấm BERT. |
| Bootstrap | 1.000 paired group bootstrap, seed20261023;30 contrast/slice đủ điều kiện. LR−POS +0,1057 [0,0829;0,1288]. Hiệu ứng tăng thêm D và M của LR có CI chứa0. |
| Error analysis | 25 lỗi LR,5 lỗi Tree khi LR đúng,10 controls LR đúng. Có40 ghi chú theo ID sau prediction, review hỗ trợ AI; không adjudication độc lập. |
| Báo cáo task và tái lập | DOCX/Markdown mới chỉ báo cáo các task vừa làm, kèm kết quả/công thức/code/audit. Fresh venv replay và restore gói đã kiểm. Final report/slide tổng kết chưa chốt theo yêu cầu mới. |

**Cập nhật tái lập:** đợt rà soát tiếp theo đã chạy từ SGML qua preparation v2/grouping/NLP/features, 52 fit của đợt đánh giá và test replay trong môi trường riêng, kết quả khớp. Chưa rerun toàn grid lịch sử sáu variant/BERT; review log và config lịch sử vẫn là input cố định. Gói trước đó giữ snapshot của mốc bàn giao; bằng chứng mới nằm ở reports/evaluation_followup/.

**Sau test:** không chỉnh feature/config/seed/nhãn theo điểm test. BERT dừng thử thêm; ý tưởng mới chỉ thuộc discussion/future work. Các báo cáo validation trước đây giữ nguyên thời điểm và số liệu.

### Điều chỉnh phạm vi báo cáo theo yêu cầu mới

Chỉ viết [báo cáo task](reports/summaries/evaluation/evaluation_tasks_report.docx) cho ablation bổ sung, khóa/test, bootstrap, audit và kiểm chứng vừa thực hiện. Final report của dự án và slide tổng kết để sau. Bản gọi là final report trước đó được lưu vào documentation_archive, không còn là tài liệu hiện hành. Kết quả test/lock không thay đổi.

### Task rà soát và tái lập tiếp theo — đã thực hiện 23/10/2026

- Rà soát report: 280 model–slice metrics, 40 CI F1, 30 CI chênh lệch và 11 bảng định lượng khớp artifact. Sửa nguồn baseline nhập gần đúng ở script; không thay điểm test.
- Kiểm lại nguồn/offset/parser 40 ca: đều khớp. Có 17 ca trong expert_review_queue.csv chờ người chuyên môn; chưa adjudication độc lập, chưa sửa gold.
- Tái lập raw → preparation v2 → grouping/split → features → 52 fit → test replay trong môi trường riêng; không khác dữ liệu, checkpoint hoặc predictions. 13 tests pass.
- Chốt phạm vi phát biểu và giới hạn RQ1/RQ2/RQ3 trong report task. Final report dự án và slide chưa viết/chưa chốt.
- Báo cáo Markdown: [review_report.md](reports/evaluation_followup/review_report.md). Không xuất DOCX mới. Artifact kiểm chứng: reports/evaluation_followup/.
