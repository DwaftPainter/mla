# BÁO CÁO TỔNG KẾT DỰ ÁN

PHÂN LOẠI NĂM LOẠI LỖI NGỮ PHÁP TIẾNG ANH TRÊN NUCLE

Câu gốc và vị trí lỗi đã biết • Đặc trưng ngôn ngữ • Logistic Regression • Decision Tree • BERT-small

Cập nhật ngày 26/10/2026. Báo cáo sử dụng kết quả thực nghiệm và kiểm chứng đã lưu đến ngày 23/10/2026. Lượt cập nhật này tổng hợp tài liệu, không huấn luyện lại hoặc chấm test mới. Ngôn ngữ: tiếng Việt.

## Tóm tắt

Dự án nghiên cứu phân loại một lỗi đã được định vị trong câu tiếng Anh gốc thành năm nhãn ArtOrDet, Prep, Nn, Vform và SVA. Từ 44.912 annotation của 1.397 bài viết, pipeline giữ đủ 15.788 annotation mục tiêu, tương ứng 11.272 câu đầu vào. Dữ liệu được chia theo nhóm bài viết có quan hệ trùng lặp, với 11.052 mẫu train, 2.368 validation và 2.368 test. Encoder đặc trưng chỉ fit trên train; cấu hình được chọn trên validation và khóa trước khi chấm test.

Trên test, LR Full đạt macro-F1 0,8163 và accuracy 0,8357; Tree Full đạt 0,7996; POS rule đạt 0,7106; Majority đạt 0,1184. Chênh lệch LR−Tree là 0,0166, CI95% [0,0018; 0,0301]; LR−POS là 0,1057, CI95% [0,0829; 0,1288]. Ablation cho thấy POS và lexical mang thông tin hữu ích, trong khi bằng chứng cải thiện do morphology/dependency phụ thuộc mô hình và thiết kế phép so sánh. BERT-small đạt validation macro-F1 trung bình 0,1984 ± 0,0030 trên ba seed và chưa được chấm test. Các phép chẩn đoán cho thấy mô hình có gradient và có thể học thuộc tập nhỏ, nhưng chưa chứng minh khả năng tổng quát hóa tốt.

Phân tích lỗi tập trung vào Nn/ArtOrDet, Vform/SVA, span nhiều token, lỗi thiếu từ được neo vào span không rỗng và giới hạn parser. Kiểm chứng đã tái lập chuỗi từ SGML đến 52 lượt fit của đợt đánh giá, không có khác biệt prediction test. Audit ngôn ngữ có hỗ trợ AI; 17 ca còn chờ chuyên gia. Báo cáo phân biệt kết quả đo được, diễn giải và giả thuyết cho nghiên cứu tiếp theo.

## Hướng dẫn đọc và quy ước

Các chỉ số P/R/F1 và accuracy nằm trên thang 0–1; 0,01 tương đương 1 điểm phần trăm trên thang 0–100. Bảng định lượng được sinh trực tiếp từ CSV/JSON; dấu chấm được giữ trong các bảng số để dễ đối chiếu artifact. CI là khoảng tin cậy bootstrap theo connected group, còn ± của BERT là độ lệch chuẩn mẫu giữa seed. N/A nghĩa là không có dữ liệu hoặc phép ước lượng không đủ điều kiện, không phải 0.

Ưu tiên bằng chứng: reviewed_overall_results.csv và review_report.md → artifact final_evaluation → báo cáo validation/chẩn đoán → proposal và tài liệu tiến độ lịch sử. Các dòng “test chưa chấm” trong tài liệu tuần 1 là trạng thái lịch sử. selection_summary.json có test_scored=false vì được ghi trước chấm test; test_metrics.json và các kiểm chứng sau đó xác nhận test đã hoàn tất. Lượt tạo báo cáo này không huấn luyện lại hoặc thay đổi gold.

Mục lục nội dung: 1. Bài toán; 2. Nghiên cứu liên quan; 3. Dữ liệu; 4. Phương pháp; 5. Thiết kế thực nghiệm; 6. Kết quả test; 7. Ablation; 8. Phân tích lỗi; 9. BERT; 10. Tái lập; 11. Thảo luận và kết luận; Phụ lục và tài liệu tham khảo.

# 1. Bài toán và câu hỏi nghiên cứu

Đơn vị quan sát là một annotation, không phải một câu độc lập. Đầu vào x=(sentence, error_start, error_end, error_span); đầu ra là một nhãn trong tập năm lớp. Hai annotation trong cùng câu có thể có nhãn khác nhau. Đây là bài toán phân loại tại gold span, không đánh giá tự phát hiện vị trí lỗi, sinh câu sửa hay chất lượng phản hồi học tập. Correction chỉ phục vụ kiểm tra dữ liệu và đọc lỗi sau prediction; không đi vào predictor.

Bảng 1. Năm nhãn và nội dung phân loại

| Nhãn | Ý nghĩa |
| --- | --- |
| ArtOrDet | Mạo từ và từ hạn định |
| Prep | Giới từ |
| Nn | Số danh từ |
| Vform | Dạng động từ |
| SVA | Hòa hợp chủ ngữ–động từ |

Nguồn: docs/note_project_guide.md; src/data/constants.py

Bảng 2. Câu hỏi nghiên cứu

| RQ | Câu hỏi | Bằng chứng |
| --- | --- | --- |
| RQ1 | LR và Tree tốt đến đâu so với Majority/POS? | Test, metric từng lớp, paired bootstrap |
| RQ2 | Lexical/POS/morphology/dependency đóng góp thế nào? | Incremental, masking, nonlexical, bỏ P/M/D |
| RQ3 | Những nhầm lẫn nào nổi bật và đặc điểm nào liên quan? | Confusion, span/POS slices, audit theo ID |

Nguồn: docs/note_project_guide.md; docs/final_evaluation_protocol.md

Đóng góp thực tế gồm pipeline giữ offset và provenance, split chống rò rỉ theo nhóm, triển khai LR/Tree và các đối chứng có thể truy vết, thí nghiệm BERT nhỏ có kết quả âm được báo cáo đầy đủ, cùng phân tích định lượng và audit không tự sửa nhãn sau test.

# 2. Cơ sở nghiên cứu và tài liệu liên quan

NUCLE [REF01] cung cấp nguồn annotation; CoNLL-2013 [REF02] và tổng quan GEC [REF03] đặt bài toán trong bối cảnh xử lý lỗi người học. Dự án thu hẹp phạm vi để đo trực tiếp khả năng phân loại loại lỗi từ câu gốc tại vị trí đã biết. Điểm F0.5 của hệ sửa câu hay điểm chất lượng giải thích không thể so trực tiếp với macro-F1 năm lớp ở đây.

Bảng 3. Nghiên cứu liên quan được vận dụng và giới hạn so sánh

| Nguồn | Input → output | Dữ liệu/đánh giá | Vận dụng và giới hạn |
| --- | --- | --- | --- |
| [REF01, NUCLE](https://aclanthology.org/W13-1703/) | Bài người học → annotation lỗi | Corpus learner English | Nguồn dữ liệu và taxonomy; không phải baseline phân loại. |
| [REF04, Davis 2022](https://aclanthology.org/2022.conll-1.25/) | Token trong ngữ cảnh → phát hiện lỗi | Probe kiến thức SVA | Động cơ đánh giá riêng SVA; detection khác five-class span classification. |
| [REF07, LET](https://aclanthology.org/2023.findings-acl.371/) | Câu → loại lỗi theo token và câu sửa | ERRANT, benchmark GEC; F0.5 | Loại lỗi có vai trò trong GEC; không so F0.5 với macro-F1 của nhóm. |
| [REF08, MoECE](https://aclanthology.org/2024.findings-emnlp.997/) | Câu → câu sửa, router dự đoán loại lỗi | GEC với error-type loss và ablation | Tham khảo ablation; không triển khai MoE. |
| [REF09, GEE!](https://aclanthology.org/2024.findings-naacl.49/) | Cặp sai–sửa → giải thích edit | Đánh giá giải thích lỗi | Bối cảnh ứng dụng, không dùng correction vào predictor. |
| [REF10, Coyne 2025](https://arxiv.org/abs/2508.06810) | Annotation upstream đã biết → feedback | Phản hồi học tập | Không coi oracle annotation là năng lực phân loại lỗi. |
| [REF11, Nagata 2022](https://aclanthology.org/2022.findings-acl.324/) | Token ngữ cảnh → đúng/sai hoặc loại lỗi giả lập | Learner corpora: nhị phân; giả lập: nhiều loại | Cơ sở thử BERT. Năm loại giả lập khác năm nhãn NUCLE; thiết kế span marker là điều chỉnh riêng. |
| [REF12, Koyama 2025](https://aclanthology.org/2025.acl-long.1026/) | Cặp câu tối thiểu → đánh giá khả năng sửa từng hiện tượng | CTSEG do giáo viên xây dựng | Cơ sở báo kết quả theo nhóm ngữ pháp; các lát cắt NUCLE không phải CTSEG. |
| [REF13, SERCL](https://aclanthology.org/2020.conll-1.7/) | Parse câu sai và câu sửa → loại edit | Phân loại cấu trúc của edit | Giải thích POS/morph/dependency; không tương đương classifier chỉ dùng câu gốc. |
| [REF14, BERT](https://aclanthology.org/N19-1423/) | Token/subword ngữ cảnh → representation và task head | Pretraining rồi fine-tuning | Cơ sở mô hình; không bảo đảm hơn LR. |
| [REF15, parser robustness](https://aclanthology.org/D16-1182/) | Câu sai/câu sửa → đối chiếu parse | Độ bền parser | Lý do audit parser; không suy tỷ lệ lỗi của spaCy từ parser khác. |

Nguồn: docs/literature/contextual_models_related_work.md; reference.md

LET [REF07] và MoECE [REF08] củng cố động cơ nghiên cứu thông tin loại lỗi và ablation. GEE! [REF09] dùng cặp câu sai–sửa, còn Coyne et al. [REF10] nghiên cứu feedback với annotation upstream đã biết. Các thiết lập đó không phải đối chứng cùng đầu vào với dự án. Nagata et al. [REF11] cùng BERT [REF14] tạo cơ sở thử biểu diễn ngữ cảnh; checkpoint compact gắn với [REF20]. Koyama et al. [REF12] hỗ trợ đánh giá theo hiện tượng. SERCL [REF13] và nghiên cứu độ bền parser [REF15] giúp xác định giới hạn của đặc trưng tự động.

Báo cáo kế thừa tổng hợp literature có trong repository, không tuyên bố systematic review/PRISMA hoàn chỉnh hoặc khoảng trống tuyệt đối của toàn lĩnh vực. Không triển khai LET, MoE, XGEC hay SHAP. Các nguồn chỉ dành cho screening hoặc chưa được dùng trong lập luận không được đưa vào danh mục trích dẫn chính.

# 3. Dữ liệu, chất lượng và tiền xử lý

Bảng 4. Quy mô dữ liệu sử dụng

| Đại lượng | Giá trị |
| --- | --- |
| Annotation toàn nguồn | 44912 |
| Bài có annotation | 1397 |
| Loại lỗi toàn nguồn | 28 |
| Annotation thuộc năm lớp | 15788 |
| Bài có ít nhất một lỗi mục tiêu | 1355 |
| Câu đầu vào preparation v2 | 11272 |
| Annotation ngoài phạm vi | 29124 |
| Mẫu mục tiêu không dựng được | 0 |

Nguồn: reports/summaries/data/nucle_data_understanding.md; preparation v2 summary.json

Bảng 5. Phân bố nhãn toàn tập và từng partition

| Nhãn | Toàn tập | Tỷ lệ % | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| ArtOrDet | 6640 | 42.06 | 4648 | 996 | 996 |
| Prep | 2413 | 15.28 | 1689 | 362 | 362 |
| Nn | 3768 | 23.87 | 2638 | 565 | 565 |
| Vform | 1443 | 9.14 | 1011 | 216 | 216 |
| SVA | 1524 | 9.65 | 1066 | 229 | 229 |

Nguồn: data_set/split/nucle_group_split/split_summary.json

![Hình 1. Phân bố năm nhãn mục tiêu trên 15.788 annotation.](figures/01_label_distribution.png)

Nguồn: preparation v2 summary.json

ArtOrDet chiếm 42,06%, cao khoảng 4,60 lần Vform. Vì mất cân bằng này, accuracy đơn lẻ dễ che khuất thất bại ở lớp nhỏ: chỉ luôn dự đoán ArtOrDet đã đạt accuracy 42,06%, nhưng macro-F1 chỉ 0,1184. Macro-F1 được dùng làm chỉ số chọn mô hình để mỗi lớp có trọng số ngang nhau. Validation và test có cùng số lượng mỗi nhãn do thiết kế chia tập, nhưng là những bài/nhóm riêng biệt.

Bảng 6. Cờ chất lượng dữ liệu, không đồng nghĩa lỗi annotation

| Quality flag | All annotations | Target annotations |
| --- | --- | --- |
| invalid_offset | 0 | 0 |
| cross_paragraph | 4 | 0 |
| missing_label | 0 | 0 |
| missing_correction | 0 | 0 |
| no_op | 30 | 6 |
| empty_edit | 0 | 0 |
| long_span | 387 | 2 |
| same_span_different_label | 47 | 11 |
| duplicate_candidate | 0 | 0 |
| overlapping_span | 531 | 170 |

Nguồn: reports/summaries/data/nucle_data_quality.md

Có 455 cặp span chồng lấp trên toàn nguồn; số cặp khác số annotation được gắn cờ. Các cờ có thể đồng thời xuất hiện nên không cộng để suy số mẫu lỗi. Không có offset sai hoặc correction bị thiếu trong audit; 4 annotation qua paragraph đều nằm ngoài năm lớp. Tập mục tiêu giữ cả 6 no-op, 170 annotation overlap, 11 same-span-different-label và 2 long-span. Các quyết định không tự xóa hoặc sửa nhãn giúp tránh làm đẹp dữ liệu bằng phán đoán sau thực nghiệm.

## 3.1. Bảo toàn offset và thay đổi preparation v1 → v2

Parser đọc paragraph từ SGML; offset dùng chỉ số Python str theo khoảng nửa mở [start,end). Không phải byte offset SGML. Invariant bắt buộc: sentence[error_start:error_end] = error_span. Không bỏ stopword, dấu câu, mạo từ hoặc chuẩn hóa lại source text. Một token giao span khi token_start < error_end và error_start < token_end. Partial-token được giữ nguyên span ký tự nhưng dùng token bao phủ làm biểu diễn.

Bảng 7. Alignment trong preparation v2

| Kiểu | Số mẫu | Tỷ lệ % |
| --- | --- | --- |
| exact | 14447 | 91.51 |
| partial_token | 1130 | 7.16 |
| whitespace_edge | 211 | 1.34 |

Nguồn: data_set/processed/nucle_preparation_v2/summary.json

Preparation v2 bổ sung luật text-only cho citation, abbreviation, ellipsis và dấu câu dính từ, không đọc gold/correction để quyết định biên câu. So v1, 544/15.788 mẫu đổi input (3,45%), 533 đổi sentence ID, 30 đổi context và 2 đổi tokenization span; số câu tăng từ 11.262 lên 11.272. Offset nguồn và metadata nguồn không đổi; không mất mẫu mục tiêu. Hai ca boundary 31297/39538 vẫn unresolved, khác với 96 annotation unresolved trong quality review.

Không có span rỗng trong tập mục tiêu. Thống kê edit cơ học ban đầu ghi 12.821 replacement và 2.967 deletion; replacement dựa trên span/correction không rỗng vẫn có thể là thêm từ bên trong hoặc quanh span. Audit sau này xác minh 16/40 ca có correction chỉ thêm từ theo phép đối chiếu token đã quy định. Do đó không diễn giải “0 empty span” thành “0 lỗi thiếu từ”.

## 3.2. Grouped split và kiểm soát trùng lặp

Bảng 8. Phân chia dữ liệu theo nhóm

| Partition | Samples | Essays | Groups | Tỷ lệ % |
| --- | --- | --- | --- | --- |
| test | 2368 | 144 | 97 | 15.00 |
| train | 11052 | 1069 | 1045 | 70.00 |
| validation | 2368 | 142 | 99 | 15.00 |

Nguồn: data_set/split/nucle_group_split/split_summary.json

Những bài liên kết qua nội dung trùng được gom vào connected components rồi chia nhóm với seed 20261013, mục tiêu 70/15/15. Trên toàn 1.397 bài có 1.280 connected groups; nhóm lớn nhất gồm 34 bài/338 mẫu. Các partition có annotation mục tiêu gồm 1.241 groups và 1.355 bài, nên không cộng số nhóm partition để suy số nhóm toàn corpus. Có 7 nhóm boilerplate được miễn theo policy và không chứa annotation mục tiêu. Không có essay/group vượt qua nhiều split trong kiểm chứng. Điều này giảm rò rỉ do trùng lặp chính xác/chuẩn hóa đã kiểm, chưa bảo đảm loại mọi near-duplicate ngữ nghĩa.

# 4. Đặc trưng và mô hình triển khai

Bảng 9. Năm nhóm đặc trưng

| Nhóm | Thông tin | Số chiều Full |
| --- | --- | --- |
| S | Độ dài câu/span, vị trí tương đối, span kind | 9 |
| L | Lexical slots và unigram/bigram của span/context | 9194 |
| P | POS signature, alignment, POS slots và local pattern | 6068 |
| M | Number, Person, Tense, VerbForm | 31 |
| D | Dependency, head, subject và quan hệ hình thái/POS | 878 |

Nguồn: src/features/nucle_features.py; số chiều suy từ các subset selection_summary.json

Tổng Full là 16.180 chiều. Cửa sổ lexical cục bộ gồm tối đa hai token mỗi phía; parser đọc câu gốc để trích thông tin cú pháp. Numeric giữ thang gốc, không z-score. Giới hạn 500 chỉ áp dụng pooled n-gram vocabulary; các lexical slot có categorical vocabulary riêng. Encoder fit trên train và có xử lý unknown/missing. Nhãn, correction, IDs và review metadata không đi vào predictors. LR vì vậy đã sử dụng một phần thông tin ngữ cảnh toàn câu thông qua dependency, không thể gọi là mô hình hoàn toàn không có ngữ cảnh.

## 4.1. Majority và POS rule

Majority lấy nhãn phổ biến nhất trên train. POS rule có hai ánh xạ cố định ADP→Prep, DET→ArtOrDet; các POS signature khác dùng nhãn đa số học từ train, signature chưa gặp quay về majority toàn train. Đây là đối chứng có học thống kê train, không phải bộ quy tắc ngữ pháp hoàn toàn thủ công. Signature nhiều token và primary_upos cho lát cắt được phân biệt rõ trong code.

## 4.2. Multinomial Logistic Regression

p(y=k|x) = exp(xᵀwₖ+bₖ) / Σⱼ exp(xᵀwⱼ+bⱼ); prediction = argmaxₖ p(y=k|x).

J(W,b) = [Σᵢ α(yᵢ)·(−log p(yᵢ|xᵢ)) / Σᵢ α(yᵢ)] + ||W||²/(2Cn). Với balanced, α(k)=n/(5nₖ); nếu không, α=1. Intercept không bị L2. Code dùng sparse COO PyTorch và LBFGS strong_wolfe, không phải gọi sklearn LogisticRegression.

max_iter=1000, max_eval=1250, tolerance_grad=10⁻⁶, tolerance_change=10⁻⁹. Run cuối dùng 10 CPU threads và seed 20261013. “Converged” theo contract có thể là dừng change_tolerance, không đồng nghĩa gradient đạt ngưỡng. Các cấu hình chạm giới hạn không eligible cho lựa chọn.

## 4.3. Decision Tree

Cây CART tự triển khai dùng Gini(t)=1−Σₖpₖ² và gain = Gini(t)−(wL/wt)Gini(L)−(wR/wt)Gini(R). Khi balanced, impurity và gain dùng sample weight; nhãn lá vẫn là majority của raw counts. Cây xác định với cùng input/config. Các giới hạn depth/min_samples_leaf giúp kiểm soát độ phức tạp. Chi tiết này quan trọng để tái lập và tránh mô tả implementation như một estimator thư viện khác.

# 5. Thiết kế thực nghiệm và phép đo

Bảng 10. Grid và ngân sách lựa chọn

| Hạng mục | Thiết lập |
| --- | --- |
| LR | C={0.01,0.1,1,10} × class_weight={None,balanced}: 8 cấu hình/variant |
| Tree | depth={4,8,12} × min_leaf={5,20} × class_weight: 12 cấu hình/variant |
| Tiêu chí | Macro-F1 validation lớn nhất; LR phải eligible; hòa theo thứ tự grid |
| Đợt đánh giá cuối | 40 grid fit mới Full−P/M + 12 refit cấu hình cũ = 52 fit |
| Alias | Full−D = S+L+P+M; 18 tên LR/Tree nhưng 16 checkpoint riêng |
| Fit cuối | Giữ model fit train, không refit train+validation |
| Test | 20 tên pipeline gồm Majority/POS; 18 predictor riêng khi loại alias |

Nguồn: docs/final_evaluation_protocol.md; selection_summary.json

Mask span lexical xóa danh tính lexical của token giao span trước khi fit masked encoder riêng trên train; không nối n-gram qua vùng che, nhưng giữ parse/POS/morph và context. Nonlexical loại toàn bộ L. Bỏ P/M/D là bỏ nhóm feature theo tên; D vẫn chứa POS/morph của head/subject. Vì mỗi variant được tune riêng, ablation phản ánh hiệu quả pipeline dưới cùng ngân sách lựa chọn, không cô lập nhân quả của một nhóm feature khi giữ mọi hyperparameter bất biến.

Precisionₖ=TPₖ/(TPₖ+FPₖ); Recallₖ=TPₖ/(TPₖ+FNₖ); F1ₖ=2TPₖ/(2TPₖ+FPₖ+FNₖ); Macro-F1=(1/K)ΣₖF1ₖ. Accuracy là tỷ lệ dự đoán đúng. Mẫu số 0 được gán 0; subset rỗng ghi N/A. Toàn tập và span bins dùng K=5. Cặp gold và POS slices dùng hai nhãn gold cố định; prediction vẫn năm lớp, prediction ngoài cặp vẫn tạo false negative.

Lock lúc 2026-10-23 08:39:33 UTC, test bắt đầu 08:39:42 UTC. Hash lock: 2f2f9fd1ff19805255f300d4eaf646aa964e3e643ce692027f528ba5b9c198ee. Cấu hình, encoder, code và input được ghi hash trước scoring. Không chọn lại mô hình theo điểm test hoặc sửa annotation sau khi xem lỗi.

## 5.1. Khoảng tin cậy theo nhóm

Bootstrap B=1.000, seed 20261023, resample có hoàn lại G=97 connected groups của test; giữ toàn bộ annotation trong group. Cùng trọng số group được áp dụng cho mọi model và subset để tính paired differences. Với confusion Cg,m của group g/model m: Cᵇm=Σg wᵇg Cg,m; Δᵇ=F1(CᵇA)−F1(CᵇB). CI95% lấy percentile 2,5% và 97,5%.

CI subset yêu cầu ít nhất 20 samples, 10 groups và đủ hai nhãn gold theo phép so sánh; cần ít nhất 950/1.000 replicate hợp lệ. Thiếu lớp vẫn giữ label set cố định; subset rỗng được loại replicate và đếm. Các CI là pointwise, chưa hiệu chỉnh nhiều so sánh. Bootstrap không đo biến thiên do seed huấn luyện hoặc thay split; CI không phải bảo đảm hiệu quả trên corpus mới.

# 6. Kết quả đánh giá test

Bảng 11. Benchmark chính trên test và validation

| Mô hình | Val F1 | Test F1 | CI95% test | Accuracy | Macro P | Macro R |
| --- | --- | --- | --- | --- | --- | --- |
| Majority | 0.1184 | 0.1184 | [0.1104; 0.1279] | 0.4206 | 0.0841 | 0.2000 |
| POS rule | 0.7048 | 0.7106 | [0.6766; 0.7365] | 0.7441 | 0.7387 | 0.7184 |
| LR Full | 0.8067 | 0.8163 | [0.7876; 0.8382] | 0.8357 | 0.8271 | 0.8073 |
| Tree Full | 0.7879 | 0.7996 | [0.7693; 0.8246] | 0.8095 | 0.8064 | 0.7937 |

Nguồn: reports/evaluation_followup/reviewed_overall_results.csv

![Hình 2. Kết quả test và khoảng tin cậy từ 1.000 paired group-bootstrap.](figures/02_benchmark_ci.png)

Nguồn: reviewed_overall_results.csv

Bảng 12. Chênh lệch paired giữa các mô hình chính

| So sánh | Δ macro-F1 | CI95% |
| --- | --- | --- |
| LR Full - Majority | 0.6978 | [0.6655; 0.7212] |
| Tree Full - Majority | 0.6812 | [0.6470; 0.7073] |
| LR Full - Tree Full | 0.0166 | [0.0018; 0.0301] |
| LR Full - POS | 0.1057 | [0.0829; 0.1288] |

Nguồn: reports/final_evaluation/paired_differences.csv

LR Full dẫn đầu benchmark chính, hơn Tree 1,66 điểm phần trăm và POS 10,57 điểm phần trăm macro-F1; CI paired của cả hai chênh lệch nằm trên 0. Khoảng tin cậy riêng của hai mô hình có thể chồng lấp nhưng CI paired vẫn dương vì phép so sử dụng cùng group resamples. Không suy kết luận từ việc nhìn CI riêng rời rạc. Mức chênh validation→test của LR là +0,0096; đây là khác biệt giữa hai partition, không phải mô hình được cải thiện sau test.

Bảng 13. Precision, recall và F1 từng lớp của LR Full trên test

| Nhãn | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| ArtOrDet | 996 | 0.8366 | 0.9046 | 0.8693 |
| Prep | 362 | 0.9239 | 0.9061 | 0.9149 |
| Nn | 565 | 0.8184 | 0.7416 | 0.7781 |
| Vform | 216 | 0.7387 | 0.6806 | 0.7084 |
| SVA | 229 | 0.8178 | 0.8035 | 0.8106 |

Nguồn: reports/final_evaluation/test_metrics.json

Bảng 14. F1 từng lớp của bốn đối chứng

| Nhãn | Majority | POS rule | LR Full | Tree Full |
| --- | --- | --- | --- | --- |
| ArtOrDet | 0.5922 | 0.7731 | 0.8693 | 0.8329 |
| Prep | 0.0000 | 0.9212 | 0.9149 | 0.9065 |
| Nn | 0.0000 | 0.7170 | 0.7781 | 0.7519 |
| Vform | 0.0000 | 0.5946 | 0.7084 | 0.6979 |
| SVA | 0.0000 | 0.5470 | 0.8106 | 0.8090 |

Nguồn: reports/final_evaluation/test_metrics.json

![Hình 3. So sánh F1 từng nhãn trên test.](figures/03_per_class_f1.png)

Nguồn: test_metrics.json

Ưu thế tổng thể của LR không đồng nghĩa hơn POS trên mọi nhãn: Prep có F1 POS 0,9212 và LR 0,9149. POS đã là đối chứng mạnh cho giới từ, trong khi LR tăng khả năng phân biệt các nhóm có cùng từ loại. Với Nn, LR còn recall 0,7416; khoảng 25,84% mẫu Nn bị gán sang lớp khác. Các lớp nhỏ và ranh giới nhãn cần được đọc riêng thay vì chỉ dùng một điểm trung bình.

# 7. Đóng góp của từng nhóm đặc trưng

Bảng 15. Toàn bộ kết quả ablation validation/test

| Variant | LR val | LR test | Tree val | Tree test |
| --- | --- | --- | --- | --- |
| S+L | 0.7164 | 0.7203 | 0.5832 | 0.5850 |
| S+L+P | 0.7956 | 0.7968 | 0.7548 | 0.7520 |
| S+L+P+M | 0.8048 | 0.8102 | 0.7929 | 0.7967 |
| Full | 0.8067 | 0.8163 | 0.7879 | 0.7996 |
| Mask lexical span | 0.8016 | 0.8028 | 0.7876 | 0.8006 |
| Không L | 0.7964 | 0.7881 | 0.7862 | 0.7894 |
| Full−P | 0.8001 | 0.7982 | 0.7711 | 0.7811 |
| Full−M | 0.8015 | 0.8123 | 0.7751 | 0.7826 |
| Full−D (alias) | 0.8048 | 0.8102 | 0.7929 | 0.7967 |

Nguồn: reviewed_overall_results.csv; Full−D trùng S+L+P+M

![Hình 4. Kết quả khi bổ sung lần lượt POS, morphology và dependency.](figures/04_incremental_features.png)

Nguồn: reviewed_overall_results.csv

Bảng 16. CI chênh lệch cho incremental và ablation

| So sánh | Δ test F1 | CI95% |
| --- | --- | --- |
| LR S+L+P minus S+L | 0.0766 | [0.0581; 0.0975] |
| LR S+L+P+M minus S+L+P | 0.0134 | [-0.0004; 0.0256] |
| LR Full minus S+L+P+M | 0.0060 | [-0.0054; 0.0191] |
| LR Full minus Full_mask_span_lexical | 0.0134 | [0.0049; 0.0201] |
| LR Full minus S+P+M+D_nonlexical | 0.0281 | [0.0142; 0.0394] |
| LR Full minus Full_minus_P | 0.0181 | [0.0069; 0.0296] |
| LR Full minus Full_minus_M | 0.0040 | [-0.0066; 0.0136] |
| LR Full minus Full_minus_D | 0.0060 | [-0.0054; 0.0191] |
| Tree S+L+P minus S+L | 0.1670 | [0.1400; 0.1897] |
| Tree S+L+P+M minus S+L+P | 0.0447 | [0.0275; 0.0648] |
| Tree Full minus S+L+P+M | 0.0029 | [-0.0077; 0.0124] |
| Tree Full minus Full_mask_span_lexical | -0.0010 | [-0.0117; 0.0086] |
| Tree Full minus S+P+M+D_nonlexical | 0.0103 | [-0.0014; 0.0205] |
| Tree Full minus Full_minus_P | 0.0186 | [0.0049; 0.0328] |
| Tree Full minus Full_minus_M | 0.0171 | [0.0008; 0.0360] |
| Tree Full minus Full_minus_D | 0.0029 | [-0.0077; 0.0124] |

Nguồn: reports/final_evaluation/paired_differences.csv

POS là bước tăng lớn nhất trong chuỗi bổ sung: LR tăng 0,0766 [0,0581; 0,0975], Tree tăng 0,1670 [0,1400; 0,1897] khi chuyển từ S+L sang S+L+P. Điều này phù hợp với taxonomy có liên hệ mạnh với loại từ, nhưng không chứng minh chỉ POS là đủ: LR Full vẫn hơn POS rule 0,1057.

Lexical hữu ích đối với LR: bỏ toàn L làm giảm 0,0281 [0,0142; 0,0394]; chỉ che lexical span làm giảm 0,0134 [0,0049; 0,0201]. Hai phép so khác nhau vì nonlexical còn loại thông tin lexical lân cận. Đây là bằng chứng lợi ích của biểu diễn lexical trong pipeline, không phải phép đo tỷ lệ “học thuộc từ” thuần túy. Tree masked đạt 0,8006, nhỉnh hơn Full 0,7996 nhưng CI Full−masked chứa 0; không thay Tree Full bằng variant thắng test.

Bỏ P làm giảm LR 0,0181 và Tree 0,0186, với CI dương. Với morphology, Tree Full−Full−M là 0,0171 [0,0008; 0,0360]; LR chỉ 0,0040 [−0,0066; 0,0136]. Với dependency, CI của cả LR và Tree chứa 0. Kết luận phù hợp là chưa có bằng chứng đủ mạnh về cải thiện ổn định ở những phép so đó, không phải morphology/dependency vô ích. Thông tin giữa nhóm có thể trùng nhau và kích thước test hữu hạn.

# 8. Phân tích lỗi và các nhóm khó

![Hình 5. Ma trận nhầm lẫn LR Full trên test: hàng gold, cột dự đoán.](figures/05_lr_confusion.png)

Nguồn: test_metrics.json

Bảng 17. Các hướng nhầm lẫn lớn nhất của LR Full

| Gold → prediction | Số ca | % lỗi LR | % support gold |
| --- | --- | --- | --- |
| Nn → ArtOrDet | 134 | 34.45 | 23.72 |
| ArtOrDet → Nn | 59 | 15.17 | 5.92 |
| Vform → SVA | 29 | 7.46 | 13.43 |
| Vform → Nn | 20 | 5.14 | 9.26 |
| ArtOrDet → Prep | 19 | 4.88 | 1.91 |
| Vform → ArtOrDet | 18 | 4.63 | 8.33 |
| SVA → Vform | 18 | 4.63 | 7.86 |
| Prep → ArtOrDet | 15 | 3.86 | 4.14 |

Nguồn: Tính từ confusion trong test_metrics.json

LR có 389 lỗi trên 2.368 mẫu. Hai chiều Nn↔ArtOrDet chiếm 193/389 lỗi (49.61%); Vform↔SVA chiếm 47/389 (12.08%). Tổng hai cặp chiếm 61.70% lỗi LR. Đây là tỷ trọng trong prediction test, không phải tỷ lệ nguyên nhân ngôn ngữ hay tỷ lệ annotation sai.

Bảng 18. Kết quả theo span và cặp nhãn/POS trên test

| Subset | n | Groups | Majority F1 | POS F1 | LR F1 | Tree F1 |
| --- | --- | --- | --- | --- | --- | --- |
| empty | 0 | 0 | N/A | N/A | N/A | N/A |
| single_token | 2102 | 97 | 0.1154 | 0.7173 | 0.8320 | 0.8145 |
| multi_token | 266 | 57 | 0.1405 | 0.4921 | 0.5370 | 0.4489 |
| Vform/SVA | 445 | 76 | 0.0000 | 0.6095 | 0.8021 | 0.7959 |
| Vform/SVA@VERB | 220 | 64 | 0.0000 | 0.3548 | 0.8416 | 0.8412 |
| Vform/SVA@AUX | 105 | 44 | 0.0000 | 0.4670 | 0.8477 | 0.8677 |
| Vform/SVA@NOUN | 37 | 21 | 0.0000 | 0.0000 | 0.1667 | 0.1071 |
| Vform/SVA@DET | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Nn/ArtOrDet | 1561 | 95 | 0.3895 | 0.7663 | 0.8453 | 0.8148 |
| Nn/ArtOrDet@VERB | 33 | 22 | 0.4500 | 0.0000 | 0.3864 | 0.4843 |
| Nn/ArtOrDet@AUX | 3 | 3 | 0.2500 | 0.0000 | 0.0000 | 0.0000 |
| Nn/ArtOrDet@NOUN | 682 | 86 | 0.2431 | 0.4044 | 0.8142 | 0.7028 |
| Nn/ArtOrDet@DET | 343 | 64 | 0.4827 | 0.4827 | 0.4827 | 0.4827 |

Nguồn: reports/final_evaluation/test_metrics.json

![Hình 6. Kết quả các lát cắt; span dùng 5 nhãn, cặp dùng 2 nhãn.](figures/06_slices.png)

Nguồn: test_metrics.json; không cộng hoặc lấy trung bình các slice để suy F1 toàn tập

Bảng 19. CI LR−POS theo từng lát cắt đủ điều kiện

| Subset | Δ F1 | CI95% |
| --- | --- | --- |
| single_token | 0.1147 | [0.0886; 0.1406] |
| multi_token | 0.0449 | [-0.0401; 0.1262] |
| Vform/SVA | 0.1927 | [0.1407; 0.2394] |
| Vform/SVA@VERB | 0.4867 | [0.4214; 0.5405] |
| Vform/SVA@AUX | 0.3807 | [0.1145; 0.4900] |
| Vform/SVA@NOUN | 0.1667 | [0.0400; 0.2895] |
| Nn/ArtOrDet | 0.0790 | [0.0463; 0.1195] |
| Nn/ArtOrDet@VERB | 0.3864 | [0.2500; 0.4706] |
| Nn/ArtOrDet@NOUN | 0.4098 | [0.3555; 0.4685] |
| Nn/ArtOrDet@DET | 0.0000 | [0.0000; 0.0000] |

Nguồn: reports/final_evaluation/paired_differences.csv

Trên multi-token, LR−POS chỉ +0,0449 và CI [−0,0401; 0,1262] chứa 0; chưa thể khẳng định cải thiện ổn định trong nhóm này. Trên Vform/SVA@VERB, mức tăng +0,4867 [0,4214; 0,5405] cho thấy chỉ biết token là động từ chưa đủ phân biệt dạng động từ và hòa hợp. Nn/ArtOrDet@NOUN tăng +0,4098 [0,3555; 0,4685]. Ngược lại, Nn/ArtOrDet@DET có Δ=0: cải thiện không đồng đều giữa mọi lát cắt. Các slice ít support phải đọc cùng n/groups và không suy diễn mạnh từ điểm 0 hoặc 1.

## 8.1. Audit ngôn ngữ và giới hạn diễn giải

Audit test gồm 25 lỗi LR, 5 lỗi Tree khi LR đúng và 10 ca LR đúng; IDs được chọn có chủ đích theo seed trước khi đọc correction. Cả 40/40 ca được kiểm lại nguồn, offset, correction, tokens và đặc trưng. Đây là review có hỗ trợ AI, không phải chuyên gia độc lập và không phải mẫu ngẫu nhiên đại diện corpus.

Bảng 20. Các kiểu khó quan sát được trong audit

| Nhóm | ID minh họa | Nhận định có giới hạn |
| --- | --- | --- |
| Thiếu từ, span không rỗng | 38087, 24747, 17826 | Token neo có POS khác loại từ bị thiếu; dễ dự đoán theo token đang thấy. |
| Một phần token | 17814 | Offset hợp lệ nhưng token bao phủ được gán ADP, trong khi edit ký tự liên quan số danh từ. |
| Span rộng/nhiều thay đổi | 19312, 34483, 32904 | Một nhãn bao phủ nhiều tín hiệu; không thể đồng nhất span với edit tối thiểu. |
| Parser đáng xem lại | 24526, 9351, 41235, 35455, 211 | POS/morph tự động có thể không phù hợp ngữ cảnh; chưa xác minh nhân quả. |
| Chủ ngữ gián tiếp | 36821, 17311 | Subject gắn vào head; extractor chỉ đọc subject child trực tiếp của target. |
| Gold/edit mơ hồ | 34539, 12324, 26635, 32705, 37909 | Cần adjudication; giữ nguyên gold và phép chấm. |
| LR đúng, POS sai | 11916, 44353, 6079 | LR có thể dùng tín hiệu ngoài POS token neo; đúng gold chưa chứng minh hiểu ngữ pháp. |

Nguồn: reports/evaluation_followup/review_report.md; audit_followup.json; evaluation_tasks_report.md

Hàng đợi chuyên gia gồm 17 ca: 7 ca nhãn/edit mơ hồ nổi bật, 2 ca phạm vi cấu trúc, 2 ca determiner phụ thuộc cách đọc và 6 ca nghi parser. Không khẳng định 17 annotation sai. Hai correction rỗng ở 26635 và 37909 là thao tác xóa thật, không phải trường bị parser làm mất. Không loại các ca khó rồi báo lại điểm test cao hơn.

Bảng 21. Phân rã logit ArtOrDet−Nn của LR tại ID 6079

| Thành phần | Đóng góp |
| --- | --- |
| S | −0.0422 |
| L | +1.6728 |
| P | −0.8141 |
| M | +0.3155 |
| D | −0.1921 |
| Intercept | +0.1070 |
| Tổng | +1.0470 |

Nguồn: reports/evaluation_followup/review_report.md; số làm tròn

Tổng logit dương phù hợp với quyết định ArtOrDet của LR trong khi POS dự đoán Nn. Lexical có đóng góp dương lớn hơn đóng góp âm của P trong phép phân rã này. Đây là giải thích đại số của checkpoint cho một ca, không phải độ quan trọng nhân quả, không phải chứng minh một đặc trưng riêng khiến mô hình hiểu từ đang thiếu.

# 9. Thử nghiệm BERT-small và chẩn đoán

Checkpoint google/bert_uncased_L-2_H-128_A-2, revision 30b0a37ccaaa32f332884b96992754e246e48c5f: 2 layers, hidden 128, 2 attention heads, khoảng 4,39 triệu tham số. Không phải BERT-base. Input đặt [SPAN_START]/[SPAN_END] quanh gold span, tokenize trái/span/phải riêng, max_length=128 và ưu tiên giữ toàn span. Có 5 train và 6 validation bị cắt context nhưng không span nào bị cắt. Encoder và head cùng fine-tune.

Pooled CLS đi qua dropout/classifier; loss cross-entropy. Batch32, 5 epochs, AdamW, weight_decay=0,01 trừ bias/LayerNorm, warmup10% và linear decay, clip_norm1, CPU4 threads. Hai learning rate 3e−5/5e−5 chạy ở seed đầu; LR được chọn chạy thêm hai seed. Chọn checkpoint bằng validation macro-F1, không chọn seed tốt nhất để đại diện. Không có BERT test.

Bảng 22. Các lượt fine-tune BERT trên validation

| Learning rate | Seed | Epoch được chọn | Macro-F1 | Runtime giây |
| --- | --- | --- | --- | --- |
| 3e-05 | 20261013 | 5 | 0.1706 | 213.4 |
| 5e-05 | 20261013 | 5 | 0.2003 | 205.5 |
| 5e-05 | 20261014 | 5 | 0.1999 | 209.6 |
| 5e-05 | 20261015 | 4 | 0.1949 | 196.9 |

Nguồn: docs/contextual_model_validation_report.md; bert run artifacts

Bảng 23. So sánh validation gồm BERT và mô hình cổ điển

| Model | n | Macro-F1 | Accuracy |
| --- | --- | --- | --- |
| Majority | 2368 | 0.1184 | 0.4206 |
| LR Full (saved) | 2368 | 0.8067 | 0.8463 |
| Tree Full | 2368 | 0.7879 | 0.8142 |
| POS rule | 2368 | 0.7048 | 0.7576 |
| Tree S+L+P+M | 2368 | 0.7929 | 0.8264 |
| BERT-small seed 20261013 | 2368 | 0.2003 | 0.4535 |
| BERT-small seed 20261014 | 2368 | 0.1999 | 0.4341 |
| BERT-small seed 20261015 | 2368 | 0.1949 | 0.4282 |

Nguồn: reports/contextual_model_validation/comparison.csv

![Hình 7. BERT-small và đối chứng trên cùng validation; không phải điểm test.](figures/07_bert_validation.png)

Nguồn: contextual_model_validation/comparison.csv

BERT ở learning rate được chọn đạt 0,1984 ± 0,0030 (độ lệch chuẩn mẫu, n=3 seed). Seed chính đạt 0,2003, thấp hơn LR validation 0,8067 là 0,6064. Prediction tập trung 1.828 ArtOrDet, 538 Nn và 2 Prep; không dự đoán Vform/SVA. F1 Prep/Vform/SVA đều 0 ở seed chính. Accuracy 0,4535 chỉ cao hơn majority một ít và không thể hiện năng lực đủ năm lớp. Mean/std theo seed không phải CI và không gộp với bootstrap test.

## 9.1. Chẩn đoán tối ưu hóa và sử dụng span

Bảng 24. Bằng chứng chẩn đoán BERT: Nguồn / readout

| Nguồn / readout | Embedding norm | Encoder layer0 norm | Layer1 norm | Classifier norm | Δ marker đầu sau1step |
| --- | --- | --- | --- | --- | --- |
| pretrained_cls | 0.654593 | 0.307446 | 0.235027 | 0.512786 | 0.000566 |
| pretrained_span_mean | 0.865756 | 0.575860 | 0.352035 | 0.841210 | 0.000566 |
| saved_failed_run_cls | 6.074696 | 2.479423 | 1.680808 | 2.182717 | 0.000566 |
| saved_failed_run_span_mean | 3.937057 | 1.633337 | 1.107149 | 1.935566 | 0.000566 |

Nguồn: docs/bert_learning_diagnostics_report.md; reports/bert_learning_diagnostics

Bảng 25. Bằng chứng chẩn đoán BERT: Readout dùng với encoder cũ

| Readout dùng với encoder cũ | Số prediction thay đổi | Mean L1 xác suất | Cosine representation trung bình |
| --- | --- | --- | --- |
| cls | 28/100 | 0.3481 | 0.7959 |
| span_mean | 11/100 | 0.3497 | 0.5967 |

Nguồn: docs/bert_learning_diagnostics_report.md; reports/bert_learning_diagnostics

Bảng 26. Bằng chứng chẩn đoán BERT: Readout

| Readout | Learning rate | Lần đầu quan sát acc100% | Update cuối | Accuracy cuối | CE cuối | Tiêu chí chặt |
| --- | --- | --- | --- | --- | --- | --- |
| cls | 5e-05 | 175 | 400 | 100.0% | 0.1515 | Chưa đạt CE≤0,05 |
| cls | 0.001 | 50 | 100 | 100.0% | 0.0073 | Đạt |
| span_mean | 0.001 | 50 | 75 | 100.0% | 0.0104 | Đạt |

Nguồn: docs/bert_learning_diagnostics_report.md; reports/bert_learning_diagnostics

![Hình 8. Đường học thuộc trên 100 mẫu train được chọn; không đo generalization.](C:/Hanu/year4/semester1/MLA/Final_project/reports/bert_learning_diagnostics/memorization_curves.png)

Nguồn: reports/bert_learning_diagnostics/memorization_curves.png

Gradient probe kiểm pretrained model và checkpoint cũ: loss khớp cross-entropy độc lập, optimizer đủ tham số trainable, không gradient thiếu/NaN/Inf, marker/encoder/classifier đều cập nhật. Với CLS, gradient trực tiếp ở hidden target layer cuối bằng 0 là cấu trúc readout dự kiến; gradient qua self-attention ở layer trước vẫn tồn tại. Vì vậy không coi đó là bằng chứng đứt đường học.

Dời marker trên 100 mẫu train làm CLS đổi prediction 28/100, cho thấy mô hình có phản ứng với marker nhưng chưa chứng minh định vị đúng lỗi. Trên 50 cặp target khác nhãn trong cùng câu train, CLS chỉ đổi prediction ở 7 cặp và đúng cả hai ở 2 cặp. Đây là stress test có chủ đích trên train, không phải đánh giá validation mới.

Học thuộc 100 mẫu: CLS 5e−5 đạt accuracy100% nhưng CE cuối0,1515 nên chưa qua ngưỡng confidence CE≤0,05. CLS1e−3 và span_mean1e−3 đạt tiêu chí chặt sau 100 và 75 updates. Phép thử nhỏ thay dropout, weight decay, schedule và số lần thấy lại mỗi ví dụ so với fine-tuning toàn tập. Không quy thất bại cũ chỉ cho learning rate; cũng không suy span_mean đã tốt hơn LR. Kết luận hiện tại là mô hình có thể tối ưu trên tập nhỏ, nhưng recipe đã thử chưa tổng quát hóa tốt.

# 10. Kiểm chứng, tái lập và độ tin cậy

Bảng 27. Kết quả kiểm chứng đã lưu

| Hạng mục | Kết quả |
| --- | --- |
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

Nguồn: reports/evaluation_followup/review_report.md; pipeline_reproduction.json; evidence_review.json

Pipeline replay mất 13,89 phút trong môi trường riêng, Python3.13.7, NumPy2.5.3, PyTorch2.14.0+cpu, spaCy3.8.7 và en_core_web_sm3.8.0, không dùng system-site-packages. Preparation có 7 data outputs giống byte; parse 11.272 câu, tạo 15.788 vectors/16.180 chiều; 52 fit tái tạo đúng cấu hình/metric/termination, 16 checkpoint riêng khớp nội dung và 20 tên pipeline×2.368 prediction test không khác. Đây là kết quả replay đã ghi trong artifact, không phải chạy lại trong lượt viết báo cáo.

Kiểm số học độc lập đã đối chiếu 280 model–slice metrics, 40 CI F1, 30 CI chênh lệch và 1.000 bootstrap. Hai validation baseline trước đây nhập gần đúng, sai lệch tối đa khoảng 3,32×10⁻⁸; báo cáo này đọc reviewed_overall_results.csv để dùng số chính xác. Không đổi kết luận làm tròn bốn chữ số.

Sensitivity v1/v2 giữ cấu hình LR: macro-F1 0,8069→0,8078 trong run 4 threads; baseline gốc 10 threads là 0,8067. Hai prediction khác giữa refit và baseline được giải thích bằng độ nhạy số học LBFGS/threads trong kiểm tra đã làm. Không lấy điểm refit cao hơn thay baseline. Cải thiện preprocessing nhỏ chưa có CI và không chứng minh tăng chất lượng mô hình.

Phạm vi tái lập đã bao phủ raw→features→52 fit đợt đánh giá→test replay. Review logs và cấu hình sáu variant lịch sử vẫn là input cố định; chưa chạy lại toàn bộ grid lịch sử hoặc BERT. Tái lập prediction trên cùng test không phải thêm bằng chứng từ một test độc lập. Corpus, encoder/vocabulary và checkpoints giữ local; người tái lập toàn bộ cần nguồn NUCLE hợp lệ và các inputs đã ghi trong docs/reproduction.md.

# 11. Thảo luận, trả lời RQ và kết luận

Bảng 28. Trả lời ba câu hỏi nghiên cứu

| RQ | Kết luận trong phạm vi bằng chứng | Giới hạn |
| --- | --- | --- |
| RQ1 | LR Full là mô hình chính mạnh nhất: test F1 0,8163; hơn Tree/POS với paired CI dương. | Một grouped split; một seed cổ điển; known-span, không phải GEC end-to-end. |
| RQ2 | POS tăng mạnh từ S+L; lexical hữu ích với LR; morphology có bằng chứng rõ hơn ở Tree. | Từng variant tune riêng; nhóm có thông tin trùng; CI M/D ở LR chứa 0. |
| RQ3 | Nhầm Nn/ArtOrDet và Vform/SVA chiếm phần lớn lỗi; LR vượt POS ở các slice cùng POS quan trọng. | Audit chủ đích có hỗ trợ AI; quan hệ mô tả không chứng minh nguyên nhân. |

Nguồn: Các bảng test/ablation/slice và review_report.md

Kết quả hỗ trợ lựa chọn LR Full làm mô hình tham chiếu của dự án: hiệu quả tốt trên tập test đã khóa, biểu diễn thưa và có thể phân rã logit theo nhóm feature. Không có benchmark latency/bộ nhớ chuẩn hóa để kết luận triển khai nhanh hơn BERT bao nhiêu; thời gian các run có ngân sách và môi trường khác nhau nên chỉ được dùng mô tả thực thi.

Bằng chứng mạnh nhất về giá trị đặc trưng là lợi ích của POS và lexical trong các phép so đã định. Bằng chứng yếu hơn nằm ở lợi ích bổ sung của dependency hoặc morphology cho LR. Một nhóm có CI chứa 0 có thể cung cấp thông tin trùng với nhóm khác, bị hạn chế bởi extractor hoặc cần dữ liệu nhiều hơn; thí nghiệm hiện tại không phân biệt được các khả năng đó.

Kết quả BERT âm là một kết quả thực nghiệm cần giữ, không phải lý do bỏ ngữ cảnh. LR đã sử dụng dependency câu; BERT nhỏ, pooled CLS, marker mới và ngân sách ngắn là một cấu hình cụ thể. Chẩn đoán giúp loại trừ một số lỗi cơ học đã kiểm, đồng thời chỉ ra câu hỏi mới về readout và tối ưu hóa, chưa xác nhận giải pháp.

## 11.1. Hạn chế và hướng tiếp theo

Bảng 29. Hạn chế và hướng kiểm tra tiếp theo

| Giới hạn | Tác động | Hướng thực hiện sau báo cáo |
| --- | --- | --- |
| Gold span được cho trước | Không đo khả năng tìm lỗi hoặc sửa câu | Đánh giá detection→classification trên dữ liệu mới nếu mở rộng mục tiêu. |
| Một split/seed cổ điển | CI không bao phủ biến thiên training/split | Protocol nhiều seed/split hoặc external test mới. |
| Parser và taxonomy | Một số lỗi có thể bắt nguồn từ biểu diễn hoặc nhãn | Chuyên gia xem 17 ca; lưu phiên bản gold mới nếu có adjudication. |
| Subject chỉ là child trực tiếp | Bỏ quan hệ qua auxiliary/head/conj | Extractor kế thừa subject có kiểm soát, đánh giá trên dữ liệu mới. |
| BERT-small chưa thành công | Chưa kiểm lợi ích contextual mạnh hơn | So CLS/span_mean cùng ngân sách, theo dõi learning curve; chưa dùng test cũ để tune. |
| Nhiều CI pointwise | Không bảo đảm kết luận đồng thời | Đăng ký ít contrast chính hoặc hiệu chỉnh đa so sánh ở nghiên cứu tiếp. |
| Đầu vào nguồn riêng | Checkout không đủ tái lập toàn nghiên cứu | Giữ hướng dẫn, hashes, môi trường và điều kiện truy cập nguồn rõ ràng. |

Nguồn: Tổng hợp từ protocol, code và các kết quả kiểm chứng

Dự án hoàn thành chuỗi từ chuẩn bị dữ liệu đến đánh giá test có khóa cấu hình, bootstrap theo nhóm, audit và replay. Kết luận trọng tâm là một classifier tuyến tính kết hợp đặc trưng lexical/cú pháp đã phân loại tốt năm nhãn trong thiết lập original-only known-span; các lỗi còn lại tập trung ở ranh giới taxonomy và cách biểu diễn span/ngữ cảnh. Các nhận định này không mở rộng thành tuyên bố hiểu ngữ pháp hoàn chỉnh, chất lượng sửa lỗi hay khả năng tổng quát hóa sang mọi corpus.

# Phụ lục A. Cấu hình được chọn và bảng đầy đủ

Bảng 30. Cấu hình của toàn bộ tên pipeline LR/Tree

| ID | Variant | Chiều | Cấu hình |
| --- | --- | --- | --- |
| LR_00 | S+L | 9203 | {"C": 1.0, "class_weight": null} |
| Tree_00 | S+L | 9203 | {"max_depth": 12, "min_samples_leaf": 5, "class_weight": null} |
| LR_01 | S+L+P | 15271 | {"C": 1.0, "class_weight": null} |
| Tree_01 | S+L+P | 15271 | {"max_depth": 12, "min_samples_leaf": 5, "class_weight": null} |
| LR_02 | S+L+P+M | 15302 | {"C": 1.0, "class_weight": null} |
| Tree_02 | S+L+P+M | 15302 | {"max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"} |
| LR_03 | S+L+P+M+D | 16180 | {"C": 0.1, "class_weight": null} |
| Tree_03 | S+L+P+M+D | 16180 | {"max_depth": 8, "min_samples_leaf": 5, "class_weight": null} |
| LR_04 | Full_mask_span_lexical | 16180 | {"C": 0.1, "class_weight": null} |
| Tree_04 | Full_mask_span_lexical | 16180 | {"max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"} |
| LR_05 | S+P+M+D_nonlexical | 6986 | {"C": 1.0, "class_weight": null} |
| Tree_05 | S+P+M+D_nonlexical | 6986 | {"max_depth": 8, "min_samples_leaf": 20, "class_weight": "balanced"} |
| LR_06 | Full_minus_P | 10112 | {"C": 1.0, "class_weight": null} |
| Tree_06 | Full_minus_P | 10112 | {"max_depth": 8, "min_samples_leaf": 5, "class_weight": "balanced"} |
| LR_07 | Full_minus_M | 16149 | {"C": 0.1, "class_weight": null} |
| Tree_07 | Full_minus_M | 16149 | {"max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"} |
| LR_08 | Full_minus_D | 15302 | {"C": 1.0, "class_weight": null} |
| Tree_08 | Full_minus_D | 15302 | {"max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"} |

Nguồn: reports/final_evaluation/selection_summary.json; Full−D alias của S+L+P+M

Bảng 31. Toàn bộ 20 kết quả cuối, gồm CI và alias

| ID | Val F1 | Test F1 | CI95% test | Alias |
| --- | --- | --- | --- | --- |
| Majority | 0.1184 | 0.1184 | [0.1104; 0.1279] | — |
| POS | 0.7048 | 0.7106 | [0.6766; 0.7365] | — |
| LR_00 | 0.7164 | 0.7203 | [0.6875; 0.7467] | — |
| Tree_00 | 0.5832 | 0.5850 | [0.5601; 0.6052] | — |
| LR_01 | 0.7956 | 0.7968 | [0.7659; 0.8195] | — |
| Tree_01 | 0.7548 | 0.7520 | [0.7193; 0.7774] | — |
| LR_02 | 0.8048 | 0.8102 | [0.7775; 0.8349] | — |
| Tree_02 | 0.7929 | 0.7967 | [0.7674; 0.8201] | — |
| LR_03 | 0.8067 | 0.8163 | [0.7876; 0.8382] | — |
| Tree_03 | 0.7879 | 0.7996 | [0.7693; 0.8246] | — |
| LR_04 | 0.8016 | 0.8028 | [0.7758; 0.8245] | — |
| Tree_04 | 0.7876 | 0.8006 | [0.7727; 0.8252] | — |
| LR_05 | 0.7964 | 0.7881 | [0.7598; 0.8108] | — |
| Tree_05 | 0.7862 | 0.7894 | [0.7572; 0.8170] | — |
| LR_06 | 0.8001 | 0.7982 | [0.7700; 0.8204] | — |
| Tree_06 | 0.7711 | 0.7811 | [0.7533; 0.8032] | — |
| LR_07 | 0.8015 | 0.8123 | [0.7850; 0.8347] | — |
| Tree_07 | 0.7751 | 0.7826 | [0.7512; 0.8060] | — |
| LR_08 | 0.8048 | 0.8102 | [0.7775; 0.8349] | LR_02 |
| Tree_08 | 0.7929 | 0.7967 | [0.7674; 0.8201] | Tree_02 |

Nguồn: reports/evaluation_followup/reviewed_overall_results.csv

# Phụ lục B. Ánh xạ code và bằng chứng

Bảng 32. Vai trò các module trong pipeline

| Nhóm module | Vai trò |
| --- | --- |
| src/data/nucle_sgml.py, parse_nucle.py | Đọc SGML, tạo annotation table và kiểm offset |
| data_understanding.py, data_quality.py | Thống kê, missing/no-op/overlap/duplicate audit |
| prepare_nucle.py, sentence_boundaries.py, boundary_challenges.py | Preparation v1/v2, invariants và boundary flags |
| group_split_nucle.py, compare_preparation.py | Connected groups/split; đối chiếu preparation |
| src/features/nucle_pos.py, nucle_features.py | POS baseline và encoder train-only S/L/P/M/D |
| src/models/nucle_logistic.py, nucle_tree.py | Sparse LR và deterministic Gini Tree |
| src/training/nucle_benchmark.py, nucle_ablation.py, nucle_final_selection.py | Grid, ablation, refit, selection và lock |
| src/training/bert_span_classifier.py; models/bert_span_readout.py | Span-marked BERT và CLS/span mean |
| src/evaluation/nucle_final_evaluation.py, metrics.py, nucle_final_error_audit.py | Scoring, bootstrap, slices và audit IDs |
| src/evaluation/review_saved_evidence.py, review_audit_evidence.py, reproduce_evaluation_pipeline.py | Kiểm số học, nguồn audit và raw-to-prediction replay |
| src/reporting/; notebooks/01–04 | Sinh bảng/hình/tài liệu; notebook là snapshot dữ liệu/validation |
| tests/ | Kiểm invariants, feature/model, metric, bootstrap và serialization |

Nguồn: Code hiện có; danh mục từng file và checksum trong source_manifest.json

Artifact kiểm số liệu: reviewed_overall_results.csv; final_evaluation/test_metrics.json, paired_differences.csv, bootstrap.json, per_class_results.csv và per_class_deltas.csv. Artifact kiểm tái lập: evaluation_followup/pipeline_reproduction.json, evidence_review.json, audit_followup.json và expert_review_queue.csv. Báo cáo này không nhúng câu nguồn/correction riêng của NUCLE; dùng ID và mô tả lỗi để truy vết.

# Phụ lục C. Trạng thái bàn giao, đóng góp cá nhân và sử dụng AI

Đã hoàn tất đánh giá test cho các pipeline cổ điển, paired group-bootstrap, audit và kiểm chứng tái lập trong phạm vi mục 10. Báo cáo này là bản tổng kết kết quả hiện hành; các báo cáo tuần 1 và báo cáo task là snapshot lịch sử. BERT-small chỉ có kết quả validation và chẩn đoán train, chưa có điểm test. Theo dõi tiến độ hiện tại tại docs/status.md.

Trước khi nộp bản cuối, nhóm cần xác nhận tên thành viên và nhiệm vụ thực tế, liên kết đến code/log hoặc phân tích do từng người phụ trách, quyết định đã kiểm tra và kết quả kiểm tra. Báo cáo hiện chưa có xác nhận phân công theo tên; không suy đóng góp cá nhân từ danh sách module hoặc từ kết quả chung. Không bổ sung nhật ký hồi tố như thể đã được ghi tại thời điểm thực nghiệm.

Audit ngôn ngữ có hỗ trợ AI và chưa thay thế đánh giá độc lập của chuyên gia. Nhóm cần hoàn thiện khai báo tên công cụ AI đã dùng cho giải thích, viết, gỡ lỗi hoặc phân tích, phạm vi hỗ trợ và cách từng thành viên tự kiểm tra. Replay, kiểm số học và đối chiếu theo ID là bằng chứng kiểm chứng kỹ thuật trong phạm vi đã nêu; không tự xác nhận chất lượng phán đoán ngôn ngữ của AI. Có 17 ca chờ chuyên gia; chưa đổi gold.

Cho interview, lưu bản báo cáo hiện tại cùng script/cấu hình và output có thể xem ngoại tuyến; chuẩn bị một mẫu có ID từ đầu vào/đặc trưng đến prediction và gold, một trường hợp lỗi đã tự đọc, sơ đồ split và một kiểm tra nhỏ trên dữ liệu phát triển hoặc output đã lưu. Mỗi thành viên cần tự giải thích toàn bộ pipeline và phần việc của mình. Chuẩn bị PDF, kiểm tra bố cục và quyền truy cập trước hạn nộp trên LMS; chưa xác nhận việc nộp bài trong báo cáo này.

# Tài liệu tham khảo

[REF01] Dahlmeier, D., Ng, H. T., & Wu, S. M. (2013). **Building a Large Annotated Corpus of Learner English: The NUS Corpus of Learner English.** Proceedings of the Eighth Workshop on Innovative Use of NLP for Building Educational Applications, 22–31. [ACL Anthology](https://aclanthology.org/W13-1703/).

[REF02] Ng, H. T., Wu, S. M., Wu, Y., Hadiwinoto, C., & Tetreault, J. (2013). **The CoNLL-2013 Shared Task on Grammatical Error Correction.** Proceedings of the Seventeenth Conference on Computational Natural Language Learning: Shared Task, 1–12. [ACL Anthology](https://aclanthology.org/W13-3601/).

[REF03] Bryant, C., Yuan, Z., Qorib, M. R., Cao, H., Ng, H. T., & Briscoe, T. (2023). **Grammatical Error Correction: A Survey of the State of the Art.** Computational Linguistics, 49(3), 643–701. [DOI/nhà xuất bản](https://doi.org/10.1162/coli_a_00478).

[REF04] Davis, C., Bryant, C., Caines, A., Rei, M., & Buttery, P. (2022). **Probing for targeted syntactic knowledge through grammatical error detection.** Proceedings of the 26th Conference on Computational Natural Language Learning, 360–373. [ACL Anthology](https://aclanthology.org/2022.conll-1.25/).

[REF07] Yang, L., Li, H., Li, L., Xu, C., Xia, S., & Yuan, C. (2023). **LET: Leveraging Error Type Information for Grammatical Error Correction.** Findings of the Association for Computational Linguistics: ACL 2023, 5986–5998. [ACL Anthology](https://aclanthology.org/2023.findings-acl.371/).

[REF08] Qorib, M. R., Aji, A. F., & Ng, H. T. (2024). **Efficient and Interpretable Grammatical Error Correction with Mixture of Experts.** Findings of the Association for Computational Linguistics: EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.997/).

[REF09] Song et al. (2024). **GEE! Grammar Error Explanation with Large Language Models.** Findings of the Association for Computational Linguistics: NAACL 2024. [ACL Anthology](https://aclanthology.org/2024.findings-naacl.49/).

[REF10] Coyne et al. (2025). **Annotating Errors in English Learners’ Written Language Production: Advancing Automated Written Feedback Systems.** arXiv:2508.06810, bản tác giả. [arXiv](https://arxiv.org/abs/2508.06810).

[REF11] Nagata, R., Kimura, M., & Hanawa, K. (2022). **Exploring the Capacity of a Large-scale Masked Language Model to Recognize Grammatical Errors.** Findings of the Association for Computational Linguistics: ACL 2022, 4107–4118. [ACL Anthology](https://aclanthology.org/2022.findings-acl.324/).

[REF12] Koyama, A., Mita, M., Yoon, S.-Y., Takama, Y., & Komachi, M. (2025). **Targeted Syntactic Evaluation for Grammatical Error Correction.** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 21108–21125. [ACL Anthology](https://aclanthology.org/2025.acl-long.1026/).

[REF13] Choshen, L., Nikolaev, D., Berzak, Y., & Abend, O. (2020). **Classifying Syntactic Errors in Learner Language.** Proceedings of the 24th Conference on Computational Natural Language Learning, 97–107. [ACL Anthology](https://aclanthology.org/2020.conll-1.7/).

[REF14] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** Proceedings of NAACL-HLT 2019 (Volume 1: Long and Short Papers), 4171–4186. [ACL Anthology](https://aclanthology.org/N19-1423/).

[REF15] Hashemi, H. B., & Hwa, R. (2016). **An Evaluation of Parser Robustness for Ungrammatical Sentences.** Proceedings of EMNLP 2016. [ACL Anthology](https://aclanthology.org/D16-1182/).

[REF20] Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **Well-Read Students Learn Better: On the Importance of Pre-training Compact Models.** arXiv:1908.08962. [bài gốc](https://arxiv.org/abs/1908.08962); [model card Google](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).

Thông tin thư mục và cách vận dụng được kế thừa từ reference.md và docs/literature của dự án; đây không phải đợt rà literature trực tuyến mới. Các mục Song/Coyne giữ cách ghi et al. theo danh mục nguồn. Các kết quả của dự án lấy từ artifact nội bộ, không suy từ các bài tham khảo.
