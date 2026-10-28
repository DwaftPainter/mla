
## BỔ SUNG THEO NHẬN XÉT TUẦN 9 — ngày 20/10/2026

Phần bổ sung ngày 20/10/2026 xem xét kết quả validation theo từng nhóm span và cặp nhãn, thay vì chỉ dựa vào điểm trung bình toàn tập. Các phép tính dùng dự đoán đã lưu, không huấn luyện lại hoặc chấm test. Kết quả nằm trong reports/nucle_span_diagnostics_v2/summary.json và metrics.csv; mã thực hiện là src/evaluation/nucle_span_analysis.py.


## 1. Tiếp thu nhận xét và điều chỉnh kết luận

Câu hỏi cần làm rõ là: khi POS của span chưa đủ để phân biệt nhãn, mô hình có thêm thông tin gì? POS rule đạt 0,705 và LR không dùng từ vựng đạt 0,796, nhưng hai con số chưa cho thấy nhãn chỉ phụ thuộc POS. LR không từ vựng vẫn có S/P/M/D, còn parser đọc cả câu. Tương tự, Nn và ArtOrDet có thể liên quan cùng một danh ngữ, nhưng không phải mọi span trong cặp đều cùng POS hoặc đã được parser xác nhận là một cụm danh từ.

Cần phân biệt hai loại khó: (a) cặp nhãn thật Vform/SVA hoặc Nn/ArtOrDet; (b) các mẫu trong cặp còn có cùng POS signature. Phân tích chính dưới đây dùng (a), và mục 5 bổ sung (b) để kiểm tra sát giả thuyết POS không đủ. Không thay đổi taxonomy hoặc huấn luyện lại classifier nhị phân.


## 2. Số lượng span rỗng, một token và nhiều token

Quy ước: rỗng khi error_start=error_end; một token khi span giao đúng một token; nhiều token khi giao ít nhất hai token trong artifact preparation. “Một từ” được vận hành hóa bằng một token, không đếm theo khoảng trắng. Partial-token vẫn thuộc nhóm một token nếu chỉ giao một token; dấu câu theo tokenizer cũng là token.

| Split | Rỗng | Một token | Nhiều token | Tổng |
| --- | --- | --- | --- | --- |
| train | 0 | 9807 | 1245 | 11052 |
| validation | 0 | 2114 | 254 | 2368 |
| test | 0 | 2102 | 266 | 2368 |
| Tổng | 0 | 14023 | 1765 | 15788 |

Số lượng test ở đây chỉ là thống kê metadata của split đã tồn tại, không phải điểm test. Đối chiếu token count preparation với POS signature của validation có 0 bất đồng; phép kiểm này không chứng minh mọi token boundary hoặc POS đều đúng về ngôn ngữ.

Có 0 span rỗng trong 15.788 mẫu mục tiêu; CSV annotation nguồn tương ứng cũng có 0 annotation cùng paragraph và start_off=end_off. Vì vậy không thể báo F1 cho span rỗng: phải ghi N/A, support=0, không ghi F1=0. Không suy từ đó rằng NUCLE hoặc người học không có lỗi thiếu từ: lỗi thiếu từ có thể được annotation bằng đoạn không rỗng tùy quy ước. Cần đọc nguồn annotation/correction nếu muốn thống kê missing-edit về ngôn ngữ.

| Nhóm span validation | Support | POS Macro-F1 (5 lớp) | LR Full Macro-F1 (5 lớp) | POS accuracy | LR accuracy |
| --- | --- | --- | --- | --- | --- |
| empty | 0 | N/A | N/A | N/A | N/A |
| single_token | 2114 | 0.7087 | 0.8154 | 0.7606 | 0.8557 |
| multi_token | 254 | 0.4505 | 0.5195 | 0.7323 | 0.7677 |

| Span validation | ArtOrDet | Prep | Nn | Vform | SVA |
| --- | --- | --- | --- | --- | --- |
| single_token | 834 | 352 | 530 | 176 | 222 |
| multi_token | 162 | 10 | 35 | 40 | 7 |

Span nhiều token khó hơn theo Macro-F1: LR 0,5195 so với 0,8154 ở span một token. Tuy nhiên multi-token chỉ có 254 mẫu, gồm 7 SVA và 10 Prep; accuracy 0,7677 bị chi phối bởi phân bố lớp, trong khi Macro-F1 tính đều năm lớp. Không quy toàn bộ giảm điểm cho độ dài: lớp, cấu trúc span và parser có thể đồng thời thay đổi. Chưa có CI để xác nhận độ ổn định.


## 3. Nhầm Nn → ArtOrDet có nằm ở span rỗng không?

| Model | Rỗng | Một token | Nhiều token | Tổng lỗi Nn → ArtOrDet |
| --- | --- | --- | --- | --- |
| POS rule | 0 | 51 | 30 | 81 |
| LR S+L+P+M+D | 0 | 78 | 16 | 94 |

Câu trả lời với artifact hiện tại là không: cả 94 lỗi LR Nn→ArtOrDet đều không rỗng, trong đó 78 một token và 16 nhiều token. POS có 81 lỗi theo hướng này, gồm 51 và 30. LR tốt hơn Macro-F1 chung nhưng không giảm mọi ô nhầm; không dùng kết quả tổng thể để khẳng định tất cả cặp đã cải thiện. Các con số chưa giải thích nguyên nhân ngôn ngữ; cần audit một mẫu lỗi để xác nhận.


## 4. F1 trên hai cặp nhãn khó — định nghĩa phép tính

Chọn mẫu bằng nhãn thật: H_V={i:y_i∈{Vform,SVA}}, H_N={i:y_i∈{Nn,ArtOrDet}}. Giữ nguyên prediction năm lớp của POS và LR. Không ép argmax vào hai nhãn, không loại mẫu dự đoán ra ngoài cặp, không fine-tune trên validation.

Trong mỗi H, tính TP/FP/FN và F1 riêng cho hai nhãn của cặp, rồi Macro-F1_pair=(F1_a+F1_b)/2. Dự đoán ngoài cặp vẫn là false negative của nhãn thật và là lỗi accuracy; báo số ngoài cặp riêng. Precision được tính lại sau khi lọc ground truth, nên F1_pair không phải trung bình hai F1 lấy từ bảng toàn validation. Đây là đánh giá post-hoc trên validation, không phải ước lượng test.

| Cặp nhãn | Support | POS Macro-F1_pair | LR Full Macro-F1_pair | Δ LR−POS | POS ngoài cặp | LR ngoài cặp |
| --- | --- | --- | --- | --- | --- | --- |
| Vform/SVA | 445 | 0.5739 | 0.7417 | 0.1678 | 73 | 62 |
| Nn/ArtOrDet | 1561 | 0.8065 | 0.8869 | 0.0804 | 88 | 51 |

| Cặp | Nhãn | Support | POS F1 trong cặp | LR Full F1 trong cặp |
| --- | --- | --- | --- | --- |
| Vform/SVA | Vform | 216 | 0.6311 | 0.7062 |
| Vform/SVA | SVA | 229 | 0.5167 | 0.7773 |
| Nn/ArtOrDet | Nn | 565 | 0.7887 | 0.8649 |
| Nn/ArtOrDet | ArtOrDet | 996 | 0.8242 | 0.9089 |

LR Full cải thiện +0,1678 trên Vform/SVA và +0,0804 trên Nn/ArtOrDet. Đây là bằng chứng trực tiếp hơn rằng model dùng thêm tín hiệu có ích khi chỉ POS lookup chưa đủ. Chưa kết luận cần BERT hoặc chỉ ngữ cảnh xa mới giải quyết được, vì LR hiện đã có morphology/dependency và đặc trưng parser từ toàn câu.


## 5. Phân tích bổ sung: giữ cố định POS signature

Đã kiểm bốn signature đơn được xác định rõ VERB, AUX, NOUN, DET cho cả hai cặp; toàn bộ kết quả nằm trong summary.json, kể cả nhóm nhỏ/thiếu lớp. Bảng dưới trình bày VERB/AUX cho Vform/SVA và NOUN/DET cho Nn/ArtOrDet, phù hợp câu hỏi giảng viên. Đây là POS của span do parser dự đoán; không gọi NOUN signature là bằng chứng noun phrase hoàn chỉnh.

| Cặp | POS đơn | Support hai lớp | POS Macro-F1_pair | LR Full Macro-F1_pair |
| --- | --- | --- | --- | --- |
| Vform/SVA | VERB | Vform=129, SVA=114 | 0.3468 | 0.7594 |
| Vform/SVA | AUX | Vform=14, SVA=84 | 0.4615 | 0.7216 |
| Nn/ArtOrDet | NOUN | Nn=470, ArtOrDet=161 | 0.4269 | 0.8445 |
| Nn/ArtOrDet | DET | Nn=15, ArtOrDet=338 | 0.4891 | 0.5524 |

Cùng signature VERB, POS đạt 0,3468 còn LR 0,7594; cùng NOUN, POS 0,4269 còn LR 0,8445. POS lookup trả một nhãn cố định cho signature, nên khó phân biệt hai nhãn trong nhóm này. Kết quả LR cho thấy có thông tin hữu ích ngoài signature span. Tuy vậy các nhóm có độ cân bằng khác nhau; DET chỉ có 15 Nn so với 338 ArtOrDet, cần báo support và tránh kết luận mạnh từ nhóm nhỏ.


## 6. Thông tin nào giúp trên tập khó? Dùng ablation đã có

| Model/variant | Vform/SVA Macro-F1_pair | Nn/ArtOrDet Macro-F1_pair |
| --- | --- | --- |
| POS rule | 0.5739 | 0.8065 |
| LR S+L | 0.6114 | 0.8460 |
| LR S+L+P | 0.7089 | 0.8829 |
| LR S+L+P+M | 0.7349 | 0.8817 |
| LR S+L+P+M+D | 0.7417 | 0.8869 |
| LR Full_mask_span_lexical | 0.7441 | 0.8728 |
| LR S+P+M+D_nonlexical | 0.7404 | 0.8610 |

Vform/SVA: thêm P nâng LR 0,6114→0,7089; thêm M lên 0,7349; thêm D lên 0,7417. Nonlexical còn 0,7404, rất gần Full; masked 0,7441 nhỉnh hơn Full một lượng nhỏ chưa có CI. Nn/ArtOrDet: Full 0,8869 so với nonlexical 0,8610, cho thấy lexical có lợi rõ hơn về mức chênh mô tả trong cặp này. Các cấu hình được chọn theo Macro-F1 toàn validation từ thí nghiệm cũ, không retune theo tập khó.


## 7. Split, seed và phiên bản parser/tagger

| Split | Mẫu target | Essay có target | Group có target | Tất cả essay | Tất cả group |
| --- | --- | --- | --- | --- | --- |
| test | 2368 | 144 | 97 | 147 | 97 |
| train | 11052 | 1069 | 1045 | 1108 | 1084 |
| validation | 2368 | 142 | 99 | 142 | 99 |

Tất cả essay cộng thành 1.397; tất cả group cộng thành 1.280. Essay có target khác tổng essay: một essay không có target vẫn có thể thuộc group có target và theo group sang validation/test. Group hoàn toàn không có target được gán train. Vì vậy không cộng các bảng có mẫu số khác nhau rồi cho rằng split bị thiếu dữ liệu.

Seed split: 20261013; 128 restarts; group policy nucle-grouping-v1. spaCy 3.8.7, en_core_web_sm 3.8.0; các thành phần tok2vec, tagger, parser, attribute_ruler, lemmatizer; NER tắt. Tagger và parser là hai thành phần của cùng model package, không có số phiên bản độc lập được báo trong artifact. Encoder fit train; test chưa chấm.


## 8. Trả lời hai câu hỏi của giảng viên

Với một lỗi thiếu mạo từ có span rỗng: span POS=NOT_APPLICABLE, M của span không có giá trị và các D dựa trên token span không có chủ ngữ/head. S báo span_kind=insertion, span_tokens=0, span_chars=0 và vị trí; L/P của token lân cận có thể gợi ý trước một danh từ cần determiner. POS baseline hiện fallback ArtOrDet vì đó là Majority train, không phải vì nó nhận diện được cấu trúc thiếu mạo từ. Dữ liệu hiện có 0 span rỗng nên chưa có bằng chứng model đã học xử lý tình huống này; encoder có thể xem insertion là giá trị chưa thấy. Không tuyên bố hỗ trợ thực nghiệm cho insertion chỉ vì code có nhánh xử lý.

Để phân biệt SVA và Vform, người chấm cần nhìn chủ ngữ và đặc điểm số/ngôi, động từ hữu hạn, các trợ động từ/modal, dạng infinitive/participle và cấu trúc chi phối động từ trong câu. LR hiện đã có M và D từ parse toàn câu, nên có thể lấy thông tin chủ ngữ ngoài cửa sổ lexical ± 2. Hạn chế là parser có thể sai và các đặc trưng tổng hợp không biểu diễn đầy đủ mọi quan hệ. Vì vậy luận điểm cần kiểm tra là contextual representation có bổ sung cho cả L/P/M/D không, không phải LR hoàn toàn không nhìn được chủ ngữ xa.


## 9. Thử mô hình ngữ cảnh — kế hoạch mở rộng, chưa chạy

Giữ BERT fine-tune như đề xuất thử nghiệm của giảng viên; không có điểm BERT trong tài liệu này. Trước khi chạy, chốt budget, checkpoint/version, seeds, số epoch, optimizer, learning rate, max_length, dừng sớm và quy tắc chọn checkpoint; chỉ dùng train/validation hiện có. Việc bổ sung model sau phản hồi là thay đổi protocol cần ghi ngày, không trình bày là đã đăng ký từ đầu.

Thiết kế so sánh ưu tiên classifier năm lớp trên cùng train để giữ nhiệm vụ: câu gốc có marker mở/đóng span, không correction hoặc nhãn gốc trong input; offsets dùng chèn marker. Chốt cách subword tokenize, xử lý partial-token và truncation bảo đảm không mất span; báo số câu bị cắt. Thêm marker vào tokenizer thì phải cập nhật embedding tương ứng. Không cần giả dữ liệu span rỗng để tạo điểm insertion.

Chọn checkpoint bằng Macro-F1 toàn validation như LR, rồi báo cùng bảng toàn tập, span bins, hai cặp và các POS slices; không chỉ báo nơi BERT thắng. Nếu train riêng binary classifier cho mỗi cặp, cần định nghĩa đó là thí nghiệm khác và huấn luyện LR/POS đối chứng phù hợp; không so trực tiếp với bảng five-class-slice này như cùng nhiệm vụ.

Muốn kết luận “BERT chỉ hơn ở phần khó”, phải định nghĩa phần dễ trước khi xem kết quả, báo support và Δ trên cả phần dễ/khó với cùng mẫu. Cặp nhãn khó không tự là phần bù của phần dễ; có thể dùng chẩn đoán signature trộn nhãn được xác định từ train hoặc các slice POS đã khóa. Sau khóa cấu hình mới chấm test và paired group-bootstrap. Không coi phần tăng điểm nhỏ chưa có CI là kết luận chắc chắn.


## 10. Bàn giao tuần sau và cách tái lập

| Yêu cầu | Trạng thái |
| --- | --- |
| Counts và metrics theo span | Đã có; rỗng N/A, single/multi validation |
| F1 cặp Vform/SVA, Nn/ArtOrDet cho POS/LR | Đã có; giữ prediction năm lớp |
| Chứng cứ khi cùng POS | Đã có thêm VERB/AUX/NOUN/DET slices |
| Split, seed, tagger/parser | Đã bổ sung cả target-only và all-essay counts |
| Audit câu/parser để giải thích nguyên nhân | Chưa làm trong lần bổ sung này |
| BERT/context model | Kế hoạch tùy ngân sách; chưa chạy |
| Final test và CI | Chưa thực hiện |

Lệnh tái tạo: py src/evaluation/nucle_span_analysis.py --output reports/nucle_span_diagnostics_new. Output phải mới. Script chỉ đọc prediction validation và metadata; kiểm cùng sample IDs/labels/split hash, kiểm prediction tái tạo đúng Macro-F1 toàn validation của từng model trước khi tính slice. summary.json giữ input hashes; metrics.csv giữ số liệu aggregate. Không chứa câu corpus hoặc correction.

Việc cần làm tiếp: review các định nghĩa slice và các khác biệt code–protocol đã nêu trong báo cáo, rồi audit một số lỗi Nn/ArtOrDet và Vform/SVA. Nếu còn ngân sách mới thêm contextual model; không cần đổi model để đáp ứng ba bảng bắt buộc tuần sau vì các bảng đó đã hoàn tất.

