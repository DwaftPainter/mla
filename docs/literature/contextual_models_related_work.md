# Related work phục vụ thực nghiệm tuần 1

Hoàn thành phần tổng hợp ngày 23/10/2026. Đây là bảng phục vụ report, không thay đổi Excel hoặc quyết định PRISMA. Mã trích dẫn theo [reference.md](../../reference.md).

| Nguồn | Input → output | Dữ liệu/đánh giá | Vận dụng và giới hạn |
|---|---|---|---|
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
| [REF16, XGEC](https://ieeexplore.ieee.org/document/11598876) | Câu/edit → sửa lỗi và nhãn phân cấp | Nhãn ánh xạ, accuracy/coverage khác task nhóm | Future work có điều kiện; đã đọc bản tác giả, chưa truy cập được IEEE trong lượt review. |

## Tổng hợp cho report

Những nghiên cứu trên hỗ trợ ba lựa chọn: dùng loại lỗi như một đầu ra có ý nghĩa cho ứng dụng GEC, kiểm tra đặc trưng cú pháp bằng các đối chứng rõ ràng, và đánh giá riêng những nhóm mà POS không phân biệt tốt. Chúng không cung cấp phép so điểm trực tiếp với classifier của nhóm vì khác đầu vào, taxonomy hoặc metric.

Thí nghiệm tuần 1 giữ nguyên câu gốc và gold span, năm nhãn và split. BERT được thêm như một cách biểu diễn ngữ cảnh, so với LR vốn đã có morphology/dependency từ parse toàn câu. Giả thuyết cần kiểm tra là representation này có cải thiện trên toàn validation và các cặp Vform/SVA, Nn/ArtOrDet hay không; không giả định BERT chắc chắn tốt hơn và không chỉ báo nhóm thắng.

REF17 không dùng làm bằng chứng thực nghiệm chính; REF18 chỉ là tài liệu tùy chọn về giải thích; REF19 chỉ phục vụ ghi chú screening lịch sử. Không thêm triển khai LET/XGEC/SHAP vào tuần 1.
