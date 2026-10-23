# Ghi chú tổng hợp related work

Lưu lập luận đã viết về MoECE/GEE!/Coyne, không phải danh sách PRISMA cuối. Thông tin thư mục chuẩn: [reference.md](../../reference.md), REF08–REF10 tương ứng R1–R3 bên dưới. Đánh giá bài mới: [bảng related work hiện có](contextual_models_related_work.md).

2. Nghiên cứu liên quan và cơ sở lựa chọn bài toán
2.1 Phạm vi lựa chọn
Ba bài được lựa chọn dưới đây có quan hệ bổ trợ: dự đoán loại lỗi trong GEC, giải thích edit đã trích xuất, và sinh phản hồi với vị trí/nhãn lỗi đã biết. Chưa xác minh được trong tập bài này một nghiên cứu trùng hoàn toàn thiết lập câu gốc + gold span → năm nhãn NUCLE, không dùng correction.
2.2 Các nghiên cứu được lựa chọn
Nghiên cứu
Năm và nơi công bố
Toàn văn miễn phí
Qorib, Aji & Ng — Efficient and Interpretable Grammatical Error Correction with Mixture of Experts [R1]
2024, Findings of EMNLP
https://aclanthology.org/2024.findings-emnlp.997.pdf
Song et al. — GEE! Grammar Error Explanation with Large Language Models [R2]
2024, Findings of NAACL
https://aclanthology.org/2024.findings-naacl.49.pdf
Coyne et al. — Annotating Errors in English Learners’ Written Language Production: Advancing Automated Written Feedback Systems [R3]
2025, bản tác giả arXiv:2508.06810
https://arxiv.org/pdf/2508.06810

2.3 Ngữ cảnh, đối chứng và quan hệ với năm nhãn
Bài
Ngữ cảnh/đầu vào và nhiệm vụ thật
Baseline hoặc đối chứng trong bài
Nhãn gần và mức liên quan
MoECE [R1]
Câu nguồn được mã hóa; router của decoder dự đoán loại lỗi theo token trong quá trình sửa câu
T5-v1.1 dense; MoE routers; ablation có/không error-type loss
Nhãn huấn luyện do ERRANT tạo; liên quan DET/PREP và các nhóm noun/verb. Không phải gold-span classifier độc lập
GEE! [R2]
Cặp câu sai–đã sửa; trích atomic edits rồi giải thích từng edit
Prompt trực tiếp GPT-4 so với pipeline trích edit rồi giải thích
Loại lỗi tiếng Anh được sinh gồm determiner/article, preposition, noun number, verb form và subject–verb agreement; nhãn tự do, không phải taxonomy năm lớp cố định
Coyne et al. [R3]
Câu gốc có error/correction và comment highlight; dùng oracle upstream annotations để sinh feedback
Keyword-guided theo các taxonomy, keyword-free baseline và template-guided
Typology sư phạm và các tag EXPECT/ERRANT. Gần bài toán span-known ở bước feedback, không đánh giá tự phân loại năm nhãn

MoECE (2024). Nhóm dùng bài này để củng cố việc xem loại lỗi là thông tin có ích cho hệ thống có thể diễn giải. Mô hình bổ sung classification head vào router, huấn luyện với error-type loss và dùng nhãn ERRANT. Baseline chính là các mô hình T5 dense có effective parameter count tương ứng. Nhóm kế thừa câu hỏi về lợi ích của error-type information và ablation, không kế thừa mô hình MoE. Vì dự đoán diễn ra trong decoder của hệ sửa lỗi, đây không phải đối chứng cùng đầu vào hay metric với LR/Tree của nhóm [R1, §3–6].
GEE! (2024). Nhóm dùng bài này làm cơ sở tách việc định vị edit khỏi xử lý thông tin về từng lỗi. Hệ thống nhận cả câu sai và câu đã sửa; edit extraction cung cấp atomic edits trước khi GPT-4 sinh giải thích. Các loại lỗi gần năm nhãn xuất hiện trong kết quả tiếng Anh nhưng là nhãn sinh tự do. Vì vậy không lấy tỷ lệ giải thích đúng để so với macro-F1 năm lớp. Bài gợi ý phân tích riêng lỗi định vị/ngữ cảnh và lỗi gán loại; không chứng minh correction cần có trong đầu vào của dự án [R2, §3–6, Table 15].
Coyne et al. (2025). Nhóm chọn bài này cho động cơ sử dụng kết quả phân loại trong phản hồi học tập. Trong thí nghiệm, error detection, classification và highlight được đặt ở ground truth; baseline keyword-free bỏ nhãn lỗi, còn các thiết lập khác dùng tag hoặc template. Vì thế bài không chứng minh khả năng dự đoán nhãn khi span đã biết. Các hệ có/không keyword cho phản hồi tương tự, nên nhóm không mặc định thêm nhãn chắc chắn tăng chất lượng feedback. Nhóm chỉ dùng nghiên cứu để xác định ứng dụng tiềm năng và ranh giới giữa classification và feedback [R3, §3–4].
2.4 Cơ sở lựa chọn bài toán và thiết kế nhóm
Nhóm giữ phạm vi phân loại năm loại lỗi tại gold span để đo trực tiếp giá trị của thông tin câu gốc. MoECE cho thấy error type là một thành phần nghiên cứu gần đây; GEE! cho thấy lợi ích của việc tách bước xử lý từng edit; Coyne et al. cung cấp bối cảnh feedback khi thông tin upstream đã biết. Từ đó, nhóm thiết kế Majority/POS rule/LR/Tree và ablation/masking như phép kiểm tra riêng của dự án, không khẳng định ba bài trên đã đề xuất đúng các baseline này.
Năm nhãn được giữ theo taxonomy của file NUCLE đang sử dụng. Không chuyển nhãn sinh tự do của GEE! hoặc typology sư phạm sang nhãn NUCLE mà không có quy tắc kiểm chứng. Các mô tả article/determiner, preposition, noun number, verb form và SVA ở GEE! chỉ là tương ứng khái niệm, không phải relabeling.
Nhóm ba bài trên là ghi chú đọc tài liệu 2024–2025, không quy định cửa sổ năm cho toàn bộ review. Tiêu chí năm trong Excel còn cần thống nhất theo kế hoạch; NUCLE/CoNLL vẫn là tài liệu nguồn dữ liệu. Nhóm chưa tuyên bố khoảng trống tuyệt đối của toàn bộ lĩnh vực: kết luận hiện tại chỉ là ba bài được xác minh chưa trùng hoàn toàn thiết lập original-only của dự án.
