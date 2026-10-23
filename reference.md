# Tài liệu tham khảo của dự án

Cập nhật: 23/10/2026. Tổng hợp từ `docs/proposal.md`, `note.md`, ghi chú đọc bài trước đây và kế hoạch hai tuần. File `docs/literature_excel_review.md` từng được tham chiếu hiện không có trong workspace; bảng tổng hợp đang có ở [contextual_models_related_work.md](docs/literature/contextual_models_related_work.md). Danh mục gồm các bài đã trích, dự kiến dùng và tài liệu có điều kiện; không phải toàn bộ kết quả tìm kiếm trong Excel.

“Đã dùng” nghĩa là đã xuất hiện trong proposal/ghi chú nghiên cứu, không có nghĩa đã triển khai mô hình của bài. “Dự kiến” nghĩa là sẽ tích hợp vào report hoặc dùng làm cơ sở thí nghiệm. Các bài chỉ phục vụ rà screening được tách riêng, không tự tính vào included studies. Mã REF dưới đây độc lập với mã R1–R3 cũ trong note.md.

Bổ sung sau tuần 1: REF07, REF11–REF15 và REF20 đã được vận dụng trong related work/protocol/report mới. Trạng thái dự kiến ở các mục dưới mô tả thời điểm lập danh mục; kết quả thực thi xem [báo cáo contextual validation](reports/summaries/contextual/contextual_model_validation_report.docx).

## A. Các bài đã dùng trong proposal hoặc ghi chú

### REF01 — Nguồn bộ dữ liệu NUCLE

Dahlmeier, D., Ng, H. T., & Wu, S. M. (2013). **Building a Large Annotated Corpus of Learner English: The NUS Corpus of Learner English.** Proceedings of the Eighth Workshop on Innovative Use of NLP for Building Educational Applications, 22–31.

- Nguồn: [ACL Anthology](https://aclanthology.org/W13-1703/).
- Trạng thái: đã trích trong proposal; tiếp tục dùng trong phần dữ liệu/preprocessing.
- Mục đích: nguồn gốc corpus và annotation. Bài giới thiệu corpus không thay thế điều khoản cấp quyền sử dụng dữ liệu.

### REF02 — CoNLL-2013 Shared Task

Ng, H. T., Wu, S. M., Wu, Y., Hadiwinoto, C., & Tetreault, J. (2013). **The CoNLL-2013 Shared Task on Grammatical Error Correction.** Proceedings of the Seventeenth Conference on Computational Natural Language Learning: Shared Task, 1–12.

- Nguồn: [ACL Anthology](https://aclanthology.org/W13-3601/).
- Trạng thái: đã trích trong proposal; tiếp tục dùng làm bối cảnh dữ liệu và nhóm lỗi.
- Giới hạn: shared task sửa lỗi không tương đương phân loại tại gold span của nhóm.

### REF03 — Tổng quan GEC

Bryant, C., Yuan, Z., Qorib, M. R., Cao, H., Ng, H. T., & Briscoe, T. (2023). **Grammatical Error Correction: A Survey of the State of the Art.** Computational Linguistics, 49(3), 643–701.

- Nguồn: [DOI/nhà xuất bản](https://doi.org/10.1162/coli_a_00478).
- Trạng thái: đã trích trong proposal.
- Mục đích: introduction và tổng quan lĩnh vực; phân biệt detection, classification, correction và evaluation.

### REF04 — Kiểm tra kiến thức cú pháp bằng phát hiện lỗi

Davis, C., Bryant, C., Caines, A., Rei, M., & Buttery, P. (2022). **Probing for targeted syntactic knowledge through grammatical error detection.** Proceedings of the 26th Conference on Computational Natural Language Learning, 360–373.

- Nguồn: [ACL Anthology](https://aclanthology.org/2022.conll-1.25/).
- Trạng thái: đã trích trong proposal; dự kiến dùng thêm khi thảo luận SVA và ngữ cảnh.
- Giới hạn: phát hiện lỗi làm diagnostic probe không phải phân loại năm nhãn tại span đã biết.

### REF05 — Prompting mô hình ngôn ngữ cho GEC

Davis, C., Caines, A., Andersen, Ø. E., Taslimipoor, S., Yannakoudakis, H., Yuan, Z., Bryant, C., Rei, M., & Buttery, P. (2024). **Prompting open-source and commercial language models for grammatical error correction of English learner text.** Findings of the Association for Computational Linguistics: ACL 2024, 11952–11967.

- Nguồn: [ACL Anthology](https://aclanthology.org/2024.findings-acl.711/).
- Trạng thái: đã trích trong proposal.
- Mục đích: bối cảnh mô hình ngôn ngữ cho GEC; không bổ sung thí nghiệm prompting LLM vào kế hoạch hai tuần.

### REF06 — EXPECT và giải thích lỗi

Fei, Y., Cui, L., Yang, S., Lam, W., Lan, Z., & Shi, S. (2023). **Enhancing Grammatical Error Correction Systems with Explanations.** Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 7489–7501.

- Nguồn: [ACL Anthology](https://aclanthology.org/2023.acl-long.413/).
- Trạng thái: đã trích trong proposal.
- Mục đích: động cơ giải thích lỗi và vai trò evidence words/error types; không coi điểm giải thích là macro-F1 phân loại của nhóm.

### REF07 — LET

Yang, L., Li, H., Li, L., Xu, C., Xia, S., & Yuan, C. (2023). **LET: Leveraging Error Type Information for Grammatical Error Correction.** Findings of the Association for Computational Linguistics: ACL 2023, 5986–5998.

- Nguồn: [ACL Anthology](https://aclanthology.org/2023.findings-acl.371/).
- Trạng thái: đã trích trong proposal, có trong sáu bài Excel; ưu tiên trong kế hoạch mới.
- Mục đích: related work về thông tin loại lỗi hỗ trợ GEC. Không triển khai LET; không so F0.5 sửa lỗi với macro-F1 phân loại.

### REF08 — MoECE

Qorib, M. R., Aji, A. F., & Ng, H. T. (2024). **Efficient and Interpretable Grammatical Error Correction with Mixture of Experts.** Findings of the Association for Computational Linguistics: EMNLP 2024.

- Nguồn: [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.997/).
- Trạng thái: đã dùng trong note.md, mã cũ R1.
- Mục đích: loại lỗi và khả năng diễn giải trong GEC; tham khảo tư tưởng ablation, không triển khai MoE.

### REF09 — GEE!

Song et al. (2024). **GEE! Grammar Error Explanation with Large Language Models.** Findings of the Association for Computational Linguistics: NAACL 2024.

- Nguồn và danh sách tác giả đầy đủ: [ACL Anthology](https://aclanthology.org/2024.findings-naacl.49/).
- Trạng thái: đã dùng trong note.md, mã cũ R2.
- Mục đích: phân biệt xử lý edit và sinh giải thích. Bài dùng cặp câu sai–đã sửa, không cùng đầu vào với classifier của nhóm.

### REF10 — Annotation phục vụ phản hồi học tập

Coyne et al. (2025). **Annotating Errors in English Learners’ Written Language Production: Advancing Automated Written Feedback Systems.** arXiv:2508.06810, bản tác giả.

- Nguồn và danh sách tác giả đầy đủ: [arXiv](https://arxiv.org/abs/2508.06810).
- Trạng thái: đã dùng trong note.md, mã cũ R3.
- Mục đích: ứng dụng nhãn/vị trí lỗi vào feedback; không coi thí nghiệm dùng oracle annotation là bằng chứng tự phân loại năm nhãn.

## B. Dự kiến ưu tiên dùng trong kế hoạch hai tuần

### REF11 — Mô hình ngữ cảnh nhận biết lỗi ngữ pháp

Nagata, R., Kimura, M., & Hanawa, K. (2022). **Exploring the Capacity of a Large-scale Masked Language Model to Recognize Grammatical Errors.** Findings of the Association for Computational Linguistics: ACL 2022, 4107–4118.

- Nguồn: [ACL Anthology](https://aclanthology.org/2022.findings-acl.324/).
- Trạng thái: đã review trong Excel; dự kiến dùng ở tuần 1 và phần methodology/discussion tuần 2.
- Mục đích: cơ sở thử BERT. Dữ liệu thật trong bài dùng phát hiện đúng/sai; phần giả lập đa loại không trùng năm nhãn NUCLE. Span-marked five-class classifier là điều chỉnh của nhóm.

### REF12 — Đánh giá theo hiện tượng cú pháp

Koyama, A., Mita, M., Yoon, S.-Y., Takama, Y., & Komachi, M. (2025). **Targeted Syntactic Evaluation for Grammatical Error Correction.** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 21108–21125.

- Nguồn: [ACL Anthology](https://aclanthology.org/2025.acl-long.1026/).
- Trạng thái: đã review trong Excel; dự kiến dùng cho evaluation và discussion.
- Mục đích: lập luận đánh giá theo hiện tượng, bên cạnh điểm tổng. Các tập con Vform/SVA, Nn/ArtOrDet của nhóm không phải tái lập CTSEG. Rà lại tính phù hợp với tiêu chí dữ liệu người học thực của review.

### REF13 — SERCL và phân loại lỗi cú pháp

Choshen, L., Nikolaev, D., Berzak, Y., & Abend, O. (2020). **Classifying Syntactic Errors in Learner Language.** Proceedings of the 24th Conference on Computational Natural Language Learning, 97–107.

- Nguồn: [ACL Anthology](https://aclanthology.org/2020.conll-1.7/).
- Trạng thái: đã review trong Excel; dự kiến dùng cho cơ sở đặc trưng và error analysis.
- Mục đích: thảo luận POS/morphology/dependency. Phương pháp gốc cần nguồn và bản sửa; không đưa correction vào feature của nhóm.

### REF14 — BERT gốc

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** Proceedings of NAACL-HLT 2019 (Volume 1: Long and Short Papers), 4171–4186.

- Nguồn: [ACL Anthology](https://aclanthology.org/N19-1423/).
- Trạng thái: bổ sung làm tài liệu phương pháp cho bước thử BERT trong kế hoạch.
- Mục đích: encoder hai chiều, pretraining và fine-tuning. Phải ghi riêng checkpoint/revision thực tế sử dụng. Bài gốc không xác nhận BERT sẽ hơn LR trên dữ liệu của nhóm.

### REF15 — Độ bền của parser trên câu sai ngữ pháp

Hashemi, H. B., & Hwa, R. (2016). **An Evaluation of Parser Robustness for Ungrammatical Sentences.** Proceedings of EMNLP 2016.

- Nguồn: [ACL Anthology](https://aclanthology.org/D16-1182/).
- Trạng thái: được đề xuất trong review; dự kiến dùng làm background ở phần limitations.
- Mục đích: giải thích vì sao cần audit parser trên learner text. Không suy ra tỷ lệ lỗi spaCy hiện tại từ kết quả parser khác.

## C. Dùng có điều kiện hoặc ưu tiên thấp

### REF16 — XGEC

Al-Sabahi, K. (2026). **XGEC: Explainable Grammatical Error Correction via Joint Sequence Tagging and Hierarchical Error Classification With Gated Cross-Task Attention.** IEEE Access. DOI: 10.1109/ACCESS.2026.3710856.

- Nguồn: [IEEE](https://ieeexplore.ieee.org/document/11598876); [bản tác giả đã đọc](https://www.researchgate.net/publication/408570984_XGEC_Explainable_Grammatical_Error_Correction_via_Joint_Sequence_Tagging_and_Hierarchical_Error_Classification_with_Gated_Cross-Task_Attention).
- Trạng thái: có trong sáu bài Excel; dự kiến dùng có điều kiện cho related work/future work. Chưa truy cập được trực tiếp IEEE trong lượt review.
- Giới hạn: phân loại phân cấp, nhãn ánh xạ và độ phủ khác bài toán nhóm. Không so trực tiếp accuracy với macro-F1; không triển khai XGEC trong hai tuần.

### REF17 — Classification model cho sửa lỗi

Zhou, S., & Liu, W. (2021). **English Grammar Error Correction Algorithm Based on Classification Model.** Complexity, article 6687337.

- Nguồn: [Nhà xuất bản/DOI](https://onlinelibrary.wiley.com/doi/full/10.1155/2021/6687337).
- Trạng thái: có trong sáu bài Excel; đã review, ưu tiên thấp; chưa chốt dùng trong report cuối.
- Mục đích nếu giữ: bối cảnh nghiên cứu. Protocol chưa đối chiếu rõ với phân loại tại gold span, không dùng làm căn cứ so điểm trực tiếp.

### REF18 — LR và SHAP cho phân loại lỗi câu

Liu, Y. (2026). **Interpretable Artificial Intelligence Used to Diagnose Common Grammatical Errors in English Writing by Non-Native Engineering Researcher.** Journal of Applied Science and Engineering, 34, article 26034065. DOI: 10.6180/jase.202611_34.065.

- Nguồn: [Toàn văn nhà xuất bản](https://jase.tku.edu.tw/jase/wp-content/uploads/2026/08/065_2026_1207_V34.pdf).
- Trạng thái: bài ngoài sáu bài được chọn, đã kiểm tra bổ sung; chỉ cân nhắc cho hướng diễn giải đặc trưng.
- Giới hạn: phân loại câu bảy lớp, có augmentation, nguồn dataset chưa đủ rõ để tái lập. Không chuyển nguyên bước bỏ stopwords/lemmatization vào pipeline cần giữ mạo từ, giới từ, hình thái và offset. Không thêm yêu cầu triển khai SHAP vào kế hoạch.

## D. Bài chỉ phục vụ sửa thông tin screening

### REF19 — Bài augmentation có URL bị ghi nhầm tiêu đề

Wan, Z., Wan, X., & Wang, W. (2020). **Improving Grammatical Error Correction with Data Augmentation by Editing Latent Representation.** Proceedings of the 28th International Conference on Computational Linguistics.

- Nguồn: [ACL Anthology](https://aclanthology.org/2020.coling-main.200/).
- Trạng thái: đối chiếu để sửa Eligibility dòng 30; chưa chọn làm tài liệu phương pháp hoặc triển khai augmentation.
- Cần làm: xác nhận bài nhóm định chọn, sửa cặp tiêu đề–URL, rồi đánh giá lại tiêu chí. Không coi việc sửa năm 2020 là tự động đủ điều kiện đưa bài vào review.

## E. Nguồn mô hình bổ sung khi thực hiện tuần 1

### REF20 — Pretrained BERT nhỏ dùng trong thực nghiệm tuần 1

Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **Well-Read Students Learn Better: On the Importance of Pre-training Compact Models.** arXiv:1908.08962.

- Nguồn: [bài gốc](https://arxiv.org/abs/1908.08962); [model card Google](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
- Trạng thái: dùng làm nguồn cho checkpoint pretrained BERT nhỏ của tuần 1, 2 layer/hidden 128/2 heads; không gọi là BERT-base. Fine-tune trực tiếp, không thực hiện distillation của bài.
- Revision và cấu hình thực tế: `docs/bert_experiment_protocol.md`; kết quả theo artifact `reports/contextual_model_validation/bert/`.

## F. Cách sử dụng danh mục

| Phần report/công việc | Tài liệu phù hợp |
|---|---|
| Nguồn dữ liệu, annotation, preprocessing | REF01, REF02 |
| Tổng quan và động cơ ứng dụng | REF03, REF05, REF06, REF08, REF09, REF10 |
| Loại lỗi và đặc trưng ngôn ngữ | REF07, REF13; REF16 có điều kiện |
| Thử BERT và thảo luận SVA | REF04, REF11, REF14 |
| Đánh giá nhóm khó, phân tích lỗi, hạn chế parser | REF12, REF13, REF15 |
| Rà soát lựa chọn bài trong Excel | REF17–REF19 và ghi chú tiêu chí ở REF11–REF13 |

- Chỉ đưa vào bibliography bản nộp các bài thực sự được trích trong nội dung. Danh mục làm việc này có thể rộng hơn bibliography cuối.
- Tài liệu nền tảng ngoài cửa sổ năm vẫn có thể được trích; không tự tính chúng vào số included studies của PRISMA.
- Với bài viết tắt tác giả bằng “et al.”, dùng metadata/BibTeX tại nguồn để xuất danh sách tác giả đầy đủ theo chuẩn trích dẫn của lớp.
- Không gọi mô hình ngoài là baseline đã chạy; không so điểm khi khác task, split, nhãn hoặc metric.
- Sửa nguồn NUCLE: URL đúng là W13-1703. URL P13-2060 từng xuất hiện trong review là bài khác và đã được sửa.
