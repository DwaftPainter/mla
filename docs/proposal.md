# Phân loại các dạng lỗi ngữ pháp trong bài viết tiếng Anh bằng Machine Learning

FINAL PROJECT PROPOSAL  •  Machine Learning Applications  •  Bản cập nhật ngày 04 tháng 10 năm 2026

## Tóm tắt đề xuất

Chúng tôi đề xuất nghiên cứu bài toán phân loại năm dạng lỗi ngữ pháp trong bài viết của người học tiếng Anh khi vị trí lỗi đã được xác định trước. Mỗi mẫu gồm câu gốc và một đoạn lỗi hoặc một vị trí thiếu từ; mô hình dự đoán ArtOrDet, Prep, Nn, Vform hoặc SVA. NUCLE là nguồn dữ liệu chính, với taxonomy được kiểm tra theo CoNLL-2013. Chúng tôi so sánh majority-class baseline, Multinomial Logistic Regression và Decision Tree trên các bài luận chưa xuất hiện khi huấn luyện.

Trọng tâm nghiên cứu là đo đóng góp của đặc trưng lexical, POS, morphology và dependency; kiểm tra mức phụ thuộc vào từ nằm trong đoạn lỗi; và giải thích các nhầm lẫn bằng bằng chứng từ dữ liệu. Chia tập theo bài luận, kiểm soát thông tin từ bản sửa và đánh giá bằng Macro-F1 là các yêu cầu cốt lõi. Kết quả dự kiến là một quy trình thực nghiệm có thể tái lập và các kết luận có giới hạn về khả năng của mô hình nhẹ. Proposal chưa báo cáo kết quả huấn luyện hay khẳng định cải thiện học tập.

## 1 Vấn đề và người sử dụng

Giáo viên cần biết người học thường gặp khó khăn ở chủ điểm ngữ pháp nào. Một câu đã sửa chưa trực tiếp cung cấp thống kê về dạng lỗi. Ví dụ, với “She go to university every day.” và đoạn “go” đã được đánh dấu, nhãn SVA giúp liên hệ lỗi với hòa hợp chủ ngữ và động từ. Việc tổng hợp nhãn có thể hỗ trợ giáo viên lựa chọn nội dung ôn tập; lợi ích này là động cơ ứng dụng, chưa phải hiệu quả giáo dục đã được dự án kiểm chứng.

Người dùng chính là giáo viên tiếng Anh; người dùng phụ là người học. Trong quy trình dự kiến, giáo viên cung cấp câu và vị trí lỗi, xem nhãn đề xuất rồi xác nhận hoặc điều chỉnh. Prototype chỉ minh họa chức năng này. Phạm vi không bao gồm phát hiện lỗi trên bài luận thô, sinh bản sửa, chấm điểm, đánh giá trình độ hoặc xây dựng hệ thống GEC đầu-cuối.

| Nhãn | Ý nghĩa |
| --- | --- |
| ArtOrDet | Mạo từ hoặc từ hạn định |
| Prep | Giới từ |
| Nn | Số ít và số nhiều của danh từ |
| Vform | Dạng động từ |
| SVA | Hòa hợp chủ ngữ và động từ |

## 2 Mục tiêu và câu hỏi nghiên cứu

Mục tiêu tổng quát là xác định khả năng phân biệt năm dạng lỗi bằng mô hình nhẹ và lượng thông tin ngôn ngữ cần thiết cho dự đoán trên bài luận mới. Các mục tiêu cụ thể là xây dựng dữ liệu theo từng annotation, thiết lập benchmark có kiểm soát, thực hiện ablation và phân tích thất bại.

| Câu hỏi | Thực nghiệm và bằng chứng |
| --- | --- |
| RQ1 | Logistic Regression và Decision Tree phân biệt năm nhãn tốt đến đâu? So sánh với majority baseline trên cùng tập chia và cùng biểu diễn compact. |
| RQ2 | Các nhóm lexical, POS, morphology và dependency ảnh hưởng thế nào đến Macro-F1 và F1 từng lớp? Thực hiện ablation tăng dần và bỏ từng nhóm; thêm thí nghiệm che đoạn lỗi. |
| RQ3 | Những nhóm lỗi nào khó phân biệt và đặc điểm nào liên quan đến nhầm lẫn? Phân tích confusion matrix, annotation và đầu ra parser trên mẫu dự đoán sai. |

### Các giả thuyết cần kiểm chứng

H1: Các mô hình học có thể vượt majority baseline về Macro-F1. H2: Morphology và dependency có thể cung cấp tín hiệu bổ sung cho SVA và Vform; lexical context có thể hữu ích cho Prep và ArtOrDet. H3: Bỏ lexical identity của đoạn lỗi có thể làm giảm hiệu suất ở một số lớp. Đây là giả thuyết thực nghiệm; kết quả không hỗ trợ giả thuyết vẫn là kết quả hợp lệ.

## 3 Định nghĩa bài toán và năm yếu tố Machine Learning

| Yếu tố | Định nghĩa trong dự án |
| --- | --- |
| Input | Vector đặc trưng từ câu gốc và span hoặc vị trí thiếu từ đã biết. |
| Output | Một trong năm nhãn mục tiêu. |
| Model | Multinomial Logistic Regression hoặc Decision Tree. |
| Prediction | Nhãn có score cao nhất; có thể kèm score hoặc phân phối xác suất của mô hình. |
| Training objective | Logistic Regression tối ưu log-loss đa lớp có regularization. Decision Tree chọn split giảm Gini impurity. Macro-F1 là metric đánh giá, không phải training loss. |

Định nghĩa hình thức: xᵢ = (câu gốc, start, end, đặc trưng suy ra); yᵢ ∈ {ArtOrDet, Prep, Nn, Vform, SVA}. Hàm fθ(xᵢ) dự đoán yᵢ trên các bài luận chưa thấy. Đây là supervised multiclass classification theo từng annotated error, không phải phân loại một nhãn cho toàn bộ câu có thể chứa nhiều lỗi.

## 4 Các nghiên cứu liên quan

### 4.1 Sửa lỗi ngữ pháp hiện đại

Sửa lỗi ngữ pháp tự động (Grammatical Error Correction, GEC) tập trung vào việc tự động phát hiện và sửa lỗi trong văn bản. Bryant et al. (2023) khảo sát sự phát triển của lĩnh vực từ phương pháp dựa trên luật và các bộ phân loại thống kê đến dịch máy thống kê và dịch máy neural. Khảo sát cho thấy nghiên cứu GEC hiện đại ngày càng chịu sự chi phối của các phương pháp neural hướng đến sửa lỗi, đồng thời ghi nhận các phương pháp phân loại và thiết kế đặc trưng ngôn ngữ trước đó.

Xu hướng này liên quan đến dự án vì nghiên cứu được đề xuất không nhằm xây dựng thêm một hệ thống sửa lỗi đầu-cuối. Thay vào đó, dự án tách riêng một bài toán dự đoán hẹp hơn: với một câu của người học và đoạn lỗi đã được xác định, phân loại đoạn đó vào một trong các nhóm lỗi ngữ pháp cố định của dự án.

### 4.2 Phát hiện lỗi và thông tin ngữ pháp cụ thể

Các nghiên cứu gần đây tiếp tục xem xét thông tin lỗi ngữ pháp độc lập với việc sửa toàn bộ câu. Davis et al. (2022) dùng phát hiện lỗi ngữ pháp như một phép thăm dò chẩn đoán để nghiên cứu liệu các mô hình ngôn ngữ tiền huấn luyện có mã hóa thông tin về hòa hợp chủ ngữ và động từ (SVA) hay không. Họ đánh giá biểu diễn của nhiều mô hình tiền huấn luyện, tập trung vào xác định sự hòa hợp là đúng hay sai. Kết quả cũng cho thấy hiệu suất thay đổi theo dữ liệu huấn luyện và cấu trúc cú pháp, gợi ý rằng thông tin ngữ pháp không phải lúc nào cũng được biểu diễn một cách vững chắc.

Nghiên cứu này liên quan vì SVA cũng là một nhóm lỗi của dự án. Tuy nhiên, Davis et al. tập trung vào một hiện tượng ngữ pháp và chủ yếu dùng phát hiện lỗi để thăm dò biểu diễn tiền huấn luyện. Dự án được đề xuất sẽ phân biệt nhiều nhóm lỗi trong cùng một thiết lập phân loại có giám sát.

### 4.3 Thông tin loại lỗi trong hệ thống GEC

Yang et al. (2023) đề xuất LET — Leveraging Error Type Information for Grammatical Error Correction. Mô hình trước tiên xử lý câu đầu vào bằng một module phân loại để dự đoán loại lỗi cho từng token. Thông tin loại lỗi dự đoán được đưa vào đầu vào decoder và cơ chế cross-attention nhằm cải thiện việc sửa lỗi ngữ pháp.

LET cung cấp bằng chứng rõ ràng rằng dự đoán loại lỗi tường minh đã được nghiên cứu trong GEC hiện đại. Tuy nhiên, dự đoán loại lỗi chủ yếu đóng vai trò thông tin hỗ trợ quá trình sửa lỗi cuối cùng.

Dự án đề xuất sử dụng cách định nghĩa bài toán khác: giả định đoạn lỗi đã biết và coi phân loại loại lỗi là nhiệm vụ dự đoán cuối cùng. Dự án không có kế hoạch sinh câu đã sửa.

### 4.4 GEC có giải thích và phản hồi giáo dục

Fei et al. (2023) giới thiệu EXPECT, một dataset được gán nhãn loại lỗi ngữ pháp và các từ cung cấp bằng chứng phục vụ GEC có giải thích. Nghiên cứu xem xét cách đưa thông tin lỗi tường minh vào lời giải thích của hệ thống. Đánh giá bởi con người do tác giả báo cáo cho thấy các lời giải thích có thể giúp người học ngôn ngữ thứ hai quyết định có chấp nhận gợi ý sửa lỗi hay không và hiểu quy tắc ngữ pháp tương ứng.

Công trình này hỗ trợ động cơ giáo dục của dự án: thông tin lỗi ngữ pháp tường minh có thể cung cấp phản hồi có cấu trúc hơn so với chỉ đưa ra văn bản đã sửa.

Tuy nhiên, EXPECT tập trung vào sửa lỗi có giải thích, kết hợp thông tin loại lỗi với xác định bằng chứng và giải thích bản sửa. Dự án đề xuất có phạm vi hẹp hơn, không sinh lời giải thích hoặc bản sửa. Thay vào đó, dự án nghiên cứu liệu các bộ phân loại nhẹ có thể phân biệt đáng tin cậy các nhóm lỗi định sẵn hay không.

### 4.5 GEC bằng LLM và đánh giá theo loại lỗi

Davis et al. (2024) đánh giá bảy mô hình ngôn ngữ mã nguồn mở và ba mô hình thương mại trên bốn benchmark GEC tiếng Anh đã được thiết lập. Ngoài đánh giá GEC tổng thể, họ báo cáo hiệu suất theo từng loại lỗi, cho thấy hành vi mô hình có thể khác biệt đáng kể giữa các nhóm ngữ pháp.

Công trình này cho thấy tầm quan trọng của việc đánh giá theo nhóm lỗi thay vì chỉ dựa vào một điểm số sửa lỗi tổng thể. Tuy nhiên, nhiệm vụ của họ vẫn là GEC đầu-cuối dựa trên prompting các mô hình ngôn ngữ lớn.

Dự án đề xuất dùng các mô hình nhẹ có giám sát và đánh giá trực tiếp phân loại loại lỗi. Precision, recall, F1 theo lớp và các mẫu nhầm lẫn sẽ được dùng để tìm hiểu nhóm ngữ pháp nào dễ hoặc khó phân biệt hơn đối với các bộ phân loại.

## 5 Khoảng trống nghiên cứu

Các công trình từ năm 2022 cho thấy nhiều hướng quan trọng trong nghiên cứu lỗi ngữ pháp. Phát hiện lỗi được dùng để nghiên cứu hiện tượng cú pháp cụ thể như SVA (Davis et al., 2022). Dự đoán loại lỗi được tích hợp vào hệ thống GEC neural (Yang et al., 2023). Loại lỗi cũng được dùng trong GEC có giải thích và phản hồi hướng đến người học (Fei et al., 2023), trong khi các nghiên cứu LLM gần đây đánh giá khả năng sửa lỗi theo từng nhóm ngữ pháp (Davis et al., 2024). Bryant et al. (2023) cung cấp bối cảnh chung và mô tả sự chuyển dịch từ các phương pháp phân loại trước đây sang hệ thống sửa lỗi neural hiện đại.

Vì vậy, dự án không khẳng định phân loại lỗi ngữ pháp, đặc trưng ngôn ngữ, thông tin loại lỗi hoặc đánh giá theo loại lỗi là các ý tưởng mới.

Thay vào đó, dự án tập trung vào một câu hỏi thực nghiệm hẹp hơn:

**Các bộ phân loại nhẹ và có thể diễn giải phân biệt một tập nhóm lỗi ngữ pháp cố định hiệu quả đến đâu khi đoạn lỗi đã biết, và những nhóm đặc trưng ngôn ngữ tường minh nào có vẻ đóng góp vào dự đoán của chúng?**

Trong phạm vi các tài liệu gần đây được rà soát cho proposal này, thiết lập có kiểm soát cụ thể nói trên nhận được tương đối ít sự chú ý trực tiếp hơn. Những nghiên cứu đã rà soát chủ yếu tập trung vào phát hiện lỗi, dùng loại lỗi để cải thiện việc sửa lỗi, giải thích bản sửa hoặc đánh giá hệ thống GEC đầu-cuối theo nhóm lỗi.

Do đó, dự án sẽ tách riêng phân loại loại lỗi thành nhiệm vụ dự đoán chính. Với một câu và đoạn lỗi đã được xác định, các mô hình dự kiến sẽ dự đoán một trong các nhóm lỗi mục tiêu cố định.

Các thí nghiệm dự kiến so sánh majority-class baseline, Logistic Regression và Decision Tree. Nghiên cứu cũng xem xét các nhóm đặc trưng lexical, POS, morphology và syntax/dependency thông qua feature ablation có kiểm soát, đồng thời đánh giá bằng Macro-F1, precision/recall/F1 theo lớp và phân tích nhầm lẫn.

Đây là các thí nghiệm dự kiến, chưa phải kết quả hiện có. Ở giai đoạn proposal và tìm hiểu dữ liệu, dự án không giả định mô hình nào sẽ vượt mô hình khác, nhóm đặc trưng nào sẽ chứa nhiều thông tin nhất hoặc các bộ phân loại sẽ đạt hiệu suất cao.

Đóng góp dự kiến vì vậy không phải taxonomy lỗi ngữ pháp mới hoặc thuật toán Machine Learning mới. Dự án đề xuất một nghiên cứu thực nghiệm có kiểm soát về phân loại loại lỗi bằng mô hình nhẹ trong thiết lập đoạn lỗi đã biết, chú trọng đóng góp của đặc trưng ngôn ngữ có thể diễn giải và hành vi mô hình theo từng lớp.

## 6 Kế hoạch dữ liệu và kiểm soát chất lượng

NUCLE do NUS xây dựng gồm khoảng 1.400 bài luận, hơn một triệu từ, với annotation của giảng viên tiếng Anh; báo cáo corpus ban đầu mô tả 1.414 bài luận (Dahlmeier et al., 2013). Những con số này mô tả nguồn tham khảo, không phải số mẫu sau xử lý của dự án. Chúng tôi sẽ ghi nhận release thực tế, điều kiện sử dụng, checksum và thống kê trước và sau lọc. Việc có tài liệu corpus không đồng nghĩa dự án đã được cấp quyền truy cập.

Năm nhóm lỗi mục tiêu là các nhóm đã được xác lập trong CoNLL-2013 (Ng et al., 2013), không phải taxonomy mới do dự án đề xuất.

### Taxonomy và schema

Ưu tiên sử dụng release có annotation tương thích CoNLL-2013. Nếu dùng raw NUCLE, chúng tôi kiểm tra taxonomy và quy trình chuyển đổi chính thức, đặc biệt đối với Prep và các nhóm nguồn như Wcip hoặc Rloc. Không đổi toàn bộ một nhóm nguồn thành Prep bằng ánh xạ một-một tùy ý. Correction chỉ có thể phục vụ chuyển đổi annotation theo quy trình đã xác minh và kiểm tra chất lượng; không đi vào feature extractor hoặc đầu vào suy luận.

Mỗi hàng gồm essay_id, sentence_id, annotation_id, sentence, error_start, error_end, error_span, error_type, span_kind và các cờ chất lượng. Lưu original_label và nguồn mapping trong bảng provenance riêng. Correction được lưu tách biệt cho audit và qualitative analysis. Xác định rõ offsets theo ký tự hay token, theo paragraph hay sentence, trước khi chuyển đổi. Schema này là kế hoạch cho giai đoạn Data Preparation sau. Từ 06/10/2026, code hiện tại chỉ triển khai Data Understanding và Data Quality: giữ bảng annotation với offset theo paragraph, thống kê và gắn cờ kiểm tra; chưa tạo mẫu theo câu hoặc feature.

### Lỗi thiếu từ và nhiều lỗi trong một câu

Span có start = end biểu thị một vị trí cần chèn từ và vẫn là mẫu hợp lệ. Lưu span_kind = insertion, dùng token bên trái và bên phải làm ngữ cảnh, đặt POS và morphology của span là NOT_APPLICABLE. Không dùng từ được chèn trong bản sửa để tạo đặc trưng. Với lỗi thay thế hoặc xóa, căn chỉnh span với các token gốc; span nhiều token dùng tập hợp hoặc thống kê đặc trưng với quy tắc cố định.

Một câu có nhiều annotation tạo nhiều mẫu; mỗi mẫu chỉ đánh dấu annotation mục tiêu. Không cung cấp nhãn hoặc vị trí các annotation khác cho model. Span chồng lấn có thể biểu diễn hai lỗi riêng, annotation nghi trùng hoặc cùng vùng được gán nhãn khác nhau. Giữ mỗi annotation thành một sample với đúng một nhãn; không tự động gộp, đổi nhãn hoặc loại vì chồng lấn/cùng span. Gắn cờ và truy vết annotation để review; chỉ sửa/loại sau quyết định có bằng chứng, lý do và nhật ký theo annotation_id. Offset không hợp lệ hoặc span qua nhiều câu/đoạn được audit riêng theo khả năng dựng mẫu, không đánh đồng với overlap.

### Quy trình kiểm tra dữ liệu

Kiểm tra missing fields, offset bounds, tính đúng của chuỗi trích xuất, phân bố nhãn, độ dài câu và span, tỷ lệ insertion, số lỗi mỗi bài và parser failures. Audit thủ công tối thiểu 20 mẫu mỗi lớp nếu đủ dữ liệu, bao gồm insertion và span nhiều token. Kiểm tra duplicate bài luận và câu trước chia tập. Các bài có câu/đoạn trùng đáng kể được nối thành nhóm chia chung hoặc xử lý bằng quy tắc loại trùng đã ghi nhận. Data Quality hiện chỉ báo cáo ứng viên trùng essay/paragraph theo exact và NFC + casefold + gộp whitespace; không nối bài thành nhóm hoặc chia fold. Chính sách grouping, boilerplate và near-duplicate cần được chốt ở giai đoạn chuẩn bị sau; xem README cho hai bước đang chạy.

Chỉ văn bản gốc được tokenization, POS tagging, morphology và dependency parsing bằng một NLP pipeline cố định. Giữ nguyên văn bản để bảo toàn offsets; lowercase chỉ áp dụng trong lexical representation. Không xóa stopwords vì article và preposition là tín hiệu cần nghiên cứu. Cache đầu ra parser và ghi nhận phiên bản để tái lập.

## 7 Pipeline đặc trưng và mô hình

Pipeline gồm: xác minh quyền và release → parse annotation → kiểm tra offsets và taxonomy → tạo mẫu theo lỗi → xử lý trùng và chia nhóm → trích đặc trưng từ câu gốc → fit preprocessing trên train → huấn luyện và chọn cấu hình bằng validation → khóa protocol → đánh giá test → phân tích lỗi và viết báo cáo.

| Nhóm | Biểu diễn dự kiến |
| --- | --- |
| L Lexical | Token trong span và cửa sổ ±2 token; n-grams cục bộ. Danh mục compact được học trên train với giới hạn 500 mục; từ còn lại là UNK. TF-IDF là nhánh bổ sung cho Logistic Regression. |
| P POS | POS của token mục tiêu và hai phía, local POS patterns; missing và NOT_APPLICABLE được biểu diễn rõ. |
| M Morphology | Number, Person, Tense, VerbForm của span; tổng hợp cố định cho span nhiều token và missing indicators. |
| D Dependency | Quan hệ phụ thuộc, POS của head, khoảng cách phụ thuộc, sự hiện diện của chủ ngữ và thuộc tính chủ ngữ khi parser tìm được. Insertion dùng thông tin hai token kề. |
| S Structural | Độ dài câu, độ dài span, vị trí tương đối, span_kind. S được giữ cố định để tránh nhầm đóng góp của lexical và cấu trúc. |

Tất cả nhóm feature phải khả dụng từ câu gốc và span tại inference. Không dùng corrected sentence, correction token, original error tag, essay_id, annotation_id hoặc độ dài correction làm predictor. IDs chỉ dùng cho grouping và truy vết. Vocabulary, TF-IDF IDF, one-hot categories, imputation và scaling chỉ được fit bằng train; validation và test chỉ transform.

### Logistic Regression

Dùng multinomial Logistic Regression với L2 regularization. Khảo sát C ∈ {0.01, 0.1, 1, 10}, class_weight ∈ {None, balanced}; tăng giới hạn iteration khi cần và kiểm tra convergence. Phân tích coefficient theo lớp để tìm association giữa đặc trưng và dự đoán. Không diễn giải coefficient như nguyên nhân ngữ pháp hoặc tác động nhân quả.

### Decision Tree và so sánh công bằng

Decision Tree dùng Gini, max_depth ∈ {4, 8, 12}, min_samples_leaf ∈ {5, 20}, class_weight ∈ {None, balanced}; ghi seed cố định. Giới hạn depth và kích thước lá giúp kiểm soát overfitting. Xem cây rút gọn và decision paths; feature importance chỉ là mô tả hành vi model và có thể thiên lệch.

Benchmark chính dùng cùng feature matrix compact và cùng partitions cho hai model. Nhánh TF-IDF dành cho Logistic Regression được báo cáo riêng là thử nghiệm biểu diễn phù hợp với mô hình, vì chênh lệch kết quả ở nhánh này không chỉ đến từ thuật toán. Majority baseline luôn dự đoán lớp phổ biến nhất trong train. Có thể thêm baseline stratified ngẫu nhiên như kiểm tra bổ sung, không thay majority baseline.

## 8 Thiết kế thực nghiệm và đánh giá

### Chia tập và chọn cấu hình

Dự kiến 70% train, 15% validation và 15% test theo essay_id hoặc nhóm bài liên kết bởi duplicate. Nếu có author_id đáng tin cậy, ưu tiên nhóm theo tác giả. Chọn split dựa trên tính hợp lệ và đủ hỗ trợ năm lớp, không dựa trên performance. Ghi seed và manifest; đóng băng test trước tuning. Không có bài hoặc nhóm nội dung trùng đáng kể theo chính sách đã khóa đi qua hai partitions; báo cáo riêng trường hợp trùng ngắn/boilerplate được miễn và giới hạn near-duplicate.

Chọn hyperparameters trên validation bằng Macro-F1. Với mỗi ablation, tune lại trong cùng grid và cùng ngân sách để so sánh cấu hình đã được chọn hợp lý. Khóa tất cả biến thể sẽ báo cáo trước khi đánh giá test. Sau đó chạy một đợt đánh giá test cho các cấu hình đã khóa; không dùng kết quả để sửa feature hoặc chọn lại model. Giữ nguyên train đã fit để báo cáo chính, tránh thay đổi protocol bằng refit tùy ý.

| Thí nghiệm | Thiết kế | Câu hỏi |
| --- | --- | --- |
| E1 | Majority, Logistic Regression và Decision Tree trên S+L+P+M+D compact. | RQ1 |
| E2 | S+L; thêm P; thêm M; thêm D. Bổ sung full trừ P, M hoặc D để kiểm tra đóng góp phụ thuộc thứ tự. | RQ2 |
| E3 | Full; che lexical span; S+P+M+D không lexical. Giữ cùng mẫu, partitions và protocol. | RQ2 |
| E4 | So sánh thay đổi metric do feature groups giữa hai model; LR TF-IDF được tách riêng. | RQ1 và RQ2 |
| E5 | Confusion matrix và audit các dự đoán sai theo nhãn, span_kind và parser evidence. | RQ3 |

### Định nghĩa thí nghiệm che đoạn lỗi

Trong điều kiện masked, parser vẫn xử lý câu gốc rồi loại lexical identity, lemma và mọi n-gram chứa token mục tiêu khỏi lexical features; thay bằng một marker cố định. Giữ POS, morphology và dependency đã suy ra từ câu gốc để đo riêng phần lexical identity. Không parse lại câu chứa &lt;ERR&gt;, vì thay đổi parsing sẽ gây nhiễu phép so sánh. Structural-only loại lexical cả span và context. Với insertion không có lexical span để che, vì vậy báo cáo riêng insertion và non-empty spans.

Giảm F1 sau masking là bằng chứng model phụ thuộc vào lexical span trong protocol này; chưa đủ chứng minh model thiếu hiểu biết ngữ pháp. POS hoặc morphology vẫn có thể gián tiếp phản ánh danh tính token. Kết luận phải mô tả rõ thông tin đã được giữ lại.

### Metrics và độ bất định

Metric chính: Macro-F1 = (F1_ArtOrDet + F1_Prep + F1_Nn + F1_Vform + F1_SVA) / 5. Báo cáo thêm accuracy, macro precision và recall, precision/recall/F1/support từng lớp, confusion matrix dạng đếm và chuẩn hóa theo nhãn thật. Luôn sử dụng đủ năm nhãn khi tính metric; xử lý zero-division bằng quy tắc nhất quán.

Ước lượng khoảng tin cậy 95% bằng 1.000 lần bootstrap theo nhóm bài luận, dùng cùng resampled groups khi tính chênh lệch hai cấu hình. Không bootstrap độc lập từng annotation vì các lỗi trong một bài có liên hệ. Ghi nhận train time, inference time và số đặc trưng trên cùng phần cứng như chỉ báo chi phí; không khẳng định hiệu quả thời gian nếu chưa đo.

## 9 Phân tích lỗi đầu ra và lịch thực hiện

Chọn các cặp nhầm lẫn phổ biến từ confusion matrix thực tế. Audit dự kiến tối đa 50 mẫu sai, phân bổ theo lớp và cặp nhầm, cùng 10 mẫu đúng để đối chiếu. Chỉ dùng correction và nhãn gốc sau dự đoán cho việc đọc và kiểm tra; chúng không được đưa trở lại model. Mỗi trường hợp lưu câu, span, nhãn thật, nhãn dự đoán, đặc trưng liên quan, parser evidence và nhận xét có mức độ chắc chắn.

Các giả thuyết giải thích gồm class imbalance, ngữ cảnh không đủ, overlap giữa loại lỗi, span alignment và parser noise. Parser trả được cây không đồng nghĩa cây đúng: báo cáo cả tỷ lệ lỗi kỹ thuật và chất lượng parsing quan sát trong audit. Nếu nghi nhãn gốc mơ hồ, ghi nhận sự mơ hồ; không tự sửa ground truth dựa trên dự đoán của model.

### Đầu ra và tiêu chí nghiệm thu

| Đầu ra | Tiêu chí hoàn thành |
| --- | --- |
| Dữ liệu và provenance | Release, license, mapping, counts trước và sau lọc, audit offsets và bảng chất lượng được ghi nhận. |
| Code và tái lập | Parser, splitter, feature extractor, training/evaluation scripts; seeds, split manifest, phiên bản môi trường và cấu hình được lưu. |
| Benchmark và ablation | E1–E4 có bảng metric đầy đủ, cùng protocol, số đặc trưng và diễn giải chênh lệch có giới hạn. |
| Error analysis | E5 có mẫu truy vết, confusion pairs và phân biệt bằng chứng với giả thuyết. |
| Báo cáo và trình bày | Trả lời RQ1–RQ3, bàn luận giới hạn, tài liệu tham khảo và dàn ý sáu slide. |
| Prototype tùy chọn | Nhập câu và span hoặc insertion position; hiển thị nhãn dự kiến và yêu cầu giáo viên kiểm tra. |

Thành công nghiên cứu được định nghĩa bằng chất lượng quy trình và bằng chứng trả lời RQ, không đặt một ngưỡng Macro-F1 tùy ý trước khi hiểu dữ liệu. Nếu model không vượt baseline, dự án báo cáo kết quả đó cùng phân tích nguyên nhân. Không sử dụng accuracy cao làm chứng minh model có ích cho giảng dạy.

### Kế hoạch thực hiện dự kiến trong sáu tuần

| Tuần | Công việc và mốc kiểm tra |
| --- | --- |
| 1 | Xác minh truy cập, release và taxonomy; chốt phạm vi nếu dữ liệu không đáp ứng. |
| 2 | Parser, offset audit, phân bố dữ liệu và group split; đóng băng manifest. |
| 3 | Feature compact, baseline, Logistic Regression và Decision Tree; kiểm tra leakage và convergence. |
| 4 | Tuning validation; E2–E4; khóa toàn bộ cấu hình đánh giá. |
| 5 | Chạy test, bootstrap, confusion matrix và qualitative audit. |
| 6 | Viết báo cáo, đóng gói code, hoàn thiện sáu slide; prototype nếu còn thời gian. |

Đây là lịch tương đối đề xuất, chưa phải deadline của học phần. NLP parsing sẽ được cache; mô hình cổ điển chạy trên CPU, không yêu cầu GPU. Dependency và morphology không được tự ý bỏ khỏi câu hỏi nghiên cứu khi thiếu thời gian; mọi thu hẹp thực nghiệm phải được ghi nhận trong báo cáo.

## 10 Rủi ro phương án dự phòng và sử dụng có trách nhiệm

| Rủi ro | Biện pháp và phương án dự phòng |
| --- | --- |
| Không truy cập được NUCLE | Giải quyết điều kiện cấp quyền trong tuần 1. Chỉ đổi corpus sau khi kiểm tra annotation, năm nhãn và group IDs; nếu đổi taxonomy, sửa RQ và công khai khả năng so sánh. |
| Mapping không xác minh được | Ưu tiên release CoNLL-compatible; không đoán mapping. Chỉ dùng subset đã xác minh và báo cáo coverage, hoặc điều chỉnh phạm vi. |
| Thiếu lớp hoặc imbalance lớn | Dùng class weighting và per-class metrics. Nếu không đủ dữ liệu để giữ năm lớp, sửa phạm vi thay vì sinh thêm annotation giả cho test. |
| Leakage | Group split; kiểm tra duplicate; fit transformations trên train; tách correction và provenance khỏi predictors. |
| Parser noise và span không hợp lệ | Missing indicators, audit mẫu và parser cache; báo cáo tỷ lệ loại mẫu từng lớp để đánh giá selection bias. |
| Lexical shortcut | Thực hiện masking và structural-only; kết luận mức phụ thuộc vào tín hiệu quan sát được. |
| Tree overfitting hoặc kết quả yếu | Giới hạn độ sâu và lá; tune bằng validation; giữ nguyên test và báo cáo kết quả âm. |
| Vượt phạm vi hoặc thiếu thời gian | Ưu tiên benchmark compact, ablation, masking và error analysis. TF-IDF bổ sung và prototype là hạng mục có thể hoãn. |

### Giới hạn suy rộng

Kết quả chỉ áp dụng cho các annotation và năm nhãn trong release được dùng, trên các bài luận cùng nguồn phân phối. NUCLE không đại diện cho mọi trình độ, ngôn ngữ mẹ đẻ hoặc người học Việt Nam. Biết trước span làm bài toán dễ hơn xử lý văn bản thô; không dùng kết quả này để khẳng định khả năng phát hiện lỗi. Năm nhãn không bao quát mọi lỗi và model luôn phải chọn một nhãn khi nhận input ngoài phạm vi, nên người sử dụng cần xác nhận input thuộc taxonomy này.

### Sử dụng có trách nhiệm

Output là gợi ý cho giáo viên kiểm tra. Score xác suất của Logistic Regression hoặc tỷ lệ nhãn ở lá Decision Tree chưa được coi là confidence đã hiệu chỉnh; prototype phải ghi rõ đây là score mô hình. Không suy ra trình độ người học từ một span, không tự động chấm điểm hoặc xếp hạng. Chỉ công bố văn bản, annotation và processed dataset khi license cho phép; mặc định chia sẻ code, thống kê tổng hợp và hướng dẫn tái tạo thay vì corpus.

### Dàn ý trình bày sáu slide

1. Vấn đề, giáo viên sử dụng và known-span task. 2. NUCLE, năm nhãn và mapping. 3. Related work, khoảng trống thực nghiệm và RQ1–RQ3. 4. Năm yếu tố ML, hai model và bốn nhóm feature. 5. Essay split, Macro-F1, ablation và masking. 6. Đóng góp dự kiến, rủi ro, deliverables và giới hạn sử dụng.

## 11 Tài liệu tham khảo

Bryant, C., Yuan, Z., Qorib, M. R., Cao, H., Ng, H. T., & Briscoe, T. (2023). Grammatical Error Correction: A Survey of the State of the Art. Computational Linguistics, 49(3), 643–701. https://doi.org/10.1162/coli_a_00478

Dahlmeier, D., Ng, H. T., & Wu, S. M. (2013). Building a Large Annotated Corpus of Learner English: The NUS Corpus of Learner English. Proceedings of the Eighth Workshop on Innovative Use of NLP for Building Educational Applications, 22–31. https://aclanthology.org/W13-1703/

Davis, C., Bryant, C., Caines, A., Rei, M., & Buttery, P. (2022). Probing for targeted syntactic knowledge through grammatical error detection. Proceedings of the 26th Conference on Computational Natural Language Learning, 360–373. https://doi.org/10.18653/v1/2022.conll-1.25

Davis, C., Caines, A., Andersen, Ø. E., Taslimipoor, S., Yannakoudakis, H., Yuan, Z., Bryant, C., Rei, M., & Buttery, P. (2024). Prompting open-source and commercial language models for grammatical error correction of English learner text. Findings of the Association for Computational Linguistics: ACL 2024, 11952–11967. https://doi.org/10.18653/v1/2024.findings-acl.711

Fei, Y., Cui, L., Yang, S., Lam, W., Lan, Z., & Shi, S. (2023). Enhancing Grammatical Error Correction Systems with Explanations. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, Volume 1, 7489–7501. https://doi.org/10.18653/v1/2023.acl-long.413

Ng, H. T., Wu, S. M., Wu, Y., Hadiwinoto, C., & Tetreault, J. (2013). The CoNLL-2013 Shared Task on Grammatical Error Correction. Proceedings of the Seventeenth Conference on Computational Natural Language Learning: Shared Task, 1–12. https://aclanthology.org/W13-3601/

Yang, L., Li, H., Li, L., Xu, C., Xia, S., & Yuan, C. (2023). LET: Leveraging Error Type Information for Grammatical Error Correction. Findings of the Association for Computational Linguistics: ACL 2023, 5986–5998. https://doi.org/10.18653/v1/2023.findings-acl.371
