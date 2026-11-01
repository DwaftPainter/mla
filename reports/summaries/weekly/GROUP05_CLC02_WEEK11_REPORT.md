# BÁO CÁO WEEK 11

**Lớp:** CLC02 · **Nhóm:** 05  
**Đề tài:** Automatic Classification of Common Grammatical Error Types for ESL Learners Using Machine Learning  
**Ngày cập nhật:** 01/11/2026

## 1. Công việc đã thực hiện

Báo cáo tuần này bổ sung kết quả BERT-small mở rộng và phần phân tích lỗi sau báo cáo Week 10. Các số liệu huấn luyện, validation và test được lấy từ đợt chạy đã lưu, trong đó bản phân tích BERT mở rộng được lập ngày 28/10/2026. Lần cập nhật ngày 01/11 tập trung đối chiếu số liệu, hoàn thiện nhận xét từng annotation và chỉnh lại tài liệu; không có lượt huấn luyện hoặc chấm test mới.

Những phần đã hoàn thành gồm:

- Tổng hợp điểm train và validation của 12 lượt BERT, so sánh BERT với LR trên các nhóm validation đã dùng ở báo cáo trước, và bổ sung Precision, Recall, F1 theo từng lớp trên test.
- Điền nhận xét và contribution cho 18 annotation validation; bổ sung bản sửa gốc và annotation liên quan để làm rõ những ca khó đọc khi xét riêng target.
- Hoàn thiện 40 review BERT/LR và 22 nhận định xử lý trong danh sách cần xem lại. Mỗi ca có phần giải thích, căn cứ và hướng xử lý cụ thể.
- Cập nhật báo cáo Word, Markdown, bản tổng hợp audit và mục lục tài liệu trên repo. Các file chứa câu nguồn vẫn được giữ ở máy nghiên cứu.

Phần khảo sát tài liệu, các bảng ablation cũ và chẩn đoán gradient/memorization đã trình bày ở Week 10 không được báo cáo lại.

## 2. Kết quả BERT-small mở rộng

### 2.1. Cấu hình thực tế đã chạy

Đợt mở rộng so sánh hai cách lấy biểu diễn: vector CLS và trung bình vector của các token thuộc span lỗi, gọi là span mean. Learning rate của encoder giữ ở 5e-5; phần phân loại dùng 1e-4 hoặc 1e-3. Mỗi tổ hợp chạy ba seed 20261013, 20261014 và 20261015, tối đa 20 epoch, với patience bằng 4. Checkpoint của mỗi lượt được chọn theo macro-F1 validation; sau đó chọn tổ hợp có điểm validation trung bình cao nhất qua ba seed.

Log không có mức learning rate 3e-4. Số epoch và cách đặt learning rate cũng chưa trùng hoàn toàn với gợi ý thử 1e-4, 3e-4, 1e-3 trong 5–10 epoch. Train và validation được đánh giá sau mỗi epoch; ngưỡng train macro-F1 trên 0,70 chưa được dùng làm điều kiện chặn trước khi xem validation. Vì vậy, báo cáo ghi nhận kết quả của thiết lập đã chạy, không coi đây là việc thực hiện đầy đủ mọi bước được đề nghị.

### 2.2. Điểm train và validation

Điểm train và validation trong mỗi lượt được lấy ở cùng checkpoint đã chọn. Bảng dưới là trung bình ba seed; dấu ± là độ lệch chuẩn mẫu giữa các seed.

| Biểu diễn | LR encoder | LR phần phân loại | Train macro-F1 | Validation macro-F1 | Epoch được chọn, theo thứ tự ba seed |
| --- | --- | --- | --- | --- | --- |
| CLS | 5e-5 | 1e-4 | 0,8108 | 0,7164 ± 0,0061 | 20 / 19 / 15 |
| CLS | 5e-5 | 1e-3 | 0,8458 | 0,7227 ± 0,0147 | 20 / 13 / 17 |
| Span mean | 5e-5 | 1e-4 | 0,9278 | 0,8183 ± 0,0011 | 18 / 14 / 14 |
| Span mean | 5e-5 | 1e-3 | 0,9225 | 0,8177 ± 0,0022 | 8 / 10 / 9 |

Cả 12 checkpoint được chọn đều có train macro-F1 trên 0,70. Với cấu hình span mean và learning rate phần phân loại 1e-4, seed chính 20261013 đạt train macro-F1 **0,9309** và validation **0,8173** ở epoch 18. Kết quả này cho thấy tình trạng mô hình gần như chưa học được trên train ở lần thử đầu đã được khắc phục trong đợt chạy mở rộng.

Span mean cho điểm validation cao hơn CLS ở cả hai mức learning rate. Tuy vậy, cấu hình được chọn vẫn có chênh lệch train–validation khoảng 0,11. Mô hình đã học được tập train, nhưng khả năng tổng quát hóa còn hạn chế. Do đợt mở rộng thay đổi cả cách lấy biểu diễn, learning rate và thời lượng huấn luyện so với lần đầu, chưa thể quy toàn bộ mức tăng cho riêng span mean.

## 3. Đối chiếu trên các nhóm validation khó

BERT được so sánh với LR trên đúng các tập con Vform/SVA và Nn/ArtOrDet đã dùng ở Week 10. Các dòng VERB, AUX, NOUN và DET lọc theo POS signature trùng chính xác. Prediction vẫn giữ năm lớp; F1 của mỗi cặp là trung bình F1 của hai nhãn sau khi lọc theo gold.

| Nhóm validation | Số mẫu | LR Full | BERT seed chính | BERT trung bình 3 seed |
| --- | --- | --- | --- | --- |
| Vform / SVA | 445 | 0,7417 | 0,7695 | 0,7726 ± 0,0061 |
| Vform / SVA, signature VERB | 243 | 0,7594 | 0,7582 | 0,7616 ± 0,0049 |
| Vform / SVA, signature AUX | 98 | 0,7216 | 0,7347 | 0,7250 ± 0,0086 |
| Nn / ArtOrDet | 1.561 | 0,8869 | 0,8909 | 0,8907 ± 0,0034 |
| Nn / ArtOrDet, signature NOUN | 631 | 0,8445 | 0,8591 | 0,8655 ± 0,0083 |
| Nn / ArtOrDet, signature DET | 353 | 0,5524 | 0,4891 | 0,5102 ± 0,0365 |

BERT seed chính tăng khoảng 0,0277 trên toàn cặp Vform/SVA, nhưng gần như ngang LR khi chỉ xét signature VERB. Với Nn/ArtOrDet, mức tăng chung nhỏ; riêng nhóm DET giảm khoảng 0,0632. Kết quả cho thấy lợi ích của BERT khác nhau giữa các nhóm lỗi, chưa thể nói mô hình giải quyết tốt mọi trường hợp hai nhãn dùng chung POS.

Độ biến động giữa seed ở nhóm DET cũng lớn hơn các nhóm còn lại. Bảng này mô tả những khác biệt quan sát được; chưa có kiểm định riêng cho từng nhóm để kết luận ý nghĩa thống kê.

## 4. Tổng hợp kết quả test

### 4.1. Thứ tự thực hiện

Hồ sơ khóa mô hình cổ điển ghi thời điểm 08:39:33 UTC ngày 23/10/2026; bản ghi bắt đầu chấm test là 08:39:42 UTC cùng ngày. Phân tích BERT mở rộng được lập ngày 28/10, sau khi kết quả test cổ điển đã có. Như vậy, phép so sánh dưới đây dùng lại cùng tập test và chưa phải một lần đánh giá độc lập trên dữ liệu mới.

BERT có hash cho manifest, lựa chọn cấu hình và checkpoint; bản ghi bắt đầu chấm test tham chiếu hash lựa chọn. File lựa chọn không có timestamp độc lập, nên báo cáo không suy thêm giờ khóa từ thời gian sửa file. Đây cũng là phần cập nhật trạng thái thực tế so với mô tả test chưa được mở ở Week 10.

### 4.2. Kết quả trên cùng 2.368 annotation

Bảng giữ riêng Tree S+L+P+M và Tree Full để trình bày đúng các cấu hình cần đối chiếu. Các giá trị được tổng hợp từ kết quả đã lưu.

| Hệ thống | Validation macro-F1 | Test macro-F1 | Test accuracy | CI 95% của test macro-F1 |
| --- | --- | --- | --- | --- |
| Majority | 0,1184 | 0,1184 | 0,4206 | [0,1104; 0,1279] |
| POS rule | 0,7048 | 0,7106 | 0,7441 | [0,6766; 0,7365] |
| LR Full | 0,8067 | **0,8163** | 0,8357 | [0,7876; 0,8382] |
| Tree S+L+P+M | 0,7929 | 0,7967 | 0,8121 | [0,7674; 0,8201] |
| Tree Full | 0,7879 | 0,7996 | 0,8095 | [0,7693; 0,8246] |
| BERT-small, seed chính | **0,8173** | 0,8146 | **0,8438** | [0,7919; 0,8350] |

S, L, P, M và D là các nhóm đặc trưng cấu trúc câu/span, lexical, POS, morphology và dependency; Full dùng cả năm nhóm. BERT seed chính là 20261013, dùng span mean, LR encoder 5e-5 và LR phần phân loại 1e-4.

LR Full nhỉnh hơn BERT seed chính về macro-F1, còn BERT cao hơn về accuracy. Chênh lệch macro-F1 BERT − LR là **−0,0017**, với CI paired 95% **[−0,0184; 0,0182]**. Khoảng này chứa 0, nên dữ liệu hiện có chưa đủ để kết luận mô hình nào tốt hơn về macro-F1.

Ba seed BERT đạt test macro-F1 lần lượt 0,8146; 0,8177; 0,8222, trung bình **0,8182 ± 0,0038**. Đây là thống kê giữa seed, không phải điểm ensemble.

Các CI đã lưu được tính bằng 1.000 lượt bootstrap theo 97 nhóm liên kết trong test. Chúng không bao gồm biến động do đổi seed huấn luyện hoặc chia lại dữ liệu.

### 4.3. Precision, Recall và F1 theo từng lớp

Bảng dưới đối chiếu LR Full với BERT seed chính 20261013 trên cùng tập test. P là Precision, R là Recall và N là số annotation theo gold. Precision cho biết tỷ lệ đúng trong những ca được gán một nhãn; Recall cho biết tỷ lệ tìm được trong toàn bộ ca thực sự thuộc nhãn đó. F1 kết hợp hai chỉ số này.

| Nhãn | N | LR P | LR R | LR F1 | BERT P | BERT R | BERT F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ArtOrDet | 996 | 0,8366 | 0,9046 | 0,8693 | 0,8522 | 0,9147 | 0,8823 |
| Prep | 362 | 0,9239 | 0,9061 | 0,9149 | 0,9155 | 0,9282 | 0,9218 |
| Nn | 565 | 0,8184 | 0,7416 | 0,7781 | 0,8652 | 0,7611 | 0,8098 |
| Vform | 216 | 0,7387 | 0,6806 | 0,7084 | 0,7085 | 0,7315 | 0,7198 |
| SVA | 229 | 0,8178 | 0,8035 | 0,8106 | 0,7689 | 0,7118 | 0,7392 |
| Macro | 2.368 | 0,8271 | 0,8073 | 0,8163 | 0,8221 | 0,8094 | 0,8146 |

Các giá trị được làm tròn bốn chữ số thập phân sau khi tính. Dòng Macro là trung bình đều giữa năm lớp cho từng chỉ số; N ở dòng này là tổng số mẫu. Macro-F1 được tính bằng trung bình năm F1, không lấy F1 từ hai giá trị macro-Precision và macro-Recall.

BERT có F1 cao hơn LR ở bốn lớp ArtOrDet, Prep, Nn và Vform. Tuy nhiên, F1 của SVA giảm khoảng 0,0713. Trong kết quả này, mức giảm ở SVA lớn hơn tổng mức tăng F1 của bốn lớp còn lại, nên macro-F1 của BERT vẫn thấp hơn khoảng 0,0017. Ở Vform cũng có sự đánh đổi: BERT tăng Recall nhưng giảm Precision, dù F1 cuối cùng tăng nhẹ.

### 4.4. SVA: chênh lệch thể hiện ở cả Precision và Recall

Với SVA, Recall của BERT thấp hơn LR khoảng **9,17 điểm phần trăm**, còn Precision thấp hơn khoảng **4,89 điểm phần trăm**. Các số đếm từ prediction đã lưu giúp đọc rõ hơn chênh lệch này:

- Trong **229 ca gold SVA**, LR nhận diện đúng **184 ca**, bỏ sót **45 ca**; BERT nhận diện đúng **163 ca**, bỏ sót **66 ca**. BERT có ít hơn 21 ca nhận diện đúng SVA.
- LR gán nhãn SVA cho **225 ca**, trong đó **41 ca thuộc nhãn khác**. BERT gán SVA cho **212 ca**, trong đó **49 ca thuộc nhãn khác**. Như vậy, dù số dự đoán SVA ít hơn, BERT vẫn có nhiều ca gán nhầm nhãn này hơn.
- BERT nhầm **35 ca SVA sang Vform**, so với **18 ca** của LR. Ở chiều ngược lại, BERT nhầm **34 ca Vform sang SVA**, so với **29 ca** của LR.

Khoảng hụt F1 ở SVA vì thế đi cùng cả việc bỏ sót nhiều hơn và gán nhầm nhiều hơn, trong đó nhầm lẫn với Vform là một hướng đáng chú ý. Các số đếm này mô tả lỗi quan sát được trên test; chúng chưa chứng minh nguyên nhân nằm ở attention, cách biểu diễn span hoặc khả năng dùng ngữ cảnh của BERT.

Trong khi số ca SVA đúng giảm 21, tổng số ca đúng ở bốn lớp còn lại tăng 40. BERT vì vậy đúng thêm 19 ca trên toàn test, giải thích việc accuracy cao hơn nhưng macro-F1 thấp hơn LR. Phần bổ sung này chỉ tính lại chỉ số từ prediction đã lưu, không chạy lại mô hình.

## 5. Hoàn thiện audit 18 annotation validation

### 5.1. Mục đích và cách chọn mẫu

Từ prediction validation đã lưu, LR Full đúng **2.004/2.368 ca** và sai **364 ca**, tương đương **15,37%**. Sai ở đây là dự đoán không khớp gold, chưa có nghĩa annotation gốc sai.

Phần audit đọc lại câu, span, bản sửa và parse để hiểu loại nhầm lẫn. Trước hết lấy ba ID đầu sau khi sắp xếp ở mỗi chiều nhầm sau:

| Chiều nhầm: gold → prediction | Tổng số ca sai | Chọn ở bước đầu |
| --- | --- | --- |
| Vform → SVA | 40 | 3 |
| SVA → Vform | 35 | 3 |
| Nn → ArtOrDet | 94 | 3 |
| ArtOrDet → Nn | 44 | 3 |

Sau 12 ca này, thêm sáu ca sai có span nhiều token, thành 18 ca, chiếm khoảng **4,95% số dự đoán sai**. Mẫu được chọn có chủ đích để tập trung vào hai cặp nhãn dễ nhầm và các span phức tạp. Con số 18 không đến từ tính toán cỡ mẫu thống kê, nên không dùng tỷ lệ trong mẫu để ước lượng nguyên nhân phổ biến nhất trên toàn validation.

### 5.2. Nội dung đã bổ sung và kết quả đọc ca

File audit hiện có đủ nhận xét và contribution cho 18 dòng. Mỗi nhận xét ghi thao tác sửa quan sát được, cách giải thích và điểm còn cần xem lại. Bản sửa gốc, ID annotation liên quan và người phụ trách cũng đã được bổ sung. Nhãn, prediction và câu gốc giữ nguyên.

Đối chiếu với SGML cho thấy cả 18 span và gold khớp nguồn. Trong parse đã lưu, 17 ca có alignment exact; ca 11696 có dấu cách cuối span nên được ghi là whitespace_edge.

Đọc thêm annotation lân cận giúp làm rõ một số trường hợp:

- **15390:** bản sửa động từ sang ngôi thứ ba số ít đi cùng annotation 15389 xóa động từ tình thái và các sửa cấu trúc ở 15388, 15391. Xét cả nhóm sửa thì nhãn SVA có cơ sở hơn khi chỉ đọc riêng target.
- **19068:** bản sửa vừa thay trợ động từ vừa sửa phân từ; annotation 19067 còn chuyển chủ ngữ sang số nhiều. Vì vậy, trong cùng một span có cả tín hiệu SVA và Vform.
- **14939:** cấu trúc đứng trước target được sửa bởi 14938. Cần đọc hai annotation cùng nhau; tính nhất quán giữa nhãn SVA và toàn bộ nhóm sửa vẫn là điểm cần xem lại.

Các ca này cho thấy một nhãn có thể phải đại diện cho nhiều thao tác, hoặc chỉ hợp lý khi xét cùng bản sửa khác. Ca 12019 còn cho thấy mô hình có thể sai dù parse chứa thông tin hòa hợp phù hợp. Ở ca 11696, POS có dấu hiệu cần kiểm tra lại. Những quan sát đó chưa chứng minh parser gây ra prediction sai; chưa có phép thay parse rồi chạy đối chứng.

### 5.3. Phân công cho 18 ca audit

| Thành viên | Số ca | Annotation ID | Phạm vi phụ trách |
| --- | --- | --- | --- |
| Quân | 6 | 10824, 11696, 12352, 11501, 12050, 12301 | Nn/ArtOrDet: số danh từ, countability, từ hạn định, POS và ranh giới span. |
| Duy | 6 | 12314, 14761, 14764, 12019, 14939, 15390 | Vform/SVA: thì, tình thái, hòa hợp chủ–vị và các bản sửa lân cận. |
| Công | 6 | 1269, 18774, 19065, 19068, 19069, 19185 | Span dài hoặc nhiều thay đổi: xác định phần sửa chính và các điểm cần thống nhất nhãn. |

Bảng này ghi phân công của phần audit 18 annotation. Các nhận xét trong file đã hoàn thiện; bảng không được dùng để suy ra đóng góp của từng người cho các phần khác của project.

## 6. Hoàn thiện 40 review BERT/LR và 22 nhận định xử lý

### 6.1. Bộ 40 ca BERT/LR

Phần này sử dụng prediction test đã lưu của BERT seed chính và LR Full. Mẫu được chọn sau khi có prediction, trước khi đọc câu và correction. Nhóm BERT sai/LR đúng ưu tiên sáu ca gold SVA.

| Nhóm | Số ca | Nội dung đã hoàn thành |
| --- | --- | --- |
| BERT sai, LR đúng | 12 | Đối chiếu thao tác sửa, nhãn và tín hiệu ngữ pháp trong từng ca. |
| BERT đúng, LR sai | 12 | Đọc ranh giới span, cấu trúc liên quan và điểm khác biệt với LR. |
| Cả hai cùng sai | 10 | Ghi nhận lỗi chung, bản sửa nhiều thành phần và nghi vấn nhãn. |
| Cả hai cùng đúng | 6 | Đọc ca đối chứng, gồm span một phần token và correction cần kiểm tra. |

Cả **40 ca đã có review và contribution**, kèm căn cứ và mức tin cậy định tính. Contribution ở đây nêu mỗi ca đóng góp gì cho việc giải thích kết quả, không phải phân công nhân sự.

Bộ mẫu gồm 24 ca hai mô hình bất đồng, 10 ca cùng sai và sáu ca cùng đúng. Riêng nhóm cùng sai có chín ArtOrDet và một Nn do cách duyệt các nhóm chọn mẫu; phân bố này không đại diện cho toàn bộ test.

### 6.2. Kết quả 22 nhận định xử lý

Danh sách cần xem lại gộp 17 ID từ audit mô hình cổ điển và sáu ID ưu tiên của audit BERT. ID 34539 trùng giữa hai đợt, nên còn 22 ID duy nhất. Tám ID cũng nằm trong bộ 40 ca; vì vậy lượt đối chiếu mở rộng bao phủ **54 annotation khác nhau**.

Cả **22 ID đã có nhận định xử lý và căn cứ**. Các nhóm dưới đây là kết quả phân tích, chưa phải các nhãn đã được sửa trong dataset.

| Hướng xử lý đề xuất | Số ca | Annotation ID |
| --- | --- | --- |
| Có cơ sở giữ nhãn theo bản sửa và ngữ cảnh | 9 | 18531, 37631, 24526, 9351, 41235, 35455, 211, 37624, 32801 |
| Cần xem lại sự phù hợp của nhãn | 6 | 34539, 12324, 26635, 32705, 37909, 16809 |
| Nghi vấn correction hoặc phạm vi thay thế | 3 | 18501, 23037, 30366 |
| Chỉnh sửa có thể tùy chọn | 3 | 201, 11916, 6079 |
| Bản sửa gồm nhiều thao tác | 1 | 32874 |

Hai ca được làm rõ nhờ đọc thêm nguồn là **32801** và **18531**. Với 32801, các annotation 32799–32800 bỏ mạo từ và chuyển danh từ đầy đủ sang số nhiều, giải thích được việc sửa số của từ viết tắt. Với 18531, các sửa ở 18529–18530 giúp xác định cấu trúc và quan hệ hòa hợp của động từ target.

Ngược lại, **23037** vẫn có nghi vấn về phạm vi thay thế: span chỉ là ký tự cuối nhưng replacement chứa cả từ. **18501** và **30366** cần xem lại cấu trúc sau khi áp correction. Các ca này đã có hướng kiểm tra cụ thể, nhưng chưa được sửa nguồn.

Các nhận định cũng làm rõ sự khác nhau giữa khớp gold và chất lượng annotation. Chẳng hạn, cả hai mô hình khớp gold ở 30366 nhưng correction vẫn cần kiểm tra. Một dự đoán đúng loại lỗi không tự xác nhận rằng bản sửa là đúng hoặc bắt buộc.

Tại thời điểm báo cáo, đã hoàn thành phần phân tích và đề xuất xử lý; **chưa có lượt thẩm định độc lập của chuyên gia**. Sáu ca nghi vấn nhãn, ba ca nghi vấn correction/span và các ca còn phụ thuộc cách đọc được giữ nguyên ghi chú, không tự coi là đã giải quyết xong vấn đề dữ liệu.

## 7. Hoàn thiện tài liệu và repository

Báo cáo Week 11 đã được chuẩn hóa thành bản Word và Markdown có cùng nội dung. Các bảng được đối chiếu với kết quả đã lưu; các file review được kiểm tra về số dòng, nội dung còn thiếu, ID và trạng thái. Bản tổng hợp audit, báo cáo BERT và các mục lục liên quan đã được cập nhật.

Tài liệu đã được cập nhật trên hai repo. Week 11 chỉ đặt trên repo **DwaftPainter/mla**; repo **MihQuannn7799/esl-grammar-error-classification** giữ các báo cáo kết quả của project bằng tiếng Anh Markdown, không chứa báo cáo tuần này. Các file có câu nguồn, correction đầy đủ và prediction từng sample được giữ nội bộ; thư mục private đã được bổ sung vào quy tắc bỏ qua của Git.

Việc chỉnh tài liệu và hoàn thiện review không làm thay đổi code mô hình, checkpoint, gold hay các bảng điểm. Bản ghi 61/61 kiểm tra trong hồ sơ BERT là kết quả đã lưu từ đợt phân tích trước, không phải một lượt kiểm thử mới trong lần biên tập này.

## 8. Nhận xét sau khi hoàn thành

Kết quả mới cho thấy BERT-small đã học được trên train và đạt validation quanh 0,818. Trên test đang có, BERT seed chính và LR Full vẫn gần nhau về macro-F1; kết quả theo lớp và theo nhóm POS còn có những đánh đổi rõ ràng. Chưa có cơ sở để kết luận BERT tốt hơn LR trong toàn bộ bài toán.

Phần audit đã có đủ nhận xét cho 18 ca validation, 40 ca BERT/LR và nhận định xử lý cho 22 ID cần xem lại. Kết quả giúp giải thích rõ hơn các trường hợp span rộng, span một phần token, nhãn chồng lấn và bản sửa phụ thuộc annotation khác. Các nhận xét này tạo căn cứ cho việc kiểm tra dữ liệu hoặc thiết kế thí nghiệm tiếp theo, chưa xác định nguyên nhân nhân quả của prediction.

Những giới hạn còn lại gồm: chưa có test mới độc lập, chưa thử can thiệp thay parse để kiểm tra tác động lên mô hình, và chưa có thẩm định chuyên gia độc lập cho các ca còn nghi vấn. Gold, correction và offset nguồn chưa được thay đổi sau review. Phần đã hoàn thành là phân tích, ghi căn cứ và hoàn thiện báo cáo; việc sửa dataset cần được xử lý riêng.

## Nguồn số liệu và hồ sơ đối chiếu

Hai tài liệu đầu vào là GROUP05_CLC02_WEEK10_REPORT.docx và Nhan_xet_Week10_CLC02_Group05.docx. Các nguồn dùng trong báo cáo:

- Cấu hình, checkpoint và điểm train/validation: reports/bert_small_experiment/manifest.json, selection.json và 12 file summary.json của từng lượt chạy.
- Nhóm validation khó: prediction đã lưu của POS, LR Full và ba seed BERT span mean.
- Kết quả test, từng lớp và CI: reports/evaluation_followup/reviewed_overall_results.csv; reports/bert_final_analysis/overall_results.csv, per_class_metrics.csv, confusions.csv, paired_differences.csv và summary.json. Bảng Precision/Recall/F1 và các số đếm SVA đã được đối chiếu lại với prediction test đã lưu của LR Full và BERT seed chính.
- Mốc khóa: reports/final_evaluation/lock.json và test_scoring_started.json; reports/bert_small_experiment/selection.sha256 và test/test_started.json.
- Audit: file CSV 18 annotation, hai bộ packet đã lưu, SGML gốc, các file review nội bộ và reports/bert_final_analysis/review_summary.md.

Hash khóa cổ điển: `2f2f9fd1ff19805255f300d4eaf646aa964e3e643ce692027f528ba5b9c198ee`. Hash lựa chọn BERT: `f61405c7f5cc5d1576488a112bd2b2b411cc6d4789d96dfc0bf3f86cbd304582`.

Hồ sơ phân tích ngày 28/10 còn ghi nhận thiếu bốn file mã nguồn so với khóa cổ điển và hai CSV đã đổi định dạng. Số liệu có thể đối chiếu được không đồng nghĩa toàn bộ workspace đã khớp byte với khóa lịch sử. Lần biên tập báo cáo này không chạy lại quy trình tái lập đầy đủ.
