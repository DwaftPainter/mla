# BÁO CÁO WEEK 11

**Lớp:** CLC02 · **Nhóm:** 05  
**Đề tài:** Automatic Classification of Common Grammatical Error Types for ESL Learners Using Machine Learning  
**Ngày cập nhật:** 01/11/2026

## 1. Những việc bổ sung sau báo cáo Week 10

Vướng mắc lớn nhất ở báo cáo trước là BERT-small chưa học được ngay trên tập train. Vì vậy, điểm validation thấp khi đó chưa đủ để nhận xét về khả năng dùng ngữ cảnh của mô hình. Báo cáo tuần này bổ sung kết quả đợt huấn luyện mở rộng, đối chiếu BERT với LR trên các nhóm khó đã xác định, và tổng hợp kết quả test cùng các mốc khóa thí nghiệm.

Phần audit 18 annotation đã được hoàn thiện và toàn bộ 18 nhận xét đã được rà soát và chốt ngày 01/11/2026. File làm việc có nhận xét cho từng ca, bản sửa gốc, annotation liên quan và người phụ trách. Báo cáo chỉ trình bày những điểm làm rõ thêm, không lặp lại toàn bộ 18 nhận xét đã có trong Week 10. Phần contribution bên dưới chỉ dành cho việc audit này.

Bản cập nhật này cũng hoàn tất nội dung phân tích cho 40 ca BERT/LR và nhận định xử lý cho cả 22 ID trong hàng đợi cần xem lại. Nhóm đã rà soát và chốt toàn bộ kết quả ngày 01/11/2026; bộ 18 ca validation cũng đã được rà soát và chốt riêng.

Các kết quả BERT bổ sung được lưu trong đợt phân tích ngày 28/10/2026. Lần cập nhật báo cáo này sử dụng kết quả và prediction đã lưu, không huấn luyện lại hoặc chấm lại test. Các bảng khảo sát tài liệu, ablation cũ và chẩn đoán gradient/memorization được giữ ở báo cáo trước.

## 2. BERT-small đã học được trên train

### 2.1. Thay đổi trong đợt chạy mở rộng

Đợt chạy bổ sung so sánh hai cách lấy biểu diễn: vector CLS của câu và trung bình các vector token thuộc span lỗi (span mean). Learning rate của encoder giữ ở 5e-5; phần phân loại dùng 1e-4 hoặc 1e-3. Mỗi tổ hợp chạy với ba seed 20261013, 20261014 và 20261015, tối đa 20 epoch, patience bằng 4. Cấu hình được chọn bằng macro-F1 validation trung bình qua ba seed; trong mỗi lần chạy, checkpoint được chọn theo validation.

Đây là cấu hình thực tế trong log. Nó chưa trùng hoàn toàn với gợi ý thử 1e-4, 3e-4, 1e-3 trong 5–10 epoch: chưa có mức 3e-4 và encoder không dùng ba mức learning rate này. Train và validation được đánh giá cùng mỗi epoch; điều kiện train macro-F1 trên 0,70 chưa được cài thành bước chặn trước khi đọc validation. Vì vậy, kết quả dưới đây chứng minh mô hình đã học được trên train, nhưng không được xem là đã thực hiện nguyên xi quy trình được đề nghị.

### 2.2. Kết quả train và validation

Mỗi dòng là trung bình ba seed. Train và validation được lấy ở **cùng checkpoint được chọn** của từng seed; dấu ± là độ lệch chuẩn mẫu qua ba seed, không phải khoảng tin cậy.

| Biểu diễn | LR encoder | LR phần phân loại | Train macro-F1 | Validation macro-F1 | Epoch được chọn, theo thứ tự ba seed |
| --- | --- | --- | --- | --- | --- |
| CLS | 5e-5 | 1e-4 | 0,8108 | 0,7164 ± 0,0061 | 20 / 19 / 15 |
| CLS | 5e-5 | 1e-3 | 0,8458 | 0,7227 ± 0,0147 | 20 / 13 / 17 |
| Span mean | 5e-5 | 1e-4 | 0,9278 | 0,8183 ± 0,0011 | 18 / 14 / 14 |
| Span mean | 5e-5 | 1e-3 | 0,9225 | 0,8177 ± 0,0022 | 8 / 10 / 9 |

Cả 12 lần chạy đều có train macro-F1 trên 0,70 tại checkpoint được chọn. Với span mean và LR phần phân loại 1e-4, seed chính 20261013 đạt train macro-F1 **0,9309**, validation **0,8173**, ở epoch 18. Như vậy, vấn đề train macro-F1 quanh 0,22 ở lần thử trước đã được khắc phục trong đợt chạy này.

Span mean đạt validation tốt hơn CLS ở cả hai mức learning rate của phần phân loại. Kết quả gợi ý rằng biểu diễn gắn trực tiếp với vùng lỗi phù hợp hơn cho bài toán phân loại annotation trong thiết lập đang dùng. Tuy nhiên, chênh lệch train–validation của cấu hình được chọn còn khoảng 0,11; mô hình học được trên train không có nghĩa là đã giải quyết hết vấn đề tổng quát hóa. Do cách lấy biểu diễn, learning rate và số epoch đều được thay đổi so với lần thử đầu, chưa thể quy toàn bộ mức tăng cho riêng span mean.

## 3. BERT có cải thiện trên những nhóm khó không?

Để trả lời phần nhận xét về Vform/SVA và Nn/ArtOrDet, BERT được đối chiếu trên đúng các tập con validation đã dùng ở Week 10. Các dòng VERB, AUX, NOUN và DET dùng **POS signature trùng chính xác**, không thay bằng POS đại diện. Chỉ lọc theo hai nhãn thật của từng cặp; prediction vẫn giữ đủ năm lớp. F1 trong bảng là trung bình F1 của hai nhãn sau khi lọc.

| Nhóm validation | Số mẫu | LR Full | BERT seed chính | BERT trung bình 3 seed |
| --- | --- | --- | --- | --- |
| Vform / SVA | 445 | 0,7417 | 0,7695 | 0,7726 ± 0,0061 |
| Vform / SVA, signature VERB | 243 | 0,7594 | 0,7582 | 0,7616 ± 0,0049 |
| Vform / SVA, signature AUX | 98 | 0,7216 | 0,7347 | 0,7250 ± 0,0086 |
| Nn / ArtOrDet | 1.561 | 0,8869 | 0,8909 | 0,8907 ± 0,0034 |
| Nn / ArtOrDet, signature NOUN | 631 | 0,8445 | 0,8591 | 0,8655 ± 0,0083 |
| Nn / ArtOrDet, signature DET | 353 | 0,5524 | 0,4891 | 0,5102 ± 0,0365 |

Điểm mới rõ nhất là BERT seed chính tăng khoảng 0,0277 trên toàn cặp Vform/SVA. Tuy nhiên, khi chỉ xét signature VERB, điểm gần như ngang LR. Với cặp Nn/ArtOrDet, mức tăng chung nhỏ; riêng nhóm DET lại giảm khoảng 0,0632. Vì vậy, kết quả không ủng hộ nhận định BERT sẽ tự động giải quyết mọi trường hợp hai nhãn dùng chung POS.

Các nhóm nhỏ cũng biến động khác nhau giữa seed, đặc biệt là DET. Bảng này dùng để mô tả nơi mô hình cải thiện hoặc còn yếu; chưa có kiểm định riêng cho các nhóm để kết luận chênh lệch có ý nghĩa thống kê.

## 4. Kết quả test và giới hạn của phép so sánh

### 4.1. Mốc khóa và thứ tự thực hiện

Khi đối chiếu lại hồ sơ chạy, các mô hình cổ điển đã được khóa lúc **08:39:33 UTC ngày 23/10/2026**; bản ghi bắt đầu chấm test là **08:39:42 UTC cùng ngày**. Đợt BERT mở rộng được tổng hợp ngày **28/10/2026**, sau khi kết quả test của các mô hình cổ điển đã có. Đây là trạng thái cập nhật theo log, thay cho mô tả test chưa được mở ở bản Week 10.

Hồ sơ BERT lưu hash manifest, lựa chọn cấu hình và checkpoint; bản ghi bắt đầu chấm test tham chiếu hash lựa chọn đó. File lựa chọn không có timestamp độc lập, nên không ghi thêm một giờ khóa BERT không có trong dữ liệu. Bộ kiểm tra đã lưu tại `reports/bert_final_analysis/verification.json` ghi nhận 61/61 kiểm tra đạt.

Hai hash dùng để đối chiếu:

- Khóa các mô hình cổ điển: `2f2f9fd1ff19805255f300d4eaf646aa964e3e643ce692027f528ba5b9c198ee`.
- Lựa chọn BERT: `f61405c7f5cc5d1576488a112bd2b2b411cc6d4789d96dfc0bf3f86cbd304582`.

Vì BERT được bổ sung sau lần đánh giá test của mô hình cổ điển, phép so sánh này có giới hạn về thứ tự thực hiện và chưa có một test mới độc lập. Báo cáo không coi đây là thí nghiệm mà toàn bộ lựa chọn của cả hai họ mô hình đều được khóa trước lần đầu nhìn thấy bất kỳ kết quả test nào.

### 4.2. Bảng kết quả tổng hợp

Tất cả các dòng dùng cùng 2.368 annotation test. Tree S+L+P+M được đưa vào đúng theo cấu hình cần báo cáo; Tree Full được giữ thêm để đối chiếu, không thay thế Tree S+L+P+M bằng cấu hình có điểm test cao hơn. Hồ sơ gốc có nhiều biến thể cổ điển hơn; bảng này trích các hệ thống liên quan trực tiếp đến phần nhận xét, không phải một lượt chấm test mới chỉ gồm sáu hệ thống.

| Hệ thống | Validation macro-F1 | Test macro-F1 | Test accuracy | CI 95% của test macro-F1 |
| --- | --- | --- | --- | --- |
| Majority | 0,1184 | 0,1184 | 0,4206 | [0,1104; 0,1279] |
| POS rule | 0,7048 | 0,7106 | 0,7441 | [0,6766; 0,7365] |
| LR Full | 0,8067 | **0,8163** | 0,8357 | [0,7876; 0,8382] |
| Tree S+L+P+M | 0,7929 | 0,7967 | 0,8121 | [0,7674; 0,8201] |
| Tree Full | 0,7879 | 0,7996 | 0,8095 | [0,7693; 0,8246] |
| BERT-small, seed chính | **0,8173** | 0,8146 | **0,8438** | [0,7919; 0,8350] |

S, L, P, M và D lần lượt là các nhóm đặc trưng cấu trúc câu/span, lexical, POS, morphology và dependency; Full dùng cả năm nhóm. BERT seed chính dùng span mean, LR encoder 5e-5 và LR phần phân loại 1e-4, seed 20261013. Seed chính được cố định, không chọn lại bằng điểm test.

LR Full nhỉnh hơn BERT seed chính về macro-F1, còn BERT cao hơn về accuracy. Chênh lệch macro-F1 BERT − LR là **−0,0017**, với khoảng tin cậy paired 95% **[−0,0184; 0,0182]**. Khoảng này chứa 0, nên chưa đủ bằng chứng để kết luận một mô hình tốt hơn mô hình còn lại về macro-F1. CI được lấy từ 1.000 lượt bootstrap theo 97 nhóm liên kết trong test; đây là khoảng cho từng phép so sánh, không bao gồm biến động do chia lại dữ liệu hoặc huấn luyện với seed khác.

Ba seed của cấu hình BERT được chọn đạt test macro-F1 lần lượt 0,8146; 0,8177; 0,8222, trung bình **0,8182 ± 0,0038**. Đây là thống kê độ biến động giữa seed, không phải điểm của một mô hình ensemble. Ở từng lớp, BERT tốt hơn LR trên Nn (0,8098 so với 0,7781), nhưng thấp hơn trên SVA (0,7392 so với 0,8106). Việc accuracy tăng vì thế chưa đồng nghĩa mọi loại lỗi đều được nhận diện tốt hơn.

## 5. Hoàn thiện audit 18 annotation

### 5.1. Mục đích audit và cách chọn 18 annotation

Trên tập validation gồm **2.368 annotation**, LR Full dự đoán đúng **2.004 ca** và sai **364 ca**, tương đương tỷ lệ sai **15,37%**. Đây là số dự đoán không khớp nhãn gold, chưa phải số annotation được xác định là gán nhãn sai. Các số đếm được đối chiếu từ prediction validation đã lưu của LR Full.

Việc đọc kỹ từng ca nhằm giải thích những nhầm lẫn mà điểm macro-F1 chưa cho biết: mô hình có bỏ sót tín hiệu ngữ pháp không, parse có phù hợp không, span có bao nhiều thao tác sửa không và các nhãn có ranh giới rõ ràng không. Phân tích này cung cấp bằng chứng cho cách giải thích kết quả và hình thành giả thuyết cải tiến; bản thân việc đọc ca chưa chứng minh được nguyên nhân gây lỗi.

Mẫu 18 annotation được chọn có chủ đích. Trước hết, lấy ba ca ở mỗi chiều nhầm của hai cặp Vform/SVA và Nn/ArtOrDet; trong từng chiều, lấy ba ID đầu sau khi sắp xếp. Số ca trong toàn bộ validation và số được lấy ở bước đầu như sau:

| Chiều nhầm: gold → prediction | Tổng số ca sai | Chọn ở bước đầu |
| --- | --- | --- |
| Vform → SVA | 40 | 3 |
| SVA → Vform | 35 | 3 |
| Nn → ArtOrDet | 94 | 3 |
| ArtOrDet → Nn | 44 | 3 |

Sau 12 ca này, bổ sung sáu ca sai khác có span nhiều token để xem thêm những annotation chứa cụm dài hoặc nhiều thay đổi. Tổng cộng có **18 ca**, chiếm khoảng **4,95% trong 364 ca sai**. Hai cặp nhãn trên là trọng tâm phân tích; sáu ca bổ sung giúp mở rộng việc quan sát sang các span phức tạp. Bảng trên mô tả bước chọn 12 ca đầu, không phải phân bố nhãn của toàn bộ mẫu 18 ca. Việc chia sáu ca cho mỗi thành viên được thực hiện trên mẫu đã chọn này.

Con số 18 là quy mô của phần đọc định tính, không xuất phát từ tính toán cỡ mẫu thống kê. Vì mẫu không ngẫu nhiên và chưa bao phủ hết các nhóm lỗi, chưa thể dùng tỷ lệ trong mẫu để ước lượng nguyên nhân nào phổ biến nhất trên toàn bộ validation. Muốn trả lời câu hỏi đó cần mở rộng audit hoặc lấy mẫu ngẫu nhiên trong từng nhóm nhầm lẫn với quy mô được xác định rõ.

### 5.2. Những gì đã cập nhật

File `reports/contextual_model_validation/audit/validation_18_errors_review.csv` đã hoàn thiện nhận xét và contribution cho cả 18 annotation. Mỗi dòng có phân tích riêng, phân công cụ thể, bản sửa lấy từ SGML và annotation liên quan khi cần. Phần rà soát kết quả đã hoàn thành; nhãn thật, prediction và dữ liệu câu gốc được giữ nguyên.

Đối chiếu với nguồn cho thấy cả 18 span và gold đều khớp annotation gốc. Trong parse đã lưu, 17 ca có alignment exact; ca 11696 là whitespace_edge vì span chứa dấu cách cuối. Nhận xét phân biệt ba mức: điều thấy trực tiếp trong bản sửa, cách giải thích ngôn ngữ học, và giả thuyết về nguyên nhân mô hình dự đoán sai. Các giả thuyết về tác động của POS hoặc dependency chưa được ghi thành kết luận nhân quả.

Phần làm rõ thêm đáng chú ý nằm ở các annotation lân cận:

- **15390:** annotation 15389 xóa động từ tình thái, 15388 sửa cấu trúc mệnh đề, và 15391 sửa động từ phối hợp. Khi đọc cả nhóm sửa, việc đổi động từ sang ngôi thứ ba số ít có cơ sở; đọc riêng target sẽ dễ nhận xét nhầm rằng gold mâu thuẫn với câu.
- **19068:** annotation 19067 sửa chủ ngữ sang số nhiều. Bản sửa target sau đó vừa đổi trợ động từ vừa sửa phân từ, nên có cả tín hiệu SVA và Vform. Đây là một bản sửa nhiều thành phần, không thể giải thích chỉ bằng một đuôi từ.
- **14939:** annotation 14938 sửa cụm đứng trước target. Vì cấu trúc xung quanh cũng thay đổi, việc giải thích nhãn SVA cần đọc cả cụm chỉnh sửa. Ca này vẫn được đánh dấu cần kiểm tra lại tính nhất quán của nhãn và bản sửa, thay vì tự đổi gold.

Các nhận xét này giúp tránh hai cách suy luận quá nhanh: thấy mô hình sai thì quy ngay cho parser, hoặc thấy bản sửa khó hiểu trong câu gốc thì kết luận annotation sai. Audit hiện tại không được dùng để đổi nhãn rồi tính lại điểm mô hình.

### 5.3. Kết luận từ phần đọc ca

Các ca đã đọc cho thấy khó khăn của bài toán liên quan đến cả quyết định phân loại và cách annotation biểu diễn lỗi. Một bản sửa có thể đồng thời tác động đến số danh từ và từ hạn định, hoặc đến hòa hợp chủ–vị và dạng động từ, trong khi mỗi sample chỉ nhận một nhãn. Một số span rộng hơn phần sửa tối thiểu; một số bản sửa lại phụ thuộc annotation lân cận. Vì vậy, POS của riêng target chưa đủ để giải thích tất cả các nhầm lẫn.

Audit cũng ghi nhận những trường hợp mô hình vẫn sai dù parse có tín hiệu phù hợp, như ca 12019, và trường hợp có dấu hiệu POS cần kiểm tra lại, như ca 11696. Chưa có căn cứ quy toàn bộ lỗi cho parser hoặc kết luận các gold khó giải thích đều sai. Cả 18 span và gold đều khớp nguồn chỉ xác nhận dữ liệu đã được tái hiện đúng; điều đó chưa thay thế việc thẩm định chất lượng nhãn. Những nhận định này là bằng chứng từ các ca cụ thể, chưa phải kết luận về tỷ lệ nguyên nhân trong toàn bộ 364 ca sai.

### 5.4. Contribution cho phần audit

Phân công dưới đây **chỉ áp dụng cho 18 annotation**, mỗi người 6 ca. Tên người phụ trách và phạm vi công việc được ghi ngay ở từng dòng CSV.

| Thành viên | Số ca | Annotation ID | Phạm vi phụ trách |
| --- | --- | --- | --- |
| Quân | 6 | 10824, 11696, 12352, 11501, 12050, 12301 | Nn/ArtOrDet: số danh từ, countability, từ hạn định, POS và ranh giới span. |
| Duy | 6 | 12314, 14761, 14764, 12019, 14939, 15390 | Vform/SVA: thì, tình thái, hòa hợp chủ–vị và các bản sửa lân cận. |
| Công | 6 | 1269, 18774, 19065, 19068, 19069, 19185 | Span dài hoặc nhiều thay đổi: xác định phần sửa chính và những chỗ cần thống nhất nhãn. |

Nhóm đã hoàn tất rà soát kết quả audit **18/18 annotation** ngày 01/11/2026. Bảng contribution ghi phạm vi phụ trách của Quân, Duy và Công, mỗi người sáu ca.

## 6. Hoàn thiện 40 ca BERT/LR và đề xuất xử lý 22 ID

### 6.1. Phạm vi và trạng thái hoàn thành

Phần đọc ca BERT/LR sử dụng prediction test đã lưu của BERT seed chính và LR Full. Mẫu được chọn có chủ đích sau khi có prediction: 12 ca BERT sai/LR đúng, trong đó ưu tiên sáu ca gold SVA; 12 ca theo chiều ngược lại; 10 ca cả hai sai; và sáu ca cùng đúng làm đối chứng. Vì vậy, trong 40 ca có 24 ca bất đồng dự đoán, 10 ca cùng sai và sáu ca cùng đúng. Đúng hoặc sai ở đây được xác định theo gold hiện hành, không thay thế việc xem xét chất lượng annotation.

| Nhóm | Số ca | Nội dung đã hoàn thành |
| --- | --- | --- |
| BERT sai, LR đúng | 12 | Nhận xét từng ca, đối chiếu thao tác sửa và giải thích loại nhầm lẫn. |
| BERT đúng, LR sai | 12 | Xác định tín hiệu ngữ pháp, ranh giới span và các điểm cần xem lại. |
| Cả hai cùng sai | 10 | Phân tích lỗi chung, các bản sửa nhiều thành phần và nghi vấn nhãn. |
| Cả hai cùng đúng | 6 | Đọc ca đối chứng, gồm cả span một phần token và correction cần kiểm tra. |

Cả **40/40 ca đã có nội dung review và contribution hoàn chỉnh**, kèm người phân tích, ngày thực hiện và mức tin cậy định tính. Mẫu này không đại diện cho tỷ lệ lỗi trên toàn bộ test: riêng nhóm cả hai sai gồm chín ArtOrDet và một Nn do cách duyệt các nhóm chọn mẫu. Phần contribution của bộ 40 ca mô tả giá trị của từng ca đối với phân tích, không phân công thêm công việc cho Quân, Duy hoặc Công ngoài bộ 18 ca đã nêu.

Nhóm đã rà soát và chốt **40/40 review cùng 22/22 nhận định xử lý ngày 01/11/2026**. Các trường nhận xét, đóng góp của từng ca và trạng thái hoàn thành đã được cập nhật đầy đủ. Hồ sơ giữ các căn cứ đối chiếu để có thể kiểm tra lại từng nhận định.

### 6.2. Kết quả xử lý hàng đợi 22 ID

Hàng đợi gồm 17 ID từ audit mô hình cổ điển và sáu ID ưu tiên từ audit BERT; ID 34539 xuất hiện ở cả hai, nên có 22 ID duy nhất. Tám ID trong hàng đợi cũng thuộc mẫu 40 ca BERT/LR. Tổng cộng **54 annotation khác nhau** đã được đối chiếu lại về span, gold và correction với SGML trong lượt hoàn thiện này.

Đã hoàn thành và được nhóm rà soát và chốt **22/22 nhận định xử lý**, mỗi nhận định có căn cứ và annotation liên quan khi cần. Các nhóm dưới đây mô tả hướng xử lý chính của từng ca; chúng không phải số lượng nhãn đã được sửa.

| Hướng xử lý đề xuất | Số ca | Annotation ID |
| --- | --- | --- |
| Có cơ sở giữ nhãn theo bản sửa và ngữ cảnh | 9 | 18531, 37631, 24526, 9351, 41235, 35455, 211, 37624, 32801 |
| Cần xem lại sự phù hợp của nhãn | 6 | 34539, 12324, 26635, 32705, 37909, 16809 |
| Nghi vấn correction hoặc phạm vi thay thế | 3 | 18501, 23037, 30366 |
| Chỉnh sửa có thể tùy chọn | 3 | 201, 11916, 6079 |
| Bản sửa gồm nhiều thao tác | 1 | 32874 |

Một số điểm được làm rõ sau khi đọc thêm nguồn:

- **32801:** bản sửa số nhiều của từ viết tắt nhất quán với 32799 xóa mạo từ và 32800 chuyển danh từ đầy đủ sang số nhiều. Nghi vấn khi chỉ đọc target đã được giải thích bằng nhóm annotation đi kèm.
- **18531:** 18529 bỏ cụm chứa động từ và dấu nguyên mẫu, còn 18530 chia lại động từ đứng trước. Sau các sửa này, động từ target đi với chủ ngữ số ít nên nhãn SVA có cơ sở.
- **23037:** ý định sửa số danh từ có thể hiểu được, nhưng span chỉ là một ký tự cuối còn replacement chứa cả từ. Đã đề xuất hai cách kiểm tra phạm vi thay thế, chưa sửa offset hoặc correction nguồn.
- **18501 và 30366:** correction có thể làm cấu trúc sau sửa không phù hợp. Đây là vấn đề cần xem lại ở bản sửa nguồn, dù một mô hình có thể dự đoán trùng gold.

Ba ca được xếp vào nhóm có thể tùy chọn cần được đọc như lựa chọn diễn đạt hoặc tham chiếu, thay vì mặc nhiên xem mọi phần thêm vào là bắt buộc. Sáu ca đề nghị xem lại nhãn chỉ ra sự chưa tương ứng giữa thao tác trong correction và loại lỗi được gán; chưa tự thay gold bằng một nhãn dự đoán khác.

### 6.3. Kết luận từ lượt review bổ sung

Kết quả củng cố nhận xét rằng điểm đúng/sai theo gold và tính hợp lý của annotation là hai vấn đề cần đọc riêng. Ca cùng đúng vẫn có thể chứa correction cần kiểm tra, như 30366; một nghi vấn ban đầu cũng có thể được giải thích bằng annotation lân cận, như 32801. Các ca 24526, 9351, 41235 và 35455 có dấu hiệu POS hoặc dependency tự động chưa phù hợp với cách đọc câu, nhưng chưa có phép can thiệp thay parse để khẳng định tác động nhân quả lên prediction.

Nhóm đã hoàn tất rà soát **40 bản review cùng 22 nhận định xử lý**. Những ca còn điểm chưa rõ đã có nhận định và hướng xử lý cụ thể thay vì để trống. Việc rà soát chốt nội dung phân tích; các vấn đề về nhãn hoặc correction vẫn được ghi rõ trong kết quả. Không huấn luyện lại, sửa gold hoặc tính lại điểm sau review; các bảng kết quả mô hình trong báo cáo giữ nguyên.

Bản tổng hợp công khai nằm tại `reports/bert_final_analysis/review_summary.md`. Hồ sơ chi tiết từng ID ở máy nghiên cứu gồm câu gốc, span, correction, prediction, nhận xét và căn cứ. Các file chứa câu nguồn tiếp tục được giữ nội bộ.

## 7. Kết luận và phần còn lại

Kết quả mới thay đổi cách nhìn về lần thử BERT ban đầu: sau đợt huấn luyện mở rộng, mô hình học được trên train và đạt validation quanh 0,818. Tuy nhiên, trên test hiện có, BERT seed chính và LR Full vẫn gần nhau về macro-F1; BERT cũng chưa cải thiện đồng đều trên mọi lớp và mọi nhóm dùng chung POS. Với kết quả này, LR Full tiếp tục là mốc so sánh phù hợp, còn lợi ích của biểu diễn ngữ cảnh cần được trình bày theo từng nhóm lỗi.

Phần rà soát kết quả audit 18 annotation đã hoàn tất. Những ca có ranh giới nhãn chưa rõ vẫn được giữ trong danh sách cần đối chiếu thêm với hướng dẫn annotation; việc chốt nhận xét không tự động thay đổi gold hoặc điểm đánh giá. Báo cáo giữ rõ hai giới hạn của thí nghiệm: cấu hình mở rộng chưa trùng hoàn toàn grid được gợi ý, và kết quả BERT là phần bổ sung sau khi test của mô hình cổ điển đã được xem. Không điều chỉnh mô hình dựa trên các kết quả test vừa tổng hợp.

Phần bổ sung gồm 40 review BERT/LR và 22 nhận định xử lý đã được nhóm rà soát và chốt đầy đủ. Nhóm giữ các nhận định ở đúng phạm vi bằng chứng: chốt cách diễn giải kết quả audit, đồng thời giữ nguyên nhãn và dữ liệu đã khóa.

Về khả năng tái lập, bản phân tích ngày 28/10 còn ghi nhận bốn file mã nguồn của khóa cổ điển không có trong workspace lúc kiểm tra và hai CSV đã đổi định dạng. Metric và CI vẫn đối chiếu được, nhưng bước kiểm tra đầy đủ khóa cổ điển chưa qua. Vì vậy, bản ghi 61/61 kiểm tra của phần phân tích BERT không được diễn giải thành xác nhận rằng mọi thành phần của thí nghiệm cổ điển đều đã tái lập đầy đủ.

## Tài liệu và kết quả đối chiếu

Hai tài liệu đầu vào là `GROUP05_CLC02_WEEK10_REPORT.docx` và `Nhan_xet_Week10_CLC02_Group05.docx`. Các bảng trong báo cáo lấy từ các nguồn sau:

- `reports/bert_small_experiment/manifest.json`, `selection.json` và 12 file `*/summary.json`: cấu hình, seed, checkpoint và điểm train/validation.
- `reports/nucle_pos_baseline_v1/validation_predictions.csv`, `reports/nucle_feature_benchmark_solver_v2/validation_predictions.csv` và prediction validation của ba seed span mean: đối chiếu trên cùng các nhóm khó.
- `reports/evaluation_followup/reviewed_overall_results.csv`: các hệ thống cổ điển, gồm Tree S+L+P+M.
- `reports/bert_final_analysis/overall_results.csv`, `paired_differences.csv`, `summary.json` và `verification.json`: kết quả BERT, so sánh paired và trạng thái kiểm tra đã lưu.
- `reports/final_evaluation/lock.json`, `test_scoring_started.json`; `reports/bert_small_experiment/selection.sha256`, `test/test_started.json`: mốc khóa và liên kết hash.
- File audit CSV, packet parse đã lưu và annotation trong SGML gốc: căn cứ cho phần audit. Câu gốc được giữ ở file audit nội bộ; báo cáo không chép lại toàn bộ 18 câu.
- `reports/bert_final_analysis/review_summary.md`: tổng hợp 40 review và 22 nhận định được nhóm rà soát và chốt; hồ sơ từng ca giữ nội bộ. Các ID tương ứng được đồng bộ trong `reports/evaluation_followup/expert_review_queue.csv`.
