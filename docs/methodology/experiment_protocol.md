# Protocol thực nghiệm

Cập nhật tổ chức: 23/10/2026. Đây là nguồn quy tắc thực nghiệm. Baseline/ablation/masking validation đã chạy; BERT-small đã chạy trong tuần 1; test đã chấm sau khóa ngày 23/10/2026. Kết quả task ở [nhóm báo cáo đánh giá](../../reports/summaries/evaluation/README.md); các mục lịch sử giữ thời điểm cũ. Tiến độ: [status.md](../status.md); lịch: [note.md](../../note.md); schema/offset: [preparation.md](preparation.md).

Tài liệu này tổng hợp quy tắc và lịch sử thay đổi. Protocol đã ghi trước đợt final và được băm trong lock là [final_evaluation_protocol.md](../final_evaluation_protocol.md); các bổ sung giải thích ở đây không thay protocol đó. Cách chạy và input local: [reproduction.md](../reproduction.md).

## 1. Nguyên tắc và các quyết định đã chốt

- Mỗi annotation là một sample, một nhãn; giữ nguyên unresolved, overlap và input trùng khác nhãn. Không majority-vote hoặc sửa ground truth theo dự đoán model. Chỉ sửa/loại theo bằng chứng nguồn và nhật ký trước khóa dataset.
- Không đưa correction, corrected sentence, comment, nhãn hoặc IDs vào predictors. Parse câu gốc; fit mọi transformation học từ dữ liệu chỉ trên train.
- Input trùng hoàn toàn phải nằm cùng nhóm split, kể cả khác essay/nhãn; đây là ambiguity của target, khác với hai input chỉ trùng feature vector. Không dùng các mẫu test đã review để thiết kế feature hoặc tune sau khóa test.
- Không đặt ngưỡng F1 để định nghĩa thành công; kết quả âm vẫn trả lời RQ. Không dùng model/demo lịch sử làm benchmark hiện hành.
- **Đã khóa:** duplicate/boilerplate policy và seed split trong grouping/split artifacts. P/M/D dùng spaCy/model đã pin theo POS contract; tổng hợp span nhiều token giữ thứ tự token và gộp thuộc tính bằng quy tắc cố định trong feature extractor. Còn cần chốt nơi sao lưu riêng.

## 1.1 Grouping policy và split hiện hành

- Ghép essay nếu toàn văn trùng exact hoặc sau Unicode NFC + casefold + collapse whitespace. Ghép essay nếu có paragraph chuẩn hóa giống nhau, trừ boilerplate miễn theo điều kiện dưới đây. Không strip dấu câu, stem hoặc nhận paraphrase tự động.
- Boilerplate được miễn nối chỉ khi cùng paragraph có mặt trong ít nhất 10 essay, có tối đa 10 từ và 80 ký tự sau chuẩn hóa, và không có annotation mục tiêu trong bất cứ occurrence nào. Nếu paragraph có target annotation thì luôn nối, bất kể độ ngắn/tần suất.
- Input model trùng chính xác (sentence + offsets) luôn nối essay. Policy output ghi IDs/hashes và decision, không ghi text corpus.
- Trên source/preparation hash hiện hành: 62 nhóm normalized paragraph candidates, trong đó 7 được miễn boilerplate; 55 nhóm còn lại được nối. Không có exact/normalized full-essay duplicate, không có exact model-input duplicate; có 1.280 connected essay groups, nhóm lớn nhất 34 essays/338 samples.
- Split seed `20261013`: train 11.052 (70,00%), validation 2.368 (15,00%), test 2.368 (15,00%). Cả năm label xuất hiện ở mỗi partition. Manifest và summary ở `data_set/split/nucle_group_split/`; output hashes trong `manifest.json`.
- Near-duplicate/paraphrase không được tự động phát hiện; đây là hạn chế cần nêu trong report và không được diễn giải split là bảo đảm chống mọi semantic leakage.

## 2. Đặc tả feature compact

| Nhóm | Định nghĩa vận hành |
| --- | --- |
| S — Structural | Số token câu, số token và số ký tự span, vị trí start/end tương đối, span_kind theo offset. Quy tắc token cắt biên được chốt trong preparation. |
| L — Lexical | Token gốc đã lowercase trong span và tối đa hai token ngoài span ở mỗi phía; lexical n-gram cục bộ theo quy tắc dưới đây. Không xóa stopwords. |
| P — POS | POS của token span, POS hai phía và pattern POS cục bộ; tổng hợp span nhiều token bằng quy tắc cố định. |
| M — Morphology | Number, Person, Tense, VerbForm của span; biểu diễn tập giá trị/hiện diện và missing indicators, không dùng correction. |
| D — Dependency | Nhãn quan hệ, POS của head/dependent, khoảng cách, hiện diện chủ ngữ và thuộc tính ngữ pháp của chủ ngữ. Không chứa surface word hoặc lemma của governor/dependent. |

**Implementation đã chạy:** TrainOnlyEncoder dùng numeric nguyên giá trị, one-hot cho categorical và multi-hot kênh pooled n-gram. Giới hạn 500 áp dụng kênh n-gram; categorical lexical slots học vocabulary riêng. Full có 16.180 chiều, không có z-score. Parse lại câu gốc ở bước features; preparation token list không trực tiếp thay parse feature. Xem [implementation audit](implementation_data_audit.md).

Đặc tả L dự kiến ban đầu (lịch sử; các khác biệt triển khai đã xác minh ở mục 9 thay thế mô tả tương ứng khi diễn giải kết quả hiện có):

- Một vocabulary chung tối đa 500 lexical item cho toàn bộ vai trò span và context, không phải 500 mục mỗi vị trí.
- Candidate gồm unigram và bigram từ span và cửa sổ hai token ngoài span. Bigram chỉ nối token kề nhau trong câu gốc và cùng nằm trong vùng được chọn; không tạo bigram qua phần span bị lược bỏ.
- Xếp hạng candidate theo tần suất xuất hiện trong train; khi hòa dùng thứ tự từ điển cố định. Mỗi annotation train đóng góp một cửa sổ; ghi rõ cách đếm này vì các annotation trong cùng câu có thể lặp ngữ cảnh.
- Vocabulary 500 là giới hạn thiết kế ban đầu, không phải lựa chọn đã được chứng minh tối ưu. UNK và các marker đặc biệt được ghi riêng ngoài ngân sách 500.
- Token tại left_2, left_1, right_1, right_2 dùng one-hot theo vai trò với cùng vocabulary. Token trong span và bigram dùng multi-hot theo vai trò. Vì vậy 500 lexical item không đồng nghĩa tổng feature dimension là 500; lưu số chiều thực tế.
- Với span nhiều token, context bắt đầu bên ngoài toàn bộ span, không lấy các token bên trong làm trái/phải. Với insertion, dùng các token kề vị trí chèn.
- Các token giao span được xử lý theo chính sách alignment đã khóa; không lấy correction để chọn token mục tiêu.
- Nếu thử kích thước khác, chốt danh sách và ngân sách trước, chỉ chọn bằng validation. Không bắt buộc thử thêm trong lịch 5 tuần.
- Encoder, vocabulary, imputation và scaling chỉ fit trên train; unknown/missing/NOT_APPLICABLE được biểu diễn tường minh. Sử dụng cùng feature matrix compact cho LR và Tree ở từng biến thể.

## 3. Ý nghĩa của ablation

Cách chính: tune riêng từng feature set trong cùng grid và cùng ngân sách tương ứng với model. Câu hỏi được trả lời là: với cùng ngân sách tuning, mỗi pipeline feature đạt kết quả được chọn trên validation tốt đến đâu?

Đây không phải phép đo cô lập feature tại cùng hyperparameters, vì cấu hình model có thể thay đổi. Không chạy thêm nhánh giữ hyperparameters cố định trừ khi có câu hỏi cụ thể và ngân sách riêng.

- Bắt buộc: S+L → S+L+P → S+L+P+M → S+L+P+M+D; thực hiện cho cả LR và Tree.
- Bắt buộc: full, full che lexical span, S+P+M+D bỏ toàn bộ lexical.
- Đã bổ sung trước final test: Full−P và Full−M cho cả LR/Tree; Full−D trùng S+L+P+M nên reuse. Đây là phạm vi đã chốt ở final protocol, thay kế hoạch tùy ngân sách trước đó.
- Chuỗi tăng dần đo đóng góp theo thứ tự đã chọn. Morphology thêm sau POS không đo tác dụng khi morphology đứng riêng. Bỏ từng nhóm bổ sung góc nhìn có điều kiện; các nhóm vẫn có thể chứa thông tin trùng nhau.
- S+P+M+D gọi là non-lexical, không gọi là structural-only; structural-only đúng nghĩa chỉ gồm S.
- Lịch sử từng coi ablation bỏ nhóm là tùy ngân sách; trước final test đã chốt và hoàn thành ba đối chiếu P/M/D. Không tính Full−D như một lượt fit hoặc bằng chứng độc lập thứ hai.

## 4. Kiểm tra masking

Parser chạy trên câu gốc và được dùng chung cho các biến thể. Masking xảy ra trước bước học vocabulary và mã hóa lexical trong từng pipeline; vocabulary của biến thể masked chỉ học từ dữ liệu train đã áp dụng quy tắc masking.

Loại mọi lexical identity trực tiếp của token giao span khỏi surface word, lemma, n-gram, cửa sổ trái/phải, lexical governor/dependent và feature tổ hợp word–POS. Đặc tả D chính không chứa lexical governor/dependent; kiểm tra này ngăn việc vô tình bổ sung chúng sau đó.

Dùng marker cố định cho span đã che; không tạo n-gram giả nối hai phía qua span. Có test bằng span nhiều token và kiểm tra feature dictionary trước encoding. Test masking giữ cố định parse để xác minh đường truyền lexical; không yêu cầu P/M/D giữ nguyên khi thay đổi câu đầu vào và parse lại.

POS, morphology và dependency vẫn có thể tương quan với danh tính token. Vì vậy diễn giải kết quả là mức phụ thuộc vào lexical identity trực tiếp trong protocol này, không phải bằng chứng model hiểu ngữ pháp. Với insertion không có token span để che; báo cáo riêng nếu có dữ liệu, nếu không ghi N/A và support = 0.

## 5. Split, lựa chọn model và metric

- Chia xấp xỉ 70/15/15 theo essay hoặc nhóm duplicate; dùng author_id nếu đáng tin cậy. Chọn split theo tính hợp lệ và hỗ trợ lớp, không theo performance.
- Khóa quy tắc boilerplate/duplicate trước split. Không tự lấy ngưỡng lịch sử làm chính sách cuối. Báo cáo giới hạn phát hiện near-duplicate.
- Majority chọn lớp phổ biến nhất trong train.
- **POS contract đã khóa (`nucle-pos-span-v1`):** spaCy `3.8.7` cùng `en_core_web_sm 3.8.0` (wheel URL và checksum được pin trong `pyproject.toml`/`uv.lock`); dùng UPOS (`token.pos_`) từ model đã huấn luyện, giữ tagger/parser, tắt NER. Chạy trên câu gốc và chỉ dùng offset ký tự nửa mở `[error_start,error_end)`; trước ánh xạ phải xác minh `sentence[start:end] == error_span`.
- Với span không rỗng, lấy mọi token spaCy giao ký tự với span. Span khớp biên token là `exact`; cắt ít nhất một token là `partial_token` và giữ POS của mọi token giao span theo thứ tự; span chỉ thêm whitespace đầu/cuối là `whitespace_edge`, bỏ whitespace khỏi token selection nhưng giữ cờ. POS signature nhiều token là chuỗi UPOS theo thứ tự, nối bằng `+`; không dùng chính tả token, lemma, correction hoặc POS ngoài span trong baseline POS-only. Lưu `left_upos`/`right_upos` để feature sau này, nhưng không đưa chúng vào baseline này.
- Insertion có POS `NOT_APPLICABLE`: ở biên token lưu POS token trái/phải làm context; nằm giữa ký tự token gắn `insertion_inside_token`, lưu enclosing POS để audit nhưng không coi là POS mục tiêu. Cả hai trường hợp dùng lớp phổ biến nhất train trong baseline POS-only. Span whitespace-only hoặc token thiếu POS dùng `POS_MISSING` và cùng fallback. Mọi span ngoài bounds hoặc không khớp `error_span` làm pipeline dừng với lỗi, không âm thầm fallback.
- **POS rule đã khóa:** UPOS đơn `ADP → Prep`, `DET → ArtOrDet`; mọi POS signature khác đã thấy trong train dự đoán lớp đa số của signature đó; signature chưa thấy, missing, insertion dùng lớp đa số toàn train. Ties theo thứ tự `ArtOrDet, Prep, Nn, Vform, SVA`. Thống kê signature × nhãn chỉ fit từ train. Báo support riêng, nhất là signature ít mẫu và POS missing.
- Baseline này là chẩn đoán dùng POS của toàn span (chuỗi UPOS với span nhiều token), không phải POS của ngữ cảnh. Đối chiếu Majority/POS rule/LR/Tree trên test đã thực hiện sau khóa; kết quả và giới hạn được báo riêng, không thay số validation.
- LR: C ∈ {0.01, 0.1, 1, 10}, class_weight ∈ {None, balanced}; kiểm tra convergence.
- Tree: max_depth ∈ {4, 8, 12}, min_samples_leaf ∈ {5, 20}, class_weight ∈ {None, balanced}; seed cố định.
- Trong runner `src/training/nucle_benchmark.py`, LR dùng PyTorch LBFGS, tối đa 1.000 iteration/1.250 evaluations, `tolerance_grad=1e-6`, `tolerance_change=1e-9`; bias không regularize và hệ số L2 là `0.5 * ||W||² / (C * n_train)`. LR đủ điều kiện chọn khi dừng theo gradient tolerance hoặc theo change tolerance trước giới hạn iteration/evaluation; báo cáo termination reason và gradient infinity norm. Cấu hình chạm giới hạn vẫn được đánh dấu chưa hội tụ. Tree là CART Gini xác định trên sparse one-hot/numeric vectors; seed không ảnh hưởng phép chia nhưng được giữ cố định cho tính nhất quán.
- Hòa Macro-F1 được giải quyết bằng thứ tự grid cố định: LR `C` tăng dần rồi `class_weight=None` trước `balanced`; Tree `max_depth` tăng dần, `min_samples_leaf` tăng dần rồi `None` trước `balanced`. Chọn cấu hình xuất hiện đầu tiên khi điểm bằng nhau.
- Chọn cấu hình theo Macro-F1 validation. Chốt thứ tự xử lý hòa điểm trước khi chạy; không chọn tùy ý sau khi xem test.
- Quyết định của protocol này: báo cáo chính dùng model fit trên train, không refit train+validation sau tuning. Đây là lựa chọn của kế hoạch, không diễn giải thành lệnh cấm refit nói chung. Bản proposal hiện tại, mục 8, có câu “Giữ nguyên train đã fit để báo cáo chính, tránh thay đổi protocol bằng refit tùy ý.” Quyết định này nhất quán với câu đó.
- Luôn tính metric với danh sách cố định ArtOrDet, Prep, Nn, Vform, SVA; zero_division = 0. Báo cáo Macro-F1, accuracy, macro precision/recall, precision/recall/F1/support từng lớp và confusion matrix.
- Bootstrap 1.000 lần theo nhóm split độc lập, có hoàn lại; nhóm được chọn nhiều lần thì các annotation của nó được tính với bội số tương ứng. Dùng cùng lần resample khi so sánh hai cấu hình và lưu seed.
- Nếu resample thiếu lớp, vẫn giữ đủ 5 nhãn và zero_division = 0; không âm thầm bỏ lần đó hoặc lấy lại mẫu. Ghi số/tỷ lệ lần thiếu mỗi lớp, số lần thiếu ít nhất một lớp và ảnh hưởng đến độ ổn định khoảng tin cậy. Báo cáo khoảng percentile 95% và paired difference.

## 6. Lưu trữ bền vững và tái lập

- Giữ corpus, văn bản sample, correction và artifact có nội dung nguồn ngoài Git công khai. Theo yêu cầu công bố code và docs, tài liệu hiện hành, tests và báo cáo không chứa văn bản nguồn được đưa lên Git; xem reports/README.md. Quy tắc cũ chỉ push proposal không còn áp dụng.
- Không coi việc có file trên một máy là đủ tái lập. Tạo gói reproducibility có phiên bản: source/code checksum, environment lock, review decisions, split manifest, seed, grouping policy, schema, feature specification, experiment configurations và artifact checksums.
- Review quyết định và split manifest tối thiểu chỉ chứa IDs, lý do và metadata cần thiết; phần bằng chứng có văn bản nguồn lưu riêng với quyền truy cập phù hợp. ID không được mặc định là dữ liệu có thể công khai khi chưa kiểm tra điều kiện nguồn.
- Code, cấu hình thí nghiệm không chứa corpus, script tái tạo và tài liệu được quản lý phiên bản bằng Git. Giữ byte của file đã khóa; không sửa protocol lịch sử sau kết quả.
- Trước khi khóa split, chọn một nơi sao lưu riêng có kiểm soát truy cập, như ổ sao lưu thứ hai hoặc kho riêng phù hợp quyền sử dụng. Đồng bộ gói metadata tại mỗi milestone và thử khôi phục từ bản sao vào thư mục khác; so checksum, manifest và quyết định review.
- Kế hoạch không tự cấp quyền upload dữ liệu sang dịch vụ bên ngoài. Khi triển khai, dùng nơi lưu đã được người dùng cho phép; nếu chưa có, ghi trạng thái chưa hoàn thành lưu trữ bền vững, không tuyên bố đã tái lập đầy đủ.

## 7. Ngân sách đã chạy và phạm vi bàn giao

Đợt sáu feature set lịch sử gồm 8 cấu hình LR và 12 Tree mỗi set: 120 fit. Đợt final có 40 grid fit Full−P/M và 12 refit cấu hình cũ để lưu checkpoint, tổng 52 fit. Full−D reuse nên không thêm 20 fit. Hai đợt này có mục đích khác nhau; không gộp thành 52 lần tìm kiếm toàn bộ lịch sử.

Ngân sách error analysis ban đầu tối đa 50 mẫu sai và 10 mẫu đúng. Thực chạy: 25 lỗi LR, 5 lỗi Tree khi LR đúng và 10 controls, tổng 40 ca, chọn theo quy tắc trong final protocol. Lưu parser evidence, ambiguity, mức chắc chắn và annotation_id. Correction chỉ được dùng ở bước phân tích sau dự đoán. Báo cả lỗi kỹ thuật parser và chất lượng ngữ nghĩa quan sát, không coi có cây parse là parse đúng.

Đã có benchmark, ablation/masking, test uncertainty, error analysis và report task có bằng chứng cho RQ1–RQ3. Theo phạm vi người dùng chốt, final report dự án và sáu slide để sau; không coi việc hoàn tất thí nghiệm là đã hoàn tất bản nộp. Không thêm TF-IDF/random baseline/prototype hoặc tuning BERT sau test.


## 8. Bổ sung phạm vi BERT — ghi trước chạy ngày 23/10/2026 (lịch sử)

BERT đã vào kế hoạch trước bước khóa mô hình/test, chưa triển khai. Giữ cùng split và năm nhãn; đầu vào câu gốc đánh dấu span, không correction. Chọn checkpoint bằng macro-F1 toàn validation; báo cả toàn tập và các tập con đã định nghĩa. Trước lượt chạy chính phải chốt checkpoint/revision, marker/tokenization, truncation giữ span, grid, seed, optimizer, epoch/early stopping và ngân sách. Các giá trị này chưa được chốt. Lịch và số lượt dự kiến chỉ quản lý tại note.md.

Giữ quy tắc train-only, chọn bằng validation và đánh giá test sau khóa ở mục 5. Không diễn giải kế hoạch BERT thành kết quả.

Marker mới phải được thêm vào tokenizer và cập nhật embedding tương ứng; chốt xử lý partial-token và báo số câu bị truncation. Giữ dự đoán năm lớp trên các tập con; không thay bằng binary classifier để so trực tiếp. Nếu bàn về dễ/khó, định nghĩa nhóm trước khi xem kết quả, báo support và chênh lệch trên cả hai.


## 9. Implementation audit 2026-10-23

The executed baseline uses a pooled 500-item n-gram channel plus separate train-fit categorical lexical slots, with raw numeric values and reparsing of original sentences. The original shared-vocabulary proposal is not the implemented model. Masking removes span lexical features before fitting its encoder while retaining the original parse. Detailed findings: [implementation_data_audit.md](implementation_data_audit.md). Week-1 BERT configuration: [bert_experiment_protocol.md](../bert_experiment_protocol.md). This records an implementation deviation; it does not claim prior registration or change existing predictions.


## 10. Hoàn tất validation/BERT — mốc lịch sử trước final test

BERT-small đã thực hiện theo bert_experiment_protocol.md, không sửa cấu hình đăng ký sau kết quả. Report có trong workspace: [contextual_model_validation_report.docx](../../reports/summaries/contextual/contextual_model_validation_report.docx); không có bản Markdown tương ứng. Số liệu và trạng thái trong mục 8 là mốc trước chạy. Ở cuối giai đoạn validation/BERT, test/bootstrap/freeze chưa làm; mục 11 và status.md ghi những bước đã thực hiện sau đó. Sensitivity LR dùng 4 threads; chạy lại baseline đã lưu với 10 threads tái tạo prediction cũ. Ghi rõ threads khi tái lập LBFGS.

## 11. Đánh giá cuối — thực hiện 23/10/2026

Protocol bổ sung ghi trước chạy: [final_evaluation_protocol.md](../final_evaluation_protocol.md), hash lưu trong lock. Phạm vi cuối: Majority/POS và9 variants×LR/Tree. Full−D trùng S+L+P+M nên reuse;40 grid fit mới Full−P/M và12 refit cũ. Mọi validation replay cũ khớp trước khóa; không refit train+validation.

Test,1.000 paired connected-group bootstrap và audit40ca đã hoàn thành. BERT dừng thêm; không suy ngữ cảnh vô ích. Kết quả/giới hạn: [evaluation_tasks_report.docx](../../reports/summaries/evaluation/evaluation_tasks_report.docx). Không sửa ngược protocol trước test để hợp thức hóa kết quả.
