# Protocol thực nghiệm

Cập nhật: 12/10/2026. Kế thừa các quyết định trong kế hoạch 5 tuần, chưa có feature/model/split được triển khai. Lịch và tiến độ theo dõi ở [status.md](status.md); schema/offset đã chốt ở [preparation.md](preparation.md); câu hỏi nghiên cứu ở [proposal.md](proposal.md).

## 1. Nguyên tắc và các quyết định còn thiếu

- Mỗi annotation là một sample, một nhãn; giữ nguyên unresolved, overlap và input trùng khác nhãn. Không majority-vote hoặc sửa ground truth theo dự đoán model. Chỉ sửa/loại theo bằng chứng nguồn và nhật ký trước khóa dataset.
- Không đưa correction, corrected sentence, comment, nhãn hoặc IDs vào predictors. Parse câu gốc; fit mọi transformation học từ dữ liệu chỉ trên train.
- Input trùng hoàn toàn phải nằm cùng nhóm split, kể cả khác essay/nhãn; đây là ambiguity của target, khác với hai input chỉ trùng feature vector. Không dùng các mẫu test đã review để thiết kế feature hoặc tune sau khóa test.
- Không đặt ngưỡng F1 để định nghĩa thành công; kết quả âm vẫn trả lời RQ. Không dùng model/demo lịch sử làm benchmark hiện hành.
- **Cần chốt trước triển khai tương ứng:** duplicate/boilerplate policy và seed split; phiên bản model NLP cho P/M/D và quy tắc tổng hợp span nhiều token; thứ tự xử lý hòa Macro-F1 và convergence; nơi sao lưu riêng. Những mục này chưa được coi là đã khóa bởi tài liệu này.

## 2. Đặc tả feature compact

| Nhóm | Định nghĩa vận hành |
| --- | --- |
| S — Structural | Số token câu, số token và số ký tự span, vị trí start/end tương đối, span_kind theo offset. Quy tắc token cắt biên được chốt trong preparation. |
| L — Lexical | Token gốc đã lowercase trong span và tối đa hai token ngoài span ở mỗi phía; lexical n-gram cục bộ theo quy tắc dưới đây. Không xóa stopwords. |
| P — POS | POS của token span, POS hai phía và pattern POS cục bộ; tổng hợp span nhiều token bằng quy tắc cố định. |
| M — Morphology | Number, Person, Tense, VerbForm của span; biểu diễn tập giá trị/hiện diện và missing indicators, không dùng correction. |
| D — Dependency | Nhãn quan hệ, POS của head/dependent, khoảng cách, hiện diện chủ ngữ và thuộc tính ngữ pháp của chủ ngữ. Không chứa surface word hoặc lemma của governor/dependent. |

Đặc tả L dự kiến dùng làm cấu hình chính:

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
- Bổ sung nếu còn thời gian: full trừ P, full trừ M, full trừ D.
- Chuỗi tăng dần đo đóng góp theo thứ tự đã chọn. Morphology thêm sau POS không đo tác dụng khi morphology đứng riêng. Bỏ từng nhóm bổ sung góc nhìn có điều kiện; các nhóm vẫn có thể chứa thông tin trùng nhau.
- S+P+M+D gọi là non-lexical, không gọi là structural-only; structural-only đúng nghĩa chỉ gồm S.
- Việc chuyển ablation bỏ từng nhóm thành hạng mục bổ sung là điều chỉnh so với phạm vi dự kiến trong proposal. Nếu không chạy, ghi rõ phần còn thiếu và giới hạn trả lời RQ2 trong báo cáo/proposal khi cập nhật; không trình bày E2–E4 là hoàn thành toàn bộ.

## 4. Kiểm tra masking

Parser chạy trên câu gốc và được dùng chung cho các biến thể. Masking xảy ra trước bước học vocabulary và mã hóa lexical trong từng pipeline; vocabulary của biến thể masked chỉ học từ dữ liệu train đã áp dụng quy tắc masking.

Loại mọi lexical identity trực tiếp của token giao span khỏi surface word, lemma, n-gram, cửa sổ trái/phải, lexical governor/dependent và feature tổ hợp word–POS. Đặc tả D chính không chứa lexical governor/dependent; kiểm tra này ngăn việc vô tình bổ sung chúng sau đó.

Dùng marker cố định cho span đã che; không tạo n-gram giả nối hai phía qua span. Có test bằng span nhiều token và kiểm tra feature dictionary trước encoding. Test masking giữ cố định parse để xác minh đường truyền lexical; không yêu cầu P/M/D giữ nguyên khi thay đổi câu đầu vào và parse lại.

POS, morphology và dependency vẫn có thể tương quan với danh tính token. Vì vậy diễn giải kết quả là mức phụ thuộc vào lexical identity trực tiếp trong protocol này, không phải bằng chứng model hiểu ngữ pháp. Với insertion không có token span để che; báo cáo riêng nếu có dữ liệu, nếu không ghi N/A và support = 0.

## 5. Split, lựa chọn model và metric

- Chia xấp xỉ 70/15/15 theo essay hoặc nhóm duplicate; dùng author_id nếu đáng tin cậy. Chọn split theo tính hợp lệ và hỗ trợ lớp, không theo performance.
- Khóa quy tắc boilerplate/duplicate trước split. Không tự lấy ngưỡng lịch sử làm chính sách cuối. Báo cáo giới hạn phát hiện near-duplicate.
- Majority chọn lớp phổ biến nhất trong train.
- Bổ sung baseline chẩn đoán POS-only để trả lời liệu POS của token lỗi gần như quyết định loại lỗi hay không. Trước khi xem test, khóa POS tagger/version, cách tổng hợp POS span nhiều token, xử lý partial-token/missing/conflicting POS, và tie/fallback. Luật khởi điểm cần đánh giá là ADP → Prep, DET → ArtOrDet, các POS khác → lớp phổ biến nhất của train; đây là giả thuyết, không phải quy tắc đã được xác nhận. Báo cáo thêm bảng POS × nhãn lỗi trên train và so sánh Majority/POS rule/LR/Tree trên cùng split test.
- LR: C ∈ {0.01, 0.1, 1, 10}, class_weight ∈ {None, balanced}; kiểm tra convergence.
- Tree: max_depth ∈ {4, 8, 12}, min_samples_leaf ∈ {5, 20}, class_weight ∈ {None, balanced}; seed cố định.
- Chọn cấu hình theo Macro-F1 validation. Chốt thứ tự xử lý hòa điểm trước khi chạy; không chọn tùy ý sau khi xem test.
- Quyết định của protocol này: báo cáo chính dùng model fit trên train, không refit train+validation sau tuning. Đây là lựa chọn của kế hoạch, không diễn giải thành lệnh cấm refit nói chung. Bản proposal hiện tại, mục 8, có câu “Giữ nguyên train đã fit để báo cáo chính, tránh thay đổi protocol bằng refit tùy ý.” Quyết định này nhất quán với câu đó.
- Luôn tính metric với danh sách cố định ArtOrDet, Prep, Nn, Vform, SVA; zero_division = 0. Báo cáo Macro-F1, accuracy, macro precision/recall, precision/recall/F1/support từng lớp và confusion matrix.
- Bootstrap 1.000 lần theo nhóm split độc lập, có hoàn lại; nhóm được chọn nhiều lần thì các annotation của nó được tính với bội số tương ứng. Dùng cùng lần resample khi so sánh hai cấu hình và lưu seed.
- Nếu resample thiếu lớp, vẫn giữ đủ 5 nhãn và zero_division = 0; không âm thầm bỏ lần đó hoặc lấy lại mẫu. Ghi số/tỷ lệ lần thiếu mỗi lớp, số lần thiếu ít nhất một lớp và ảnh hưởng đến độ ổn định khoảng tin cậy. Báo cáo khoảng percentile 95% và paired difference.

## 6. Lưu trữ bền vững và tái lập

- Giữ corpus, văn bản sample, correction và artifact có nội dung nguồn ngoài Git công khai. Trong docs vẫn chỉ push proposal.md; protocol này là file local theo .gitignore hiện tại.
- Không coi việc có file trên một máy là đủ tái lập. Tạo gói reproducibility có phiên bản: source/code checksum, environment lock, review decisions, split manifest, seed, grouping policy, schema, feature specification, experiment configurations và artifact checksums.
- Review quyết định và split manifest tối thiểu chỉ chứa IDs, lý do và metadata cần thiết; phần bằng chứng có văn bản nguồn lưu riêng với quyền truy cập phù hợp. ID không được mặc định là dữ liệu có thể công khai khi chưa kiểm tra điều kiện nguồn.
- Code, cấu hình thí nghiệm không chứa corpus và script tái tạo được quản lý phiên bản bằng Git ngoài docs. Giữ chính sách Git hiện hành.
- Trước khi khóa split, chọn một nơi sao lưu riêng có kiểm soát truy cập, như ổ sao lưu thứ hai hoặc kho riêng phù hợp quyền sử dụng. Đồng bộ gói metadata tại mỗi milestone và thử khôi phục từ bản sao vào thư mục khác; so checksum, manifest và quyết định review.
- Kế hoạch không tự cấp quyền upload dữ liệu sang dịch vụ bên ngoài. Khi triển khai, dùng nơi lưu đã được người dùng cho phép; nếu chưa có, ghi trạng thái chưa hoàn thành lưu trữ bền vững, không tuyên bố đã tái lập đầy đủ.

## 7. Ngân sách và đầu ra bắt buộc

Sáu feature set chính, mỗi set 8 cấu hình LR và 12 cấu hình Tree: khoảng 120 lần fit một seed. Ba ablation bỏ P/M/D bổ sung khoảng 60 lần fit. Đo runtime trước khi cam kết phần bổ sung; ghi thay đổi ngân sách trước test và áp dụng nhất quán.

Error analysis sau test: tối đa 50 mẫu sai và 10 mẫu đúng, phân bổ theo nhãn/cặp nhầm; ghi số thực tế khi không đủ mẫu. Lưu parser evidence, ambiguity, mức chắc chắn và annotation_id. Correction chỉ được dùng ở bước phân tích sau dự đoán. Báo cả lỗi kỹ thuật parser và chất lượng ngữ nghĩa quan sát, không coi có cây parse là parse đúng.

Đầu ra bắt buộc: benchmark Majority/LR/Tree, ablation tăng dần và masking, bảng metric/test uncertainty, error analysis, báo cáo trả lời RQ1–RQ3, sáu slide và gói tái lập. TF-IDF, random baseline, prototype và ablation bổ sung được giảm trước; bảo đảm ngân sách test/báo cáo. Không âm thầm bỏ morphology/dependency khỏi full benchmark.
