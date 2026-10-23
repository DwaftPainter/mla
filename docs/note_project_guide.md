# Hướng dẫn toàn bộ project NUCLE

Cập nhật sau rà soát/tái lập 23/10/2026. [Report task rà soát](../reports/evaluation_followup/review_report.md) bổ sung kiểm raw → features → 52 fit → test replay và giới hạn audit; không xuất DOCX mới. Report task hiện hành: [evaluation_tasks_report.docx](../reports/summaries/evaluation/evaluation_tasks_report.docx). Các bảng validation bên dưới giữ số lịch sử, không phải test. File này giải thích pipeline, preparation v1/v2 và quan hệ giữa các file. Lịch thực hiện chỉ quản lý tại [note.md](../note.md). Các ví dụ câu dưới đây là tự tạo, không trích corpus. Tài liệu này thay hướng dẫn cũ có lịch sáu tuần và code sklearn minh họa không phản ánh implementation hiện hành.

## 1. Bài toán, RQ và trạng thái

Một sample là một annotation: **câu gốc + vị trí lỗi đã biết → một trong năm nhãn**. Nhãn gồm ArtOrDet (mạo từ/từ hạn định), Prep (giới từ), Nn (số danh từ), Vform (dạng động từ), SVA (hòa hợp chủ ngữ–động từ). Một câu có thể tạo nhiều sample. Không tự tìm vị trí lỗi, sinh câu sửa hay chấm điểm người học.

| RQ | Bằng chứng trước đợt task này | Bằng chứng bổ sung trong report task |
|---|---|---|
| RQ1: LR và Tree phân loại tốt đến đâu so với Majority? | Benchmark validation; thêm POS làm đối chứng | Đã có trong report task |
| RQ2: Lexical/POS/morphology/dependency ảnh hưởng thế nào? | Sáu biến thể tăng dần/masking/non-lexical cho LR và Tree | Full−P/M đã chạy, Full−D reuse, test đã chấm |
| RQ3: Nhóm nào khó phân biệt và đặc điểm nào liên quan? | Confusion, span/cặp nhãn/POS slices, audit 18 lỗi validation có hỗ trợ AI | Đã audit 40 ca test, có nhận xét và giới hạn |

Đã hoàn thành preparation, split theo nhóm, features, benchmark/ablation/masking validation, sensitivity v1/v2, thử BERT-small và chẩn đoán BERT. **Đã khóa, chấm test và bootstrap: LR Full test F1=0,8163; Tree Full=0,7996; POS=0,7106.** Dừng thử thêm BERT; giữ kết quả như thử nghiệm chưa thành công trong cấu hình/ngân sách đã dùng. Không suy ra ngữ cảnh vô ích.

| Kết quả validation | Macro-F1 | Diễn giải |
|---|---:|---|
| Majority | 0,1184 | Lớp đa số học từ train |
| POS rule | 0,7048 | Đối chứng từ loại mạnh |
| LR Full | 0,8067 | Ứng viên chính hiện tại |
| Tree Full | 0,7879 | Cùng biểu diễn Full với LR |
| Tree S+L+P+M | 0,7929 | Biến thể tốt hơn Tree Full trên validation, báo riêng |
| BERT-small, LR chọn, ba seed | 0,1984 ± 0,0030 | Trung bình và độ lệch chuẩn mẫu; không chọn riêng seed tốt nhất |

## 2. Sơ đồ mapping toàn dự án

Đường liền là luồng đã có; vùng H đã hoàn thành bằng module đánh giá cuối. Các tên trong sơ đồ được giải thích bằng đường dẫn đầy đủ ở bảng mapping phía sau.

~~~mermaid
flowchart TB
  subgraph SOURCE["A. Nguồn và audit — đã có"]
    RAW["data_set/raw/nucle3.2.sgml"]
    PARSE["nucle_sgml.py + parse_nucle.py<br/>essay / paragraph / annotation / offset"]
    UNDER["data_understanding.py<br/>interim/nucle_errors.csv"]
    QUALITY["data_quality.py<br/>reports/nucle_annotation_audit.csv"]
    REVIEW["interim/nucle_overlap_review_log_all.csv<br/>retain_original / unresolved"]
    RAW --> PARSE --> UNDER --> QUALITY
  end
  subgraph PREP["B. Preparation — không fit model, không chia split"]
    BUILD["prepare_nucle.py<br/>xác minh source / CSV / audit / review"]
    TOK["spacy.blank en + sentencizer<br/>spaCy 3.8.7"]
    V1["v1: biên câu từ sentencizer"]
    RULE["sentence_boundaries.py<br/>repaired_starts: luật text-only"]
    V2["v2: biên câu sau sửa"]
    ALIGN["construct_input<br/>paragraph → sentence offsets<br/>span / token / context alignment"]
    P1["processed/nucle_preparation_v1<br/>samples / sentences / provenance / manifest"]
    P2["processed/nucle_preparation_v2<br/>artifact chính + boundary_challenges"]
    CMP["compare_preparation.py<br/>reports/nucle_preparation_v2_comparison"]
    BUILD --> TOK
    TOK --> V1 --> ALIGN
    TOK --> RULE --> V2 --> ALIGN
    ALIGN --> P1
    ALIGN --> P2
    P1 --> CMP
    P2 --> CMP
  end
  RAW --> BUILD
  UNDER --> BUILD
  QUALITY --> BUILD
  REVIEW --> BUILD
  subgraph SPLIT["C. Grouping và split — partition đã cố định"]
    GROUP["group_split_nucle.py<br/>duplicate edges + boilerplate policy<br/>connected essay groups"]
    MAN["split/nucle_group_split<br/>essay_groups.csv + split_manifest.csv<br/>seed 20261013"]
    TRAIN["Train: 11052"]
    VAL["Validation: 2368"]
    TEST["Test: 2368<br/>đã chấm sau khóa cấu hình"]
    GROUP --> MAN
    MAN --> TRAIN
    MAN --> VAL
    MAN --> TEST
  end
  RAW --> GROUP
  P2 --> GROUP
  subgraph CLASSIC["D. Nhánh cổ điển — train và validation"]
    POS["nucle_pos.py<br/>Majority + POS baseline"]
    FEAT["nucle_features.py<br/>en_core_web_sm 3.8.0<br/>S/L/P/M/D; encoder fit train-only"]
    FX["features/nucle_features_v1<br/>Full: 16180 chiều"]
    MOD["nucle_logistic.py + nucle_tree.py"]
    BENCH["nucle_benchmark.py<br/>grid Full"]
    ABL["nucle_ablation.py<br/>6 variants đã chạy<br/>mask vocabulary fit riêng"]
    CR["reports/nucle_pos_baseline_v1<br/>nucle_feature_benchmark_solver_v2<br/>nucle_feature_ablation_v1"]
    FEAT --> FX --> BENCH --> CR
    FX --> ABL --> CR
    MOD --> BENCH
    MOD --> ABL
    POS --> CR
  end
  P2 --> POS
  P2 --> FEAT
  MAN --> POS
  TRAIN --> FEAT
  VAL --> FEAT
  subgraph AUDIT["E. Phân tích validation"]
    SPAN["nucle_span_analysis.py<br/>span / gold pairs / POS slices"]
    SENS["preprocessing_sensitivity_audit.py<br/>cùng IDs/split, encoder riêng v1/v2"]
    THREAD["logistic_thread_sensitivity.py<br/>refit LR theo threads"]
    AR["reports/nucle_span_diagnostics_v1,v2<br/>contextual_model_validation/audit"]
    SPAN --> AR
    SENS --> AR
    THREAD --> AR
  end
  CR --> SPAN
  P1 --> SENS
  P2 --> SENS
  MAN --> SENS
  FX --> THREAD
  subgraph BERT["F. BERT-small — đã thử; dừng mở rộng"]
    CACHE[".cache/huggingface<br/>google/bert_uncased_L-2_H-128_A-2"]
    BT["bert_span_classifier.py<br/>WordPiece + markers<br/>CLS fine-tune; chọn bằng validation"]
    BR["contextual_model_validation/bert<br/>config / history / checkpoint / predictions"]
    REPLAY["bert_checkpoint_verification.py"]
    READOUT["bert_span_readout.py<br/>CLS / span mean"]
    DIAG["bert_learning_diagnostics.py<br/>100 train / gradients / markers"]
    PAIR["bert_target_pair_probe.py<br/>50 cặp span cùng câu train"]
    DR["reports/bert_learning_diagnostics"]
    CACHE --> BT --> BR --> REPLAY
    CACHE --> DIAG
    BR --> DIAG
    READOUT --> DIAG --> DR
    BR --> PAIR --> DR
  end
  P2 --> BT
  MAN --> BT
  TRAIN --> DIAG
  TRAIN --> PAIR
  subgraph DOC["G. Báo cáo và tài liệu"]
    REPORT["build_contextual_validation_report.py<br/>build_bert_diagnostics_report.py<br/>build_ablation_notebook.py<br/>build_evaluation_tasks_report.py"]
    OUTPUT["docs: phương pháp/protocol<br/>reports/summaries: báo cáo<br/>reports: JSON/CSV/biểu đồ<br/>notebooks: trình bày kết quả"]
    GUIDE["note_project_guide.md: mapping<br/>note.md: lịch hiện hành<br/>reference.md: nguồn học thuật"]
    REPORT --> OUTPUT --> GUIDE
  end
  CR --> REPORT
  AR --> REPORT
  BR --> REPORT
  DR --> REPORT
  subgraph FUTURE["H. Đánh giá cuối — đã chạy"]
    EXTRA["Full−P / Full−M / Full−D<br/>LR + Tree, chọn trên validation"]
    LOCK["Khóa 18 pipeline LR/Tree + Majority/POS<br/>hash / config / metric / phép so sánh"]
    FINAL["Test → paired group bootstrap<br/>error analysis → bằng chứng cho RQ1–RQ3"]
    EXTRA --> LOCK --> FINAL
  end
  ABL --> EXTRA
  TEST --> FINAL
  FINAL --> OUTPUT
~~~

ALIGN được vẽ chung vì hai policy dùng cùng thuật toán. Khi chạy thật, v1 và v2 là **hai run, hai thư mục riêng**, không trộn sentence giữa hai version. Validation đi qua feature encoder để transform, không được dùng để fit. Không có luồng test vào tuning hiện tại.

## 3. Preparation v1/v2: đầu vào và các hàm chính

Preparation chuyển annotation tham chiếu paragraph thành sample có sentence và offset cục bộ, giữ đường truy vết nguồn. Chưa tạo vector 16.180 chiều, chưa gán POS/dependency, chưa huấn luyện.

| File/hàm | Công việc |
|---|---|
| src/data/nucle_sgml.py — NucleSGMLParser, load_nucle | Đọc DOC, TITLE/P và MISTAKE/TYPE/CORRECTION/COMMENT; giữ thứ tự paragraph |
| src/data/parse_nucle.py — documents_to_dataframe | Dựng bảng annotation, tái tạo span nguồn và kiểm offset |
| src/data/data_understanding.py | Xuất data_set/interim/nucle_errors.csv, thống kê corpus |
| src/data/data_quality.py | So CSV/SGML, gắn cờ offset/overlap/no-op/long-span; không tự sửa annotation |
| src/data/prepare_nucle.py — prepare, read_metadata | Kiểm nguồn, CSV, audit và review trước khi dựng sample |
| make_nlp, segment_paragraph | Khởi tạo tokenizer/sentencizer; chia paragraph theo policy |
| construct_input | Gán annotation vào câu, đổi offset, căn token và context |
| build_dataset | Phân loại retained/out_of_scope/not_constructible và ghi provenance |
| export_dataset | Xuất staging, kiểm hash, đổi tên sang output mới; không overwrite |

Input mặc định gồm:

- data_set/raw/nucle3.2.sgml.
- data_set/interim/nucle_errors.csv.
- reports/nucle_annotation_audit.csv.
- data_set/interim/nucle_overlap_review_log_all.csv.
- uv.lock, pyproject.toml và hash code để ghi manifest.

prepare kiểm SHA-256 source đúng giá trị kỳ vọng, đọc SGML lại để đối chiếu CSV, tính lại audit để tránh metadata stale. Review phải có ID duy nhất, đúng essay/label/source checksum; decision chỉ là retain_original hoặc unresolved, data_action phải là keep_original_annotation_unchanged. Review không cho phép âm thầm sửa câu/nhãn.

**Mốc offset:** parser dùng HTMLParser(convert_charrefs=True). “Giữ nguyên text” nghĩa là giữ paragraph sau khi parser hiện hành đọc SGML, không phải từng byte của file markup. Từ paragraph đó trở đi không strip, lowercase, sửa chính tả hoặc chuẩn hóa Unicode. Offset là chỉ số ký tự Python, không phải byte offset hoặc token index.

## 4. Preparation v1: thuật toán cụ thể

make_nlp yêu cầu spaCy 3.8.7, tạo spacy.blank("en") và thêm sentencizer với punct_chars = [".", "!", "?", "…"], overwrite=True. **Không nạp en_core_web_sm ở bước này.**

segment_paragraph với policy nucle-preparation-v1:

1. Tokenize/sentencize từng paragraph độc lập, không nối paragraph.
2. Với mỗi doc.sents, lấy vị trí ký tự của token đầu không phải whitespace. Bỏ sentence chỉ có whitespace.
3. Biên đầu đặt bằng 0 để giữ khoảng trắng đầu paragraph. Các biên sau lấy từ starts; thêm len(paragraph) làm biên cuối.
4. Mỗi câu là slice paragraph[s:e], đầu có tính/cuối không tính. Không strip; whitespace trước start câu sau thuộc slice câu trước.
5. Token không-whitespace được lưu với text, index, start/end tương đối trong sentence. Punctuation và stopwords vẫn giữ.
6. Kiểm ghép các slice trả đúng paragraph và từng token khớp substring nguồn.

~~~text
paragraph
  → tokenizer + sentencizer
  → starts của token đầu mỗi câu
  → [0, start_2, ..., len(paragraph)]
  → các slice [s,e)
  → token offsets tương đối theo từng slice
~~~

V1 là mốc segmentation xác định theo tokenizer/sentencizer đã pin. Không tương đương paragraph.split(".") vì tokenizer có quy tắc riêng. Tuy vậy, sentencizer không hiểu sâu citation, abbreviation, ellipsis và ngoặc/nháy trong câu người học; đây là lý do tạo v2.

## 5. Preparation v2: sửa chính xác những gì?

V2 chạy cùng tokenizer/sentencizer, nhưng dùng repaired_starts(doc) từ src/data/sentence_boundaries.py thay cho starts thô. Hàm chỉ nhận doc/text, **không nhận annotation offsets, labels, correction hay model predictions**. Quy tắc text-only là heuristic, không được fit bằng điểm validation.

### 5.1 Thứ tự các luật trong repaired_starts

| Thứ tự | Điều kiện | Thao tác và giới hạn |
|---|---|---|
| 1. Ngoặc/nháy mở | Token ngay trước start là (, [, {, “, ‘; hoặc dấu nháy kép thẳng được xét là mở bằng đếm chẵn/lẻ số nháy trước đó | Dời start lùi để ký hiệu mở về câu sau; không bảo đảm hiểu mọi kiểu lời dẫn |
| 2. Nhận diện citation | Ngoặc tròn/vuông không lồng, tối đa 300 ký tự, không newline; có năm 18xx/19xx/20xx, n.d., URL hoặc tham chiếu số ngắn; không chứa ! hay ? | Xóa start nằm trong citation; không xem mọi ngoặc là citation |
| 3. Tiếp diễn sau citation abbreviation | Citation kết thúc kiểu et al.) hoặc n.d.); token sau bắt đầu lowercase, ở giữa chỉ whitespace | Xóa start giả ở token tiếp diễn |
| 4. Sau citation | Sau đóng ngoặc là whitespace/dấu . ! ?, rồi từ viết hoa theo regex | Thêm start ở từ đó, kể cả khi biên rơi trong token do dấu đóng ngoặc dính từ; vẫn là heuristic |
| 5. Abbreviation | 40 ký tự trước start khớp etc., et al., n.d.; token sau lowercase hoặc số | Bỏ biên trước token tiếp diễn; không triển khai mọi abbreviation tiếng Anh |
| 6. Ellipsis | Ba dấu chấm trở lên hoặc …, ngoài citation; chỉ có whitespace trước token tiếp | Thêm biên nếu token viết hoa và không thuộc subordinators; nếu không thì bỏ start ở token đó |
| 7. Chuẩn hóa starts | Bỏ start token đầu rồi chèn 0, sort các start dương | Giữ whitespace và thứ tự intervals; không đổi paragraph text |

Subordinators: because, although, though, while, if, unless, until, when, where, since, as, once, that, which, whose, whom, who. Không dùng danh sách này như bằng chứng mọi sentence boundary đều đúng ngữ nghĩa.

### 5.2 Biên v2 có thể cắt token nguồn

Ví dụ dấu đóng ngoặc và từ tiếp theo dính nhau có thể khiến start mới nằm giữa token spaCy. segment_paragraph giữ chính xác phần giao token với sentence, không sửa text hoặc tokenizer.

Với token nguồn [a,b), sentence [s,e):

\[
a'=\max(a,s),\quad b'=\min(b,e).
\]

Fragment text là paragraph[a':b']; offset trong sentence là [a'−s,b'−s). Con trỏ chỉ tăng khi token đã được tiêu thụ hết; phần còn lại có thể được lưu trong sentence kế tiếp.

**Tokenizer không đổi cấu hình; token lưu trong artifact có thể đổi thành fragment vì biên câu mới.** Đây cũng là lý do không khẳng định mọi token span v1/v2 giống nhau.

### 5.3 Metadata v2 và hai trường hợp unresolved

V2 dùng schema_version=2, thêm boundary_flags/boundary_review_status và boundary_challenges.jsonl. V1 manifest có schema_version=1; schema cơ bản sentence/span vẫn tương thích.

src/data/boundary_challenges.py gắn metadata cho annotation 31297 và 39538; hai ca này vẫn retained, boundary review unresolved. Metadata này không điều khiển repaired_starts và không được đưa vào predictors. Hai ca boundary unresolved khác với 96 annotation có review_status unresolved từ log quality.

## 6. Các bước chung sau segmentation

### 6.1 Đổi offset paragraph sang sentence

Nếu sentence = paragraph[s:e] và lỗi nguồn là [p_start,p_end), phải tìm một interval chứa toàn bộ lỗi rồi tính:

\[
u=p_{start}-s,\qquad v=p_{end}-s.
\]

\[
\text{sentence}[u:v]=\text{paragraph}[p_{start}:p_{end}]=\text{error_span}.
\]

error_start=u, error_end=v. Luôn kiểm 0 ≤ u ≤ v ≤ len(sentence).

Ví dụ tự tạo: paragraph "Hi. She go." có intervals [0,4), [4,11). Lỗi "go" ở paragraph [8,10) thành [4,6) trong sentence "She go.". Nếu segmentation đổi s thì offsets cục bộ đổi; vị trí lỗi trong paragraph không đổi.

Với insertion p_start=p_end=p: đúng biên hai câu thì gán câu sau; p=len(paragraph) gán câu cuối với offset bằng len(sentence). Điểm nằm trong whitespace trước biên vẫn thuộc interval trước. Không suy insertion bằng độ dài correction; nó được xác định từ offsets.

### 6.2 Span/token/context alignment

Token [a_i,b_i) thuộc span không rỗng [u,v) khi:

\[
a_i<v \quad\land\quad u<b_i.
\]

Context trái lấy tối đa hai token gần nhất có b_i≤u; phải lấy tối đa hai token đầu có a_i≥v. Chỉ lấy trong cùng sentence; không dùng token giao span làm context.

| Alignment | Quy tắc | Ví dụ tự tạo với "Dogs run." |
|---|---|---|
| exact | Span khớp start token đầu/end token cuối | [0,4) là "Dogs" |
| partial_token | Endpoint nằm trong token; ưu tiên kiểm trước exact | [3,4) là "s", nhưng token biểu diễn là toàn "Dogs" |
| whitespace_edge | Có token giao, không cắt token, nhưng không exact | [0,5) là "Dogs " |
| insertion_boundary | Span rỗng tại biên token/whitespace | [4,4) sau "Dogs", không có span token |

Partial-token giữ nguyên gold offsets; không nới annotation. Token dùng làm feature có thể rộng hơn substring lỗi. Insertion giữa token bị từ chối, không tự dời; non-empty span chỉ có whitespace bị từ chối vì không giao token.

### 6.3 Giữ mẫu và ghi lý do

Thứ tự xét: nhãn ngoài năm lớp → out_of_scope; offset sai → invalid_offset; qua paragraph → cross_paragraph; paragraph rỗng → empty_paragraph; span không nằm trong một câu → cross_sentence; không giao token → no_token_overlap; insertion giữa token → insertion_inside_token; còn lại retained nếu invariant đúng.

Out-of-scope không có nghĩa annotation sai. Reason không dựng được là giới hạn biểu diễn một câu, không phải yêu cầu sửa corpus. Checksum/CSV/audit mismatch, ID trùng, slice sai hoặc input/code đổi trong run làm **dừng toàn run**, không bị che thành một sample rejected.

### 6.4 Artifact và ID mapping

Mỗi run xuất một thư mục riêng: data_set/processed/nucle_preparation_v1 hoặc nucle_preparation_v2.

| File | Nội dung và nơi dùng |
|---|---|
| samples.jsonl | Một dòng mỗi annotation retained: sentence, offsets, span, label, IDs, alignment/context. Chưa phải vector predictor |
| sentences.jsonl | Sentence/token dùng chung; chỉ chứa câu có sample retained, không phải mọi câu corpus |
| provenance.jsonl | Mọi annotation nguồn: paragraph offsets, label gốc, status/reason, flags/review, liên kết sample/sentence |
| rejected.jsonl | Provenance không retained, kể cả out-of-scope; không xóa nguồn |
| input_groups.jsonl | Input trùng exact nếu có; hiện không có nhóm |
| boundary_challenges.jsonl | V2: fixture hai ca boundary, có text nên local |
| summary.json | Counts, label/alignment/review/quality distributions |
| manifest.json | Policy/schema, input/code/output hashes, Git commit, tokenizer config/hash, Python/spaCy, thời điểm |
| manifest.sha256 | Digest canonical manifest, không phải tự nhúng hash vào manifest |

~~~text
sample_id   = source_sha256 : annotation_id
sentence_id = source_sha256 : essay_index : paragraph_index : sentence_index
input_sha256 = SHA256(UTF8(canonical_json([sentence, error_start, error_end])))
~~~

sample_id giữ ổn định giữa v1/v2. sentence_id có thể đổi khi segmentation đổi; cùng sentence_id cũng chưa bảo đảm cùng nội dung ở hai preparation. Khi tham chiếu phải kèm version/manifest digest. Input hash không chứa label/correction; code kiểm tuple thật để phát hiện collision.

export_dataset ghi staging cùng parent, kiểm lại hash nguồn/code rồi rename sang output hoàn chỉnh; từ chối output đã tồn tại. Manifest chứa timestamp/commit/code hashes nên digest manifest của run mới có thể khác dù sample data giống nhau.

## 7. V1 và v2 thực tế khác nhau bao nhiêu?

Nguồn: summary.json trong hai preparation và [comparison_summary.json](../reports/nucle_preparation_v2_comparison/comparison_summary.json). Đây là so sánh cơ học dataset, không phải chấm model test.

| Chỉ số | V1 | V2 |
|---|---:|---:|
| Annotation nguồn | 44.912 | 44.912 |
| Ngoài năm nhãn | 29.124 | 29.124 |
| Target retained | 15.788 | 15.788 |
| Target không dựng được | 0 | 0 |
| Sentence có sample retained | 11.262 | 11.272 |
| Exact alignment | 14.445 | 14.447 |
| Partial-token | 1.132 | 1.130 |
| Whitespace-edge | 211 | 211 |
| Span rỗng | 0 | 0 |

Trong 15.788 mẫu: **544 input đổi**, 15.244 giữ nguyên; 533 sentence ID đổi; 30 mẫu đổi context token; hai mẫu đổi token span. Không đổi source offsets của span, error_span, label hoặc retention. Các tập thay đổi chồng lấp, không cộng lại.

Input đổi theo nhãn: ArtOrDet 237, Nn 120, Prep 88, SVA 61, Vform 38. Phân bố retained không đổi: ArtOrDet 6.640, Prep 2.413, Nn 3.768, Vform 1.443, SVA 1.524.

compare_preparation.py kiểm hash và alignment toàn bộ mẫu; xuất diff.jsonl, regression_cases.jsonl, fresh_packet.jsonl và fresh_selection.json cho 50 mẫu review mới. Các packet có text nên local. Thêm sentence hay giảm partial-token không tự chứng minh v2 đúng ngữ nghĩa; hai ca unresolved vẫn giữ.

**Không có span rỗng không có nghĩa không có lỗi thiếu từ.** Audit validation đã thấy lỗi thiếu determiner được annotation bằng span danh từ không rỗng. Không chuyển thành insertion dựa trên correction.

Sensitivity riêng trên cùng IDs/split: 381 train và 96 validation input thay đổi. Encoder fit riêng cho từng preparation; LR C=0,1/class_weight=None cố định, không retune. Macro-F1 validation v1≈0,80694; v2≈0,80782. Không lấy chênh lệch nhỏ này làm căn cứ chọn version. Refit v2 khác hai prediction so baseline cũ; chẩn đoán cùng feature matrix với 4/10 threads tái tạo hai bộ prediction tương ứng. Giữ điểm gốc 0,80668 trong bảng baseline.

### Lệnh tái tạo preparation

Chỉ chạy khi cần rebuild, trong môi trường có spaCy 3.8.7; input audit/review phải có sẵn và output phải chưa tồn tại:

~~~powershell
.venv\Scripts\python.exe src/data/prepare_nucle.py --policy v1 --output data_set/processed/nucle_preparation_v1_rebuild
.venv\Scripts\python.exe src/data/prepare_nucle.py --policy v2 --output data_set/processed/nucle_preparation_v2_rebuild
.venv\Scripts\python.exe src/data/compare_preparation.py --old data_set/processed/nucle_preparation_v1_rebuild --new data_set/processed/nucle_preparation_v2_rebuild --output reports/nucle_preparation_comparison_rebuild
~~~

Compare còn dùng data_set/interim/nucle_boundary_decisions.jsonl để loại mẫu đã review khỏi packet mới. Rebuild không tự thay dataset chính hoặc split. Không chạy lại vào thư mục chuẩn để ghi đè.

## 8. Grouping, split và features sau preparation

### 8.1 Grouping và split

src/data/group_split_nucle.py dùng source/preparation nối essay thành connected components: full essay trùng exact/normalized, paragraph normalized trùng, hoặc model input exact trùng. Chuẩn hóa NFC + casefold + gộp whitespace chỉ dùng tìm duplicates, không thay sentence predictor.

Boilerplate được miễn nối khi có ở ít nhất 10 essay, tối đa 10 từ/80 ký tự và không có target annotation trong mọi occurrence. Trong 62 nhóm paragraph normalized, 7 được miễn và 55 tạo edges. Exact/normalized duplicate counts chồng lấp. Kết quả 1.280 groups trên 1.397 essay, group lớn nhất 34 essay/338 sample.

data_set/split/nucle_group_split chứa essay_groups.csv (essay→group), split_manifest.csv (sample→partition/group), grouping_policy.json, split_summary.json và manifest. Seed 20261013: train 11.052, validation 2.368, test 2.368. Không chia lại theo annotation hoặc đổi split khi so BERT/v1/v2. Chưa phát hiện tự động paraphrase/near-duplicate là một giới hạn.

### 8.2 Preparation NLP khác feature NLP

Preparation dùng blank English + sentencizer. Feature/POS dùng spaCy 3.8.7 + en_core_web_sm 3.8.0. **Code hiện tại parse lại sentence**, không trực tiếp tái dùng token list preparation như đặc tả ban đầu từng dự kiến. Audit train/validation xác nhận offsets và số token giao span không bất đồng; không chứng minh mọi POS/parse đúng ngữ nghĩa hoặc toàn bộ tokenization giống tuyệt đối.

| Nhóm | Thông tin |
|---|---|
| S | Độ dài, vị trí tương đối và cấu trúc span |
| L | Lexical span/context tối đa hai token mỗi phía, unigram/bigram |
| P | POS signatures span/context |
| M | Number, Person, Tense, VerbForm và missing indicators |
| D | Dependency relation, head/dependent POS, khoảng cách và thông tin chủ ngữ |

src/features/nucle_features.py: extract_features tạo dictionary, FeatureEncoder fit trên train rồi transform validation. Full có 16.180 chiều, artifact data_set/features/nucle_features_v1. **v1 ở tên feature artifact không có nghĩa preparation v1**; pipeline chính dùng preparation v2.

Giới hạn 500 là kênh pooled n-gram, không phải mọi lexical feature: slot trái/phải có vocabulary riêng. Key L|span_ngrams chứa candidates của cả span và context. Numeric giữ giá trị gốc, không z-score. LR có dependency từ sentence, không chỉ nhìn ±2 từ.

src/features/nucle_pos.py: ADP→Prep, DET→ArtOrDet; các signatures khác dùng majority label theo signature học trên train, thiếu/unseen fallback Majority train. Không học bảng luật từ validation.

## 9. Mapping nhánh mô hình cổ điển

| Code | Vai trò / artifact |
|---|---|
| src/models/nucle_logistic.py | Implementation LR của project, không phải code sklearn minh họa cũ |
| src/models/nucle_tree.py | Implementation Gini Tree |
| src/training/nucle_benchmark.py | Grid Full → reports/nucle_feature_benchmark_solver_v2 |
| src/training/nucle_ablation.py | Sáu variants → reports/nucle_feature_ablation_v1 |
| src/evaluation/metrics.py | Metric nhánh cổ điển theo năm nhãn |
| src/evaluation/nucle_span_analysis.py | Span/cặp nhãn/POS slices → reports/nucle_span_diagnostics_v1 và v2 |
| src/evaluation/preprocessing_sensitivity_audit.py | Sensitivity v1/v2, input checks, audit packet → reports/contextual_model_validation/audit |
| src/evaluation/logistic_thread_sensitivity.py | Refit cùng feature matrix theo threads để kiểm khác biệt số học |

Grid LR: C∈{0,01;0,1;1;10} × class_weight∈{None,balanced}. Tree: depth∈{4,8,12} × min_leaf∈{5,20} × class_weight∈{None,balanced}. Chọn macro-F1 validation. LR dừng change tolerance không đồng nghĩa gradient-converged; báo đúng tiêu chí và loại cấu hình không đủ điều kiện theo protocol.

| Variant đã chạy | Câu hỏi nó hỗ trợ |
|---|---|
| S+L | Mốc cấu trúc + lexical |
| S+L+P | Thêm POS giúp bao nhiêu? |
| S+L+P+M | Thêm morphology sau POS giúp bao nhiêu? |
| S+L+P+M+D | Full: thêm dependency |
| Full_mask_span_lexical | Phụ thuộc lexical identity trực tiếp của span |
| S+P+M+D_nonlexical | Bỏ toàn bộ lexical; không gọi structural-only |

Masking xóa lexical identity của mọi token giao span **trước khi fit vocabulary masked riêng trên train**, giữ parse gốc/context, không parse câu đã thay marker và không nối bigram qua span bị che. Variant được tune riêng trong cùng grid, nên không coi chênh lệch là hiệu ứng feature cô lập tại cùng hyperparameters.

Full−P/M đã chạy; Full−D reuse S+L+P+M. Test/CI ở report task, không thay số validation lịch sử. Cặp Vform/SVA và Nn/ArtOrDet chọn theo gold label nhưng giữ predictions năm lớp; dự đoán ngoài cặp vẫn sai. POS slices phải báo riêng, không đồng nhất cặp gold với mọi span cùng POS.

Validation có 0 span rỗng, 2.114 span một token, 254 nhiều token. LR macro-F1 năm lớp tương ứng N/A, khoảng 0,8154 và 0,5195. Không suy quan hệ nhân quả từ chiều dài; span rộng có thể chứa nhiều cấu trúc nhưng chỉ một nhãn.

## 10. Mapping BERT và chẩn đoán

src/training/bert_span_classifier.py dùng google/bert_uncased_L-2_H-128_A-2: 2 layers, hidden 128, 2 heads, không phải BERT-base. model_source.json ở reports/contextual_model_validation lưu nguồn/revision; cache ở .cache/huggingface.

Input BERT là WordPiece, không phải vector S/L/P/M/D. encode_span tách sentence theo character offsets thành trái/span/phải rồi tokenize riêng:

~~~text
[CLS] left [SPAN_START] error_span [SPAN_END] right [SEP]
~~~

Max length 128; context budget = 128−4−số subword span. Giữ toàn span, lấy cuối phần trái và đầu phần phải; span vượt budget thì báo lỗi. Thêm marker vào tokenizer và resize embeddings. Có 5 train/6 validation bị cắt context, không cắt target. BERT tokenizer khác tokenizer preparation và feature parser.

Benchmark CLS: batch32, tối đa5 epoch, LR 3e−5/5e−5, dropout0,1, weight decay0,01 với ngoại lệ bias/LayerNorm, warmup10%, linear decay, clip1, CPU4 threads. Hai LR ở seed đầu; LR chọn 5e−5 thêm hai seed, tổng bốn run. Không dùng test.

| Code / artifact | Vai trò |
|---|---|
| src/evaluation/bert_checkpoint_verification.py | Nạp lại ba checkpoint chọn, xác nhận predictions validation khớp |
| src/models/bert_span_readout.py | CLS hoặc mean hidden bên trong markers; dùng chung pooler/head, empty span fallback marker start |
| src/evaluation/bert_learning_diagnostics.py | Input, gradient, optimizer coverage, parameter updates, marker sensitivity, 100 mẫu train |
| src/evaluation/bert_target_pair_probe.py | 50 cặp target khác nhãn trong cùng câu train |
| reports/contextual_model_validation/bert | Config, history, checkpoint, validation predictions/selection |
| reports/bert_learning_diagnostics | Input/gradient/probe summaries, train histories, checkpoints, bảng và hình |

Cả ba run nhỏ đạt accuracy100% trên 100 train cân bằng: CLS5e−5 có CE0,1515 sau400 updates; CLS1e−3 có CE0,0073 sau100; span mean1e−3 có CE0,0104 sau75. Hai run LR lớn đạt tiêu chí loss chặt; run LR nhỏ nhớ nhãn nhưng chưa đạt tiêu chí đó. Dropout/weight decay bằng0 và lịch học khác benchmark; không dùng kết quả học thuộc làm validation score hoặc khẳng định LR cao sẽ giải quyết toàn bộ vấn đề.

Không thấy đứt gradient trong probes. Final target hidden gradient bằng0 ở CLS là hệ quả đọc riêng CLS, không chứng minh target không ảnh hưởng encoder. Checkpoint cũ chỉ đúng cả hai ở2/50 cặp target cùng câu, là stress test có chủ đích, không ước lượng prevalence. Checkpoint span mean phải nạp với wrapper/mode tương ứng, không dùng default CLS loader để diễn giải.

Kết luận hiện tại: BERT-small có thể học tập nhỏ, benchmark hiện chưa thành công; **dừng mở rộng BERT**, giữ bằng chứng và giới hạn trong report.

## 11. Mapping tài liệu, reporting và môi trường

| File/thư mục | Trách nhiệm |
|---|---|
| docs/proposal.md | RQ1–RQ3, thiết kế gốc; phần dự kiến không phải kết quả |
| docs/methodology/preparation.md | Đặc tả có v1 lịch sử và bổ sung v2; đối chiếu implementation/manifest |
| docs/methodology/experiment_protocol.md | Quy tắc train-only, tuning, metrics, audit sai khác; cần đồng bộ phạm vi mới khi khóa |
| docs/methodology/implementation_data_audit.md | Vocabulary, scaling, reparse, masking, threads, 18 ca review |
| docs/bert_experiment_protocol.md | Protocol benchmark BERT đã chạy, giữ nguyên lịch sử |
| docs/bert_learning_diagnostics_protocol.md | Thiết kế gradient/học thuộc train-only |
| docs/bert_target_pair_probe_protocol.md | Thiết kế probe cùng câu trước lượt probe |
| reports/summaries/data/data_preprocessing_report.docx | Báo cáo dữ liệu/preparation |
| reports/summaries/classical/modeling_ablation_results_report.docx | Báo cáo model/ablation lịch sử |
| reports/summaries/contextual/contextual_model_validation_report.docx | So sánh validation, sensitivity và BERT |
| reports/summaries/contextual/bert_learning_diagnostics_report.docx | Chẩn đoán BERT, kết quả và giới hạn |
| src/reporting/build_ablation_notebook.py | Dựng notebook từ artifact |
| src/reporting/build_contextual_validation_report.py | Bảng/hình/report validation |
| src/reporting/build_bert_diagnostics_report.py | Bảng/hình/report diagnostics |
| src/reporting/update_validation_documentation.py | Script cập nhật tại mốc cũ; không chạy lại mù quáng vì có nội dung trạng thái lịch sử |
| notebooks/03_data_visualization.ipynb, notebooks/04_feature_ablation_report.ipynb | Trình bày dữ liệu/kết quả, không thay entry point train |
| docs/literature/related_work.md, docs/literature/contextual_models_related_work.md, reference.md | Nguồn và tổng hợp học thuật; không có task sửa Excel/PRISMA |
| docs/README.md, docs/status.md | Chỉ mục và snapshot tiến độ; đề xuất thử tiếp BERT cũ được thay bằng note hiện hành |
| note.md | Lịch và đầu ra tuần 2 hiện hành |
| pyproject.toml, uv.lock | Môi trường nền |
| requirements-bert.txt | Dependencies bổ sung BERT/DOCX; không coi lock nền bao gồm tất cả |
| reports/contextual_model_validation/environment.txt | Freeze môi trường thực tế của lượt chạy |

Các báo cáo để đọc được gom tại [reports/summaries](../reports/summaries/README.md). Hai báo cáo contextual validation và BERT diagnostics hiện chỉ có DOCX; bản Markdown tương ứng không có trong workspace. Dùng DOCX và JSON/CSV artifacts hiện có. Bản đồ toàn bộ artifact xem [reports/README.md](../reports/README.md).

## 12. Đánh giá cuối đã thực hiện và quy tắc sử dụng

Theo [note.md](../note.md), các bước thí nghiệm đã thực hiện; hạng mục5 được thu hẹp thành báo cáo task:

1. Full−P/M/D cho LR/Tree trên validation rồi khóa cấu hình/hash/metric.
2. Test 18 pipeline LR/Tree cùng Majority/POS.
3. Paired bootstrap 1.000 lần theo group, CI95% và chênh lệch.
4. Phân tích lỗi test và tổng hợp BERT đã có.
5. Báo cáo task và kiểm tái lập; final report/slide tổng kết để lại bước sau.

Module mới: src/training/nucle_final_selection.py; src/evaluation/nucle_final_evaluation.py, nucle_final_error_audit.py, verify_final_results.py; src/reporting/build_evaluation_tasks_report.py. Artifact: reports/final_evaluation/. Test/CI/audit đã có; BERT không chạy thêm. Fresh venv chỉ kiểm inference từ encoded test/metric, chưa tái chạy từ raw.

Quy tắc xuyên suốt:

- Label chỉ làm target/tính metric; IDs để join/group/trace. Correction, label, IDs và review flags không vào predictors.
- Encoder fit train-only; không đổi feature/seed hoặc tune lại dựa trên test.
- Partition lấy từ manifest cố định. Bootstrap theo connected group để giữ phụ thuộc trong essay và giữa các essay đã nối.
- Code, tests, docs, note/reference, các báo cáo để đọc và bảng kết quả được chọn đã được cho phép đưa lên Git. Dữ liệu/câu nguồn, fixture NUCLE, private audit packet, checkpoint/encoder/cache và bản sao lịch sử giữ local theo .gitignore. Xem phạm vi tại reports/README.md.
- Không overwrite artifact chuẩn; rebuild vào output mới và kiểm hash/config/metric. Hash đúng không đồng nghĩa parser đúng ngữ nghĩa.
- Hoàn tất cần test/CI/error analysis trả lời RQ1–RQ3, cấu hình có truy vết và gói tái lập đã kiểm chứng. Đã có report/test/CI/audit và tái lập raw → 52 fit → test replay trong môi trường riêng. Review chuyên gia và rerun toàn grid lịch sử vẫn chưa thực hiện; xem report task rà soát mới.
