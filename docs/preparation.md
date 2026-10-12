# Đặc tả NUCLE Data Preparation v2

Cập nhật: 12/10/2026. Policy hiện hành: `nucle-preparation-v2`. Đã chạy trên nguồn có quyền sử dụng được xác nhận và giữ đủ **15.788/15.788 mẫu mục tiêu**; xem [trạng thái dự án](status.md) và [báo cáo tổng quát](../notebooks/03_data_visualization.ipynb).

Tài liệu này chốt schema, offset, căn chỉnh token và cách ghi mẫu không dựng được. Các nguyên tắc nguồn và schema bên dưới vẫn áp dụng cho v2; quy tắc tách câu v1 ở mục 2 là lịch sử. Quy tắc v2 dưới đây thay thế phần segmentation cũ. Chưa khóa duplicate grouping, split hay benchmark.

## Quy tắc segmentation hiện hành v2

- Dùng `spacy==3.8.7`, `spacy.blank("en")` với sentencizer đã chốt; quy tắc bổ sung chỉ đọc paragraph nguồn, không đọc nhãn, correction hoặc quyết định review.
- Giữ nguyên mọi ký tự, khoảng trắng và offset nguồn. Sentence intervals liên tiếp phủ đúng toàn paragraph; ghép các lát cắt phải bằng nguyên văn paragraph.
- Citation parenthetical có dấu năm, `n.d.`, URL hoặc tham chiếu số gắn với câu trước; sau dấu đóng citation, bắt đầu câu mới trước từ hoa tiếp theo, kể cả dấu câu và chữ bị dính trong cùng spaCy token như `2009).The`.
- `et al.) for` và `n.d.) for` là phần tiếp tục câu, không tách tại dấu chấm của abbreviation. Boundary do sentencizer tạo bên trong citation được bỏ. Sau ellipsis, chỉ tạo boundary trước từ hoa độc lập; từ nối phụ thuộc được giữ để review nếu mơ hồ.
- Boundary có thể rơi giữa các ký tự của token spaCy khi nguồn dính dấu câu với chữ. Artifact lưu các mảnh token theo đúng lát cắt nguồn. Hai span mục tiêu nằm tại kiểu boundary này có biểu diễn token khác v1; offset ký tự và error span vẫn nguyên vẹn.
- Không điều chỉnh label, error span, correction hoặc source offset. Annotation qua sentence boundary vẫn theo reason/status priority đã định ở mục 6.

V2 giữ 15.788 mẫu, đổi input câu cho 544 mẫu và sentence ID cho 533 mẫu so với v1; offset span tuyệt đối và metadata nguồn giữ nguyên. Schema v2 thêm `boundary_flags` và `boundary_review_status` cho sample/provenance. Hai trường hợp mơ hồ (31297, 39538) vẫn retained, được gắn cờ unresolved và xuất ra `boundary_challenges.jsonl`; regression fixtures nằm trong `tests/fixtures/nucle_boundary_challenges.jsonl`. Packet 50 mẫu mới đã được rà soát hỗ trợ bằng AI; 57 boundary/preparation tests pass. Xem [comparison report](../reports/nucle_preparation_v2_final_review7_2026_10_12/).

## 1. Đầu vào và nguyên tắc bất biến

- Nguồn hiện hành: `data_set/raw/nucle3.2.sgml`, SHA256 `8441a60060730bdc6af1e79a064c68441ecb24a66da91ea2aab0061a64a130ac`.
- Dùng `load_nucle` và `documents_to_dataframe` hiện có. Giữ nguyên Understanding CSV, không viết lại parser hoặc đánh lại annotation_id sau khi lọc.
- `annotation_id` hiện là số thứ tự toàn corpus bắt đầu từ 0; chỉ duy nhất trong cùng source checksum. `essay_index`, `mistake_index` và chỉ số paragraph cũng bắt đầu từ 0.
- Paragraph chuẩn là chuỗi do parser hiện hành trả về: TITLE và P theo thứ tự nguồn; parser đã xử lý newline định dạng ở hai đầu và việc đọc file chuyển newline theo cơ chế Python. Offset này không phải byte offset trong SGML. Preparation không chuẩn hóa thêm Unicode, khoảng trắng, xuống dòng, entities, chữ hoa hoặc dấu câu.
- Mọi offset sử dụng chỉ số Python `str` (Unicode code point), nửa mở `[start, end)`. Không dùng byte, UTF-16 code unit hoặc số ký tự hiển thị làm offset.
- Nhãn mục tiêu theo thứ tự cố định: `ArtOrDet`, `Prep`, `Nn`, `Vform`, `SVA`; identity mapping từ TYPE. Mỗi annotation đủ điều kiện thành đúng một sample, một nhãn.
- Giữ nguyên cả 96 ca unresolved và các cờ no-op, overlap, long_span, same_span_different_label khi dựng được mẫu. Không sửa nhãn, cắt/mở rộng span hoặc hợp nhất annotation.
- Sentence segmentation và tokenization chỉ nhận paragraph gốc. Hàm tạo input nhận paragraph, offsets và kết quả segmentation; label được gắn sau. Correction, comment, review decision và annotation khác không được ảnh hưởng việc dựng input.

## 2. Tách câu cố định

### 2.1 Pipeline được chọn

Dùng `spacy==3.8.7`, English tokenizer mặc định của `spacy.blank("en")` và duy nhất `sentencizer`, cấu hình:

```python
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer", config={
    "punct_chars": [".", "!", "?", "…"],
    "overwrite": True,
})
```

Đây là lựa chọn thiết kế cố định để không học sentence boundary từ corpus và không cần model POS/dependency ở giai đoạn này. Sentencizer hoạt động theo token; không dùng regex tách tại mọi dấu chấm. Không bổ sung tokenizer exceptions hoặc luật riêng dựa vào nhãn/correction. Viết tắt, số thập phân, contraction và dấu ngoặc theo tokenizer cùng phiên bản; hành vi cụ thể phải được lưu bằng fixtures khi triển khai. Cụm `...` có thể là một token khác dấu `.`; không tự đổi thành ký tự `…`.

Nguồn API: [Sentencizer](https://spacy.io/api/sentencizer), [Tokenizer](https://spacy.io/api/tokenizer), [bản phát hành 3.8.7](https://pypi.org/project/spacy/3.8.7/). Đây là bộ tách câu theo quy tắc, không bảo đảm đúng ngữ nghĩa trên bài viết lỗi; phải báo cáo hạn chế và audit trước khóa dataset.

Chạy từng paragraph một, bao gồm TITLE; không nối paragraph. Không tự tách ở newline, dấu `;` hoặc `:`. Không thêm dấu chấm cho fragment thiếu dấu kết câu, không đặt giới hạn chiều dài để cắt câu. `nlp.max_length = max(1_000_000, len(paragraph) + 1)` trước mỗi lần xử lý để giới hạn mặc định không gây cắt hay loại paragraph dài.

### 2.2 Sentence intervals và khoảng trắng

Với mỗi `Doc.sents`, lấy vị trí bắt đầu của token đầu tiên có `is_space == False`. Bỏ sentence span chỉ chứa whitespace. Gọi các vị trí thu được theo thứ tự là `a0, a1, ..., ak`.

- Nếu không có token không-whitespace: paragraph không có câu, ghi `empty_paragraph` cho annotation mục tiêu có offset hợp lệ.
- Nếu có: boundaries là `[0, a1, ..., ak, len(paragraph)]`; mỗi cặp kề nhau tạo một sentence interval `[s, e)`. Sentence index bắt đầu từ 0 trong paragraph.
- Cách này giữ whitespace đầu paragraph ở câu đầu, whitespace giữa hai câu ở cuối câu trước, whitespace cuối paragraph ở câu cuối. Không dùng `.strip()` để lấy sentence.
- `sentence = paragraph[s:e]`. Các intervals không rỗng, liên tiếp, không giao nhau và phủ đúng toàn paragraph; ghép các sentence phải bằng paragraph nguyên bản.
- Dấu ngoặc/dấu nháy được gán theo sentence span của sentencizer, không chuyển thủ công sau đó.
- Dùng tokens của paragraph Doc để lấy token cho từng interval; không tokenize lại chuỗi sentence trong preparation. Token không-whitespace không được cắt bởi sentence boundary.

Biên câu độc lập với annotation. Nếu một annotation vượt biên do bộ tách câu chưa phù hợp, ghi nhận là `cross_sentence`; không tự ghép câu chỉ cho annotation đó. Muốn sửa luật tách câu phải tăng policy version, chạy lại toàn corpus và audit trước khóa split.

## 3. Gán câu và chuyển offset

Đầu tiên kiểm tra source offsets bằng quy tắc hiện có: paragraph indices tồn tại, đúng thứ tự; endpoint nằm trong paragraph tương ứng; với cùng paragraph phải có `0 <= start_off <= end_off <= len(paragraph)`. Offset không hợp lệ không được sửa bằng clamp hoặc tìm chuỗi gần nhất.

### 3.1 Span không rỗng

Với `p_start < p_end` trong một paragraph, tìm duy nhất interval thỏa:

```text
s <= p_start < p_end <= e
sentence = paragraph[s:e]
error_start = p_start - s
error_end = p_end - s
error_span = paragraph[p_start:p_end]
span_kind = "non_empty"
```

Span kết thúc đúng `e` thuộc câu trước; span bắt đầu đúng `e` thuộc câu sau. Nếu không có một interval chứa toàn span, ghi `cross_sentence`. Không cắt bỏ whitespace khỏi annotation để làm nó vừa một câu.

### 3.2 Insertion

Với `p_start == p_end == p`, đặt `span_kind="insertion"`, `error_span=""`.

- Nếu `p < len(paragraph)`, chọn interval `s <= p < e`.
- Nếu `p == len(paragraph)`, chọn câu cuối và đặt offset bằng chiều dài câu.
- Tại biên chung hai câu, ưu tiên câu sau. Tại khoảng trắng trước biên đó, vẫn thuộc câu trước theo intervals đã chốt.
- Công thức đổi offset vẫn là `p - s`. Không suy insertion từ correction hoặc nhãn.

### 3.3 Điều kiện bắt buộc

```text
0 <= error_start <= error_end <= len(sentence)
sentence[error_start:error_end] == error_span
paragraph[s + error_start:s + error_end] == error_span
```

Sai một điều kiện là lỗi implementation hoặc artifact không nhất quán: dừng run, không âm thầm loại mẫu để cho báo cáo pass.

## 4. Token alignment và context

Chỉ đánh chỉ số cho token `is_space == False`, từ 0 trong sentence. Giữ punctuation và stopwords. Mỗi token lưu nguyên `text`, `start`, `end` tương đối với sentence; phải có `sentence[start:end] == text`. Whitespace vẫn ở sentence nhưng không là token feature.

Với non-empty span `[u, v)`, token `[t_start, t_end)` thuộc `span_token_indices` khi:

```text
t_start < v and u < t_end
```

- `alignment=exact`: có token giao span, `u` bằng start token đầu và `v` bằng end token cuối.
- `alignment=partial_token`: ít nhất một endpoint nằm nghiêm ngặt bên trong một token. **Giữ sample và offset gốc**, dùng toàn bộ token giao span cho lexical/POS/morphology/dependency; không dùng correction để chọn token. Ghi rõ feature biểu diễn token bao phủ span, có thể rộng hơn đoạn ký tự được đánh dấu.
- `alignment=whitespace_edge`: có token giao span, không có endpoint cắt token nhưng không đạt exact (span ăn thêm whitespace).
- Non-empty span không giao token: `no_token_overlap`, không tạo sample chính; đây là giới hạn biểu diễn của protocol, không kết luận annotation sai.
- Quy tắc ưu tiên alignment: partial_token trước exact trước whitespace_edge. Đánh dấu `cuts_start_token` và `cuts_end_token` riêng.
- Context trái gồm tối đa hai token nằm hoàn toàn trước span (`t_end <= u`); phải gồm những token gần nhất. Context phải tương tự với `t_start >= v`. Chỉ dùng tokens trong cùng sentence. Token giao span không được đồng thời làm context.
- Lưu `left_token_indices` và `right_token_indices` theo thứ tự văn bản; `left_1` là phần tử cuối bên trái, `left_2` là phần tử kế trước; `right_1` là phần tử đầu bên phải. Thiếu context dùng danh sách ngắn/rỗng; encoder sau này dùng marker BOS/EOS, không lấy token từ câu khác.
- Với partial-token span, masking sau này phải che toàn bộ lexical identity của mọi token giao span, kể cả phần ký tự nằm ngoài span. Không giữ phần suffix/prefix làm đường truyền lexical gián tiếp.

Insertion không có span token, luôn `span_token_indices=[]`, hai cờ cuts là false. Nếu `t_start < p < t_end` cho một token, ghi `insertion_inside_token` và không tạo sample chính; không tự dời điểm chèn. Nếu không, `alignment=insertion_boundary`, dùng context theo `t_end <= p` và `t_start >= p`. Không có token span thì P/M của span là NOT_APPLICABLE; không loại insertion chỉ vì thiếu context ở một phía.

NLP ở bước feature phải dùng lại token boundaries này, không retokenize hoặc đổi sentence intervals. Khi bổ sung POS/M/D, có thể dựng Doc từ tokens và khoảng trắng gốc, kiểm tra khôi phục đúng sentence và khóa một sentence trên Doc trước parsing. Model NLP hoặc tokenizer không tương thích phải được xử lý rõ trong bước feature, không âm thầm đổi dataset v1.

## 5. Schema artifact

Định dạng chính: JSONL UTF-8, một object mỗi dòng; số nguyên JSON cho indices, boolean thực, `null` cho giá trị không áp dụng, `[]` cho danh sách rỗng. Không biến chuỗi rỗng insertion thành missing value. Các file bảng không lặp văn bản correction. Thư mục đầu ra: `data_set/processed/nucle_preparation_v1/`, được gitignore theo chính sách hiện có.

### 5.1 `samples.jsonl` — một dòng cho mỗi annotation được giữ

| Trường | Kiểu / ý nghĩa |
| --- | --- |
| sample_id | string: `<source_sha256>:<annotation_id>`; khóa annotation, không chứa nhãn |
| annotation_id | int, ID cũ giữ nguyên |
| essay_id, essay_index | string và int, giá trị nguồn; dùng index để phân biệt cả khi essay_id trùng/rỗng |
| sentence_id | string: `<source_sha256>:<essay_index>:<paragraph_index>:<sentence_index>`; phạm vi sử dụng gắn với manifest preparation |
| sentence | string, nguyên slice source theo mục 2 |
| error_start, error_end | int, offset sentence |
| error_span | string, có thể rỗng chỉ khi insertion |
| error_type | enum 5 nhãn mục tiêu, target y |
| span_kind | enum `non_empty`, `insertion` |
| alignment | enum `exact`, `partial_token`, `whitespace_edge`, `insertion_boundary` |
| cuts_start_token, cuts_end_token | bool |
| span_token_indices | list[int], tăng dần |
| left_token_indices, right_token_indices | list[int], mỗi phía tối đa 2 |
| input_sha256 | hash định danh input chính xác theo mục 7 |

`samples` là bảng quản lý dữ liệu có y và metadata, không phải feature matrix. API extractor chỉ được nhận sentence, error_start, error_end và tokens/parse của chính sentence. Không truyền nguyên row gồm label, ID, alignment audit hoặc review metadata làm predictor.

### 5.2 `sentences.jsonl` — bảng sentence/token dùng chung

Một dòng cho mỗi sentence có ít nhất một sample giữ lại: `sentence_id`, `essay_index`, `paragraph_index`, `sentence_index`, `sentence_start_par`, `sentence_end_par`, `sentence`, `tokens`.
`tokens` là list object `{index: int, text: str, start: int, end: int}`. Các sample cùng sentence tham chiếu cùng một token sequence. Các sentence/paragraph khác vẫn có thể tái tạo từ nguồn cho bước duplicate grouping; không được coi bảng sentence này là toàn bộ corpus.

### 5.3 `provenance.jsonl` — một dòng cho mọi annotation đầu vào

Các trường bắt buộc: `annotation_id`, `source_sha256`, `policy_version`, `essay_id`, `essay_index`, `mistake_index`, `start_par`, `end_par`, `start_off`, `end_off`, `original_label`, `mapping` (`identity` với target, `null` ngoài target), `status`, `reason`, `sample_id`, `sentence_id`, `sentence_start_par`, `sentence_end_par`, `quality_flags`, `review_status`.

- `status` thuộc `retained`, `out_of_scope`, `not_constructible`; `reason=null` cho retained.
- `sample_id=null` nếu không giữ; thông tin sentence là null nếu chưa gán được câu. Annotation ngoài taxonomy không cần chạy alignment.
- `quality_flags` là list tên cờ true trong audit hiện hành, sort từ điển. `review_status` thuộc `retain_original`, `unresolved`, `not_reviewed`; dùng log hợp nhất 276 ID. Thiếu artifact audit/review bắt buộc là lỗi run, không giả thành không có cờ.
- Correction/comment tiếp tục ở nguồn và review evidence riêng; tra bằng source checksum + annotation_id, không copy vào samples/sentences.

### 5.4 `rejected.jsonl`, `summary.json`, `manifest.json`

- `rejected`: các dòng provenance có status khác retained, giúp kiểm tra từng ID. Không xóa dòng tương ứng khỏi nguồn hoặc Understanding.
- `summary`: counts đầu vào/target/ngoài taxonomy/retained/not_constructible; phân bố theo nhãn, reason, alignment, span_kind, quality_flags và review_status. Báo cáo tỷ lệ không dựng được theo từng lớp; một reason chính mỗi annotation, các quality flags có thể giao nhau.
- `manifest`: policy/schema version; source path/checksum; checksum Understanding CSV, audit và review log; code commit và hash file code thực tế (bao gồm file chưa commit); Python/spaCy versions; hash uv.lock; cấu hình pipeline và tokenizer serialized hash; ngày chạy; `usage_authorization=user_confirmed`, ngày xác nhận; counts; checksum mọi output ngoài chính manifest.
- Runtime spaCy phải đúng 3.8.7; dependency/uv.lock đã được cập nhật khi triển khai, kèm Typer <0.20 để giữ tương thích Click. Ghi đúng phiên bản Python thực tế; không lấy phiên bản README thay cho môi trường đang chạy.
- Dataset version = SHA256 của manifest JSON canonical, không thêm chính checksum đó vào nội dung được hash; lưu digest trong `manifest.sha256`. Một sample/sentence ID phải được tham chiếu kèm dataset version khi dùng ngoài artifact này.
- Xuất vào thư mục tạm, chỉ công bố artifact hoàn chỉnh sau khi mọi invariant pass; không để run thất bại ghi đè bộ artifact hợp lệ trước đó.

## 6. Quyết định khi không dựng được sample

Áp dụng thứ tự dưới đây, dừng tại reason đầu tiên. Nhờ vậy tổng counts không đếm trùng.

| Thứ tự | Điều kiện | Status / reason |
| --- | --- | --- |
| 1 | TYPE ngoài đúng 5 nhãn, gồm nhãn rỗng | `out_of_scope / label_out_of_scope` |
| 2 | Offset/paragraph indices không hợp lệ | `not_constructible / invalid_offset` |
| 3 | Hai paragraph khác nhau, endpoints hợp lệ | `not_constructible / cross_paragraph` |
| 4 | Paragraph rỗng hoặc chỉ whitespace | `not_constructible / empty_paragraph` |
| 5 | Non-empty span không nằm trọn một sentence interval | `not_constructible / cross_sentence` |
| 6 | Non-empty span không giao token không-whitespace | `not_constructible / no_token_overlap` |
| 7 | Insertion nằm nghiêm ngặt bên trong token | `not_constructible / insertion_inside_token` |
| 8 | Các trường hợp còn lại, invariants pass | `retained / null` |

Segmentation exception, thiếu dependency, checksum mismatch, ID trùng trong input, CSV khác SGML, sai token/sentence slice hoặc source thay đổi giữa run: **dừng toàn bộ run với lỗi kỹ thuật**. Không catch mọi exception rồi biến thành mẫu loại; không fallback sang paragraph đầy đủ hoặc câu đã sửa.

Việc không giữ cross-sentence/cross-paragraph là giới hạn input một câu của benchmark v1. Không suy annotation sai và không đánh đồng với loại overlap. Nếu tỷ lệ mất mẫu lớn, review trước split; thay đổi chính sách phải thành v2, có bảng so sánh counts, không sửa âm thầm v1. Chưa có tỷ lệ thực tế để đặt kết luận về selection bias.

## 7. Trùng input và liên hệ bước split

`input_sha256 = SHA256(UTF8(json.dumps([sentence, error_start, error_end], ensure_ascii=False, separators=(",", ":"))))`.

Giữ nguyên chữ hoa, whitespace và Unicode trong phép hash. Hai input hash giống nhau phải đối chiếu tuple thực trước khi nhóm; không hash nhãn hoặc correction vào input identity. Báo số nhóm input trùng, số annotation trong nhóm và nhóm có nhiều nhãn; giữ nguyên từng annotation. Bước split sau phải nối các essay chứa cùng input vào cùng nhóm, kể cả nhãn khác nhau. Hash này không thay thế policy duplicate/boilerplate/near-duplicate.

Artifact triển khai `input_groups.jsonl` lưu nhóm có ít nhất hai sample: `input_sha256`, `annotation_ids`, `essay_indices`, `label_counts`, `ambiguous`. Không có nhóm thì file rỗng. CLI từ chối output đã tồn tại; chạy lại dùng `--output` tới thư mục mới để bảo toàn bộ cũ.

## 8. Ví dụ và yêu cầu kiểm chứng khi triển khai

Ví dụ tự tạo, không phải văn bản corpus, không đưa vào benchmark:

| Trường hợp | Đầu vào / kết quả bắt buộc |
| --- | --- |
| Đổi offset | Paragraph `Hi. She go.` có intervals `[0,4)`, `[4,11)`; span paragraph `[8,10)` -> sentence `She go.`, span `[4,6)`, chuỗi `go` |
| Qua biên câu | Cùng paragraph, span `[2,5)` -> `cross_sentence`; không cắt thành hai sample |
| Điểm chung hai câu | Cùng paragraph, insertion tại 4 -> câu sau, offset 0; tại 11 -> câu cuối, offset 7 |
| Whitespace giữa câu | Cùng paragraph, insertion tại 3 -> câu trước, offset 3 |
| Cắt token | Sentence `Dogs run.`, token `Dogs` tại `[0,4)`, span `[3,4)` -> `partial_token`, error_span `s`, token span là toàn `Dogs`, context phải bắt đầu ở `run` |
| Span ăn whitespace | Sentence `Dogs run.`, span `[0,5)` -> `whitespace_edge`, token span `Dogs`; không trim về `[0,4)` |
| Chỉ whitespace | Sentence `Dogs run.`, span `[4,5)` -> `no_token_overlap` |
| Chèn giữa token | Sentence `Dogs run.`, insertion tại 2 -> `insertion_inside_token`; tại 4 -> insertion hợp lệ |

Các phép kiểm chứng bắt buộc:

1. Round-trip paragraph -> sentences -> paragraph; sentence/token/error slices khớp tuyệt đối. Kiểm tra Unicode gồm combining mark và emoji, tabs, nhiều spaces, CRLF qua parser, ampersand và literal entity theo parser hiện có.
2. Fixtures tách câu: `Dr. Smith arrived. He left.`, `The value is 3.14. Fine!`, contraction, dấu nháy/ngoặc, `...`, `…`, câu thiếu dấu chấm, newline trong paragraph, TITLE và paragraph chỉ whitespace. Snapshot theo pipeline đã khóa, review boundary trước khi chấp nhận; không khẳng định đúng ngôn ngữ chỉ vì snapshot pass.
3. Offset tại đầu/cuối câu, cuối paragraph, token boundary; nhiều token, punctuation, partial token, insertion trong whitespace và trong token; invalid/cross-paragraph/cross-sentence theo đúng reason priority.
4. Nhiều annotation cùng câu/cùng span/overlap giữ từng ID; unresolved không tự loại; thay correction/comment/nhãn không đổi sentence intervals, tokens hoặc input của một annotation vẫn thuộc taxonomy mục tiêu.
5. `N_all = N_out_of_scope + N_target`, `N_target = N_retained + N_not_constructible`; mỗi input annotation xuất hiện đúng một lần trong provenance. Snapshot hiện có N_all=44.912, N_target=15.788; chưa giả định N_retained=15.788.
6. Mỗi retained có đúng một sample và câu tương ứng; IDs/tokens tham chiếu tồn tại, label thuộc taxonomy, indices tăng dần và không trùng giữa span/context. Chạy lại cùng source/policy/environment tạo cùng dữ liệu, IDs, counts và output checksums (manifest có thời gian chạy có thể khác).

## 9. Bàn giao tiếp theo

Đã triển khai preparation v2, đối chiếu cơ học cả 15.788 mẫu với v1 và rà soát packet mới 50 mẫu. Bước tiếp theo là chốt duplicate/boilerplate grouping rồi thiết kế split theo essay/nhóm; chưa có split benchmark được khóa. Hai boundary case vẫn unresolved và cần adjudication thủ công nếu ảnh hưởng quyết định tiếp theo.
