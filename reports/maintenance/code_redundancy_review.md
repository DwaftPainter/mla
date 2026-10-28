# Rà soát code thừa — 23/10/2026

Ghi chú ngày 27/10/2026: đây là snapshot review trước khi dọn source. Sáu file reporting trong bảng đã bị xóa khỏi cây làm việc; tên file được giữ dạng đường dẫn chữ để không dẫn liên kết thiếu đích. Bốn source trong final lock có bản sao đúng checksum trong zip; xem [kiểm tra reports hiện tại](reports_check.md) trước khi chạy tái lập cũ.

Phạm vi rà soát là commit `52349bd`, gồm **46 file Python trong src, 12 file test và 4 notebook (20 code cells)**. Việc xem xét dựa vào cấu trúc mã, import, lời gọi, cách chạy, tài liệu và các kết quả đã lưu. Không huấn luyện, chấm test hoặc chạy generator có thể ghi đè kết quả trong lần rà soát này. Các nhận xét về mã thừa được giữ theo đúng mốc trước khi dọn source.

## Kết luận

- Một script đã hết vai trò cập nhật hiện hành: `src/reporting/update_validation_documentation.py`.
- Hai generator báo cáo lịch sử vẫn hữu ích để truy vết, nhưng đường dẫn và trạng thái được hardcode đã cũ.
- Code thừa rõ nhất nằm **bên trong** generator báo cáo task: `write_csv`, các danh sách `perclass`/`delta`, biến `fits` và một nhóm import.
- Phát hiện **28 tên import không được dùng trong 15 file**, tương ứng 30 khai báo do hai import có cả nhánh package/direct execution. Đây là tín hiệu từ phân tích tĩnh, đã đối chiếu import/re-export và source; không coi tác dụng phụ khi import thư viện là bằng chứng cần giữ một binding.
- Hai hằng `DOCS_DIR` và `RAW_DIR` hiện chỉ còn chỗ khai báo. Chưa có căn cứ xóa cả module chứa chúng.
- Chưa thấy file core, test hoặc notebook nào có thể kết luận là thừa chỉ vì ít hoặc không có import vào nó. CLI, pytest và notebook là các cách sử dụng độc lập.

## Các chỗ cần xử lý trước

| Ưu tiên | Vị trí | Bằng chứng | Đề xuất |
|---|---|---|---|
| 1 | `src/reporting/update_validation_documentation.py` | Không có caller trong code/test/notebook. Đòi các report ở docs/ cũ; ghi đè status thành test chưa chấm; cập nhật note, reference và DOCX lịch sử. | Đánh dấu ngừng dùng. Giữ bản đã khóa tại chỗ; chỉ loại khỏi cây code làm việc khi đã có phương án giữ snapshot tái lập. Không sửa vài đường dẫn rồi chạy lại. |
| 2 | `src/reporting/build_evaluation_tasks_report.py` | write_csv chỉ có definition; perclass/delta chỉ được tạo và append, không đọc/xuất; fits được đọc từ fit_progress.json rồi không dùng. overall vẫn được dùng trong bảng, cần giữ. | Có thể dọn các phần này cùng import thừa; file không nằm trong final lock. Khi dọn cần đối chiếu nội dung bảng báo cáo trước/sau ở output riêng. |
| 3 | `src/reporting/build_contextual_validation_report.py` | Ghi Markdown và DOCX vào docs/ thay vì reports/summaries/contextual; nội dung là snapshot trước test. | Giữ generator lịch sử; nếu cần generator hiện hành thì dùng entry point/output mới, không sửa ngược bản đã khóa. |
| 3 | `src/reporting/build_bert_diagnostics_report.py` | Ghi DOCX vào docs/; nội dung chứa đề xuất thử tiếp BERT của mốc chẩn đoán, trong khi kế hoạch hiện đã dừng. | Giữ để truy vết; không dùng generator này cập nhật trạng thái hiện hành. |
| 4 | `src/reporting/package_final_reproducibility.py` | README template còn nói chưa tái chạy spaCy từ raw; code vẫn có chức năng đóng gói và kiểm restore riêng. | Cần cập nhật template ở lần đóng gói mới; không phải file thừa và không ghi lại gói lịch sử chỉ để đồng bộ câu chữ. |

Updater cũ hiện sẽ vướng assert do thiếu report ở đường dẫn cũ trước khi chạy tới phần ghi đè. Rủi ro phát sinh nếu các điều kiện đó được khôi phục hoặc bỏ qua; đây không phải bằng chứng nó đã ghi đè tài liệu trong lượt kiểm tra này.

## Ràng buộc checksum

**40/46 file src đã nằm trong final lock**. Hàm [verify_lock](../../src/evaluation/nucle_final_evaluation.py#L15) đọc và băm từng đường dẫn. Xóa, di chuyển hoặc chỉnh một import trong các file đó đều làm kiểm tra fail. Việc được đưa vào lock không chứng minh file đã được thực thi: freeze đã lấy toàn bộ src bằng rglob.

Đã kiểm lại toàn bộ **74 hash trong lock** và checksum lock: khớp tại thời điểm review. Sáu file src ngoài lock là reproduce_evaluation_pipeline.py, review_audit_evidence.py, review_saved_evidence.py, verify_final_results.py, build_evaluation_tasks_report.py và package_final_reproducibility.py. Chúng vẫn có thể được snapshot manifest khác tham chiếu; sửa code mới cần ghi rõ phiên bản, không sửa hash lịch sử để làm kiểm tra pass.

## Import và đoạn code nhỏ không dùng

| File | Tên import không được dùng | Trong final lock |
|---|---|---|
| [src/data/group_split_nucle.py](../../src/data/group_split_nucle.py) | `math` | Có |
| [src/data/parse_nucle.py](../../src/data/parse_nucle.py) | `load_nucle` | Có |
| [src/evaluation/bert_learning_diagnostics.py](../../src/evaluation/bert_learning_diagnostics.py) | `make_loader` | Có |
| [src/evaluation/nucle_final_error_audit.py](../../src/evaluation/nucle_final_error_audit.py) | `read`, `sha` | Có |
| [src/evaluation/nucle_final_evaluation.py](../../src/evaluation/nucle_final_evaluation.py) | `random`, `defaultdict` | Có |
| [src/evaluation/reproduce_evaluation_pipeline.py](../../src/evaluation/reproduce_evaluation_pipeline.py) | `csv` | Không |
| [src/evaluation/review_audit_evidence.py](../../src/evaluation/review_audit_evidence.py) | `Counter` | Không |
| [src/evaluation/review_saved_evidence.py](../../src/evaluation/review_saved_evidence.py) | `sys` | Không |
| [src/evaluation/verify_final_results.py](../../src/evaluation/verify_final_results.py) | `csv`, `json` | Không |
| [src/features/nucle_features.py](../../src/features/nucle_features.py) | `Iterable`, `PROJECT_ROOT` | Có |
| `src/reporting/build_bert_diagnostics_report.py` | `datetime`, `timezone` | Có |
| `src/reporting/build_evaluation_tasks_report.py` | `html`, `Counter`, `np`, `plt`, `RGBColor`, `dump`, `FULL`, `MASK`, `NONLEX` | Không |
| `src/reporting/package_final_reproducibility.py` | `json` | Không |
| `src/reporting/update_validation_documentation.py` | `zipfile` | Có |
| [src/training/nucle_final_selection.py](../../src/training/nucle_final_selection.py) | `Counter` | Có |

15/28 binding trên thuộc sáu file ngoài final lock; 13 binding còn lại thuộc các file đã khóa. Sau khi bỏ write_csv, import csv trong report builder cũng sẽ không còn cần. Report builder chỉ chèn ảnh đã có, không dùng NumPy/pyplot để vẽ; cả block thiết lập matplotlib có thể được xem xét cùng lúc.

Các trường hợp khác:

- [constants.py](../../src/data/constants.py#L9): DOCS_DIR không có người dùng trong code/test/notebook hiện tại.
- [data_download.py](../../src/data/data_download.py#L55): RAW_DIR không còn được dùng; downloader vẫn hoạt động qua SGML_FILE.
- [nucle_ablation.py](../../src/training/nucle_ablation.py#L146): wall_clock được gán nhưng không đọc; run_started là biến khác vẫn được dùng.
- Một số return được unpack vào biến không dùng, như split, defs, summary. Có thể đổi binding thành `_`; không được bỏ lời gọi có công việc cần thiết.
- `lock = verify_lock()` có binding không dùng ở một số nơi nhưng **lời gọi verify_lock phải giữ**. `eval_count` trong closure LR cũng phải giữ: nó là biến nonlocal được đọc để xác định hội tụ và ghi diagnostics.

## Những file nhìn giống trùng nhưng cần giữ

[verify_final_results.py](../../src/evaluation/verify_final_results.py) và [review_saved_evidence.py](../../src/evaluation/review_saved_evidence.py) có phạm vi giao nhau, nhưng chưa thay thế hoàn toàn nhau. File đầu kiểm prediction hash với scoring manifest, thứ tự thời gian lock/test, equality của alias và 75 phép đối chiếu bootstrap bằng cách mở rộng sample. File sau kiểm toàn bộ các metric/slices, CI, CSV và bảng DOCX/Markdown. File sau còn đọc verification.json do bước đầu tạo. Chỉ hợp nhất khi đã chuyển đủ các kiểm tra riêng và cập nhật entry point/consumers.

Nhánh BERT gồm trainer, readout, ba diagnostic/verifier và hai report generator có vai trò khác nhau. Dừng thí nghiệm mới không làm các file tái tạo/giải thích kết quả cũ trở thành vô dụng. Hai file test BERT kiểm tokenization/cropping và gradient/readout, không trùng nhiệm vụ.

Các helper đọc CSV/JSON, tính hash, require và fmt có lặp nhỏ. Hai hàm main trong feature/POS giống cấu trúc nhưng gọi run và constants của module khác nhau, nên không thể coi là bản sao chức năng. Metric tính độc lập trong review có mục đích kiểm chứng; không gom về cùng implementation rồi vẫn gọi đó là phép kiểm độc lập.

`__init__.py` là dấu mốc package; không suy luận file rỗng là file cần xóa. parse_nucle.py là thư viện được import, không phải entry point xuất CSV. Điểm vào xuất CSV hiện là data_understanding.py.

## Dependency

Không thấy import bs4/BeautifulSoup trong code thực thi; parser hiện dùng html.parser.HTMLParser. `bs4` là ứng viên bỏ khỏi dependencies trực tiếp khi chỉnh môi trường ở phiên bản tiếp theo. `lxml` không được import trực tiếp bởi project nhưng **python-docx đang phụ thuộc lxml**, nên không kết luận có thể gỡ lxml khỏi môi trường. pyproject.toml và uv.lock cũng nằm trong final lock.

## Bản sao local ngoài repository

Ba thư mục `.cache/evaluation_tasks_bundle_restore`, `.cache/final_bundle_restore`, `.cache/publication_checkout` có tổng 150 file Python dạng export/snapshot (46 + 46 + 58). Đây là bản sao phục vụ kiểm restore/checkout, không phải 150 module sản phẩm cần bảo trì.

Có 17 script trợ giúp một lần ở gốc .cache trước lượt audit này, ví dụ reorganize_documents.py, write_document_indexes.py, update_publication_docs.py. Không cần đưa các script này vào src; chỉ dọn khi không còn cần thao tác hoặc bằng chứng gốc. Các script mới dùng để audit trong lượt này cũng nằm ở .cache và bị Git ignore.

Năm file Python trong reports/**/before_layout hoặc documentation_archive là snapshot lịch sử. Một bản build_final_evaluation_report.py đã bị thay bằng report task nằm trong archive, không phải entry point hiện hành. Các thư viện trong .venv/.cache/uv là dependency cache, không tính là code project thừa. Chưa xóa các bản sao hay cache trong lượt này.

## Danh mục toàn bộ src

Số nơi import vào bao gồm source, tests và từng code cell notebook; không tính lời nhắc trong tài liệu. Số 0 không đồng nghĩa không sử dụng.

| File | Nơi import vào | Final lock | Vai trò / kết luận |
|---|---:|---|---|
| [src/__init__.py](../../src/__init__.py) | 0 | Có | Giữ dấu mốc package; không phải code thừa. |
| [src/data/__init__.py](../../src/data/__init__.py) | 0 | Có | Giữ dấu mốc package; không phải code thừa. |
| [src/data/boundary_challenges.py](../../src/data/boundary_challenges.py) | 2 | Có | Hai boundary cases cần giữ và gắn cờ trong preparation; test hồi quy. |
| [src/data/compare_preparation.py](../../src/data/compare_preparation.py) | 1 | Có | So sánh v1/v2, kiểm alignment, tạo packet review; test gọi check_alignment. |
| [src/data/constants.py](../../src/data/constants.py) | 23 | Có | Đường dẫn và taxonomy dùng chung; chỉ DOCS_DIR hiện không còn được tham chiếu. |
| [src/data/data_download.py](../../src/data/data_download.py) | 2 | Có | Downloader NUCLE, retry và xử lý file tải dở; có test riêng. |
| [src/data/data_quality.py](../../src/data/data_quality.py) | 6 | Có | Audit offset, missing correction, overlap và duplicate; dùng khi tái lập. |
| [src/data/data_understanding.py](../../src/data/data_understanding.py) | 2 | Có | Xuất toàn bộ annotation và báo cáo nguồn; entry point được README hướng dẫn. |
| [src/data/group_split_nucle.py](../../src/data/group_split_nucle.py) | 1 | Có | Connected groups và split; được pipeline tái lập gọi. |
| [src/data/nucle_sgml.py](../../src/data/nucle_sgml.py) | 17 | Có | Parser HTMLParser giữ nguyên văn bản/offset; thư viện lõi. |
| [src/data/parse_nucle.py](../../src/data/parse_nucle.py) | 13 | Có | Chuyển document thành DataFrame và tái dựng span; là thư viện, không cần main. |
| [src/data/prepare_nucle.py](../../src/data/prepare_nucle.py) | 7 | Có | Preparation v1/v2, provenance, alignment và manifest. |
| [src/data/sentence_boundaries.py](../../src/data/sentence_boundaries.py) | 1 | Có | Sửa biên câu text-only cho v2; được preparation gọi. |
| [src/evaluation/__init__.py](../../src/evaluation/__init__.py) | 0 | Có | Giữ dấu mốc package; không phải code thừa. |
| [src/evaluation/bert_checkpoint_verification.py](../../src/evaluation/bert_checkpoint_verification.py) | 0 | Có | Reload checkpoint BERT và so prediction train/validation lịch sử. |
| [src/evaluation/bert_learning_diagnostics.py](../../src/evaluation/bert_learning_diagnostics.py) | 0 | Có | Input/gradient/optimizer/marker và khả năng nhớ train nhỏ. |
| [src/evaluation/bert_target_pair_probe.py](../../src/evaluation/bert_target_pair_probe.py) | 0 | Có | Probe hai span khác nhãn trong cùng câu train. |
| [src/evaluation/logistic_thread_sensitivity.py](../../src/evaluation/logistic_thread_sensitivity.py) | 0 | Có | Giải thích khác biệt số học của LR khi chạy 4/10 threads. |
| [src/evaluation/metrics.py](../../src/evaluation/metrics.py) | 9 | Có | Metric năm nhãn và accuracy dùng bởi benchmark, ablation, evaluation và tests. |
| [src/evaluation/nucle_final_error_audit.py](../../src/evaluation/nucle_final_error_audit.py) | 1 | Có | Chọn audit/control và dựng packet nguồn; choose được review mới gọi. |
| [src/evaluation/nucle_final_evaluation.py](../../src/evaluation/nucle_final_evaluation.py) | 6 | Có | Verify lock, score, bootstrap và replay các model đã khóa. |
| [src/evaluation/nucle_span_analysis.py](../../src/evaluation/nucle_span_analysis.py) | 5 | Có | Metric theo span/cặp nhãn/POS; dùng cả trong nhánh contextual. |
| [src/evaluation/preprocessing_sensitivity_audit.py](../../src/evaluation/preprocessing_sensitivity_audit.py) | 1 | Có | Đối chứng preparation v1/v2 và validation audit; có helper được thread diagnostic dùng. |
| [src/evaluation/reproduce_evaluation_pipeline.py](../../src/evaluation/reproduce_evaluation_pipeline.py) | 0 | Không | Tái lập raw → preparation → split → features → 52 fit → test replay. |
| [src/evaluation/review_audit_evidence.py](../../src/evaluation/review_audit_evidence.py) | 0 | Không | Kiểm nguồn/parser, margin LR và lập hàng chờ review chuyên gia. |
| [src/evaluation/review_saved_evidence.py](../../src/evaluation/review_saved_evidence.py) | 0 | Không | Kiểm độc lập metric/CI, bảng CSV và sự khớp Markdown/DOCX. |
| [src/evaluation/verify_final_results.py](../../src/evaluation/verify_final_results.py) | 0 | Không | Kiểm hash predictions, thứ tự lock/test, aliases và bootstrap bằng cách mở rộng sample. |
| [src/features/__init__.py](../../src/features/__init__.py) | 0 | Có | Giữ dấu mốc package; không phải code thừa. |
| [src/features/nucle_features.py](../../src/features/nucle_features.py) | 9 | Có | Feature S/L/P/M/D và encoder train-only. |
| [src/features/nucle_pos.py](../../src/features/nucle_pos.py) | 8 | Có | Alignment và baseline POS/Majority; nhiều module khác dùng. |
| [src/models/__init__.py](../../src/models/__init__.py) | 0 | Có | Giữ dấu mốc package; không phải code thừa. |
| [src/models/bert_span_readout.py](../../src/models/bert_span_readout.py) | 3 | Có | CLS/span mean và target mask; giữ để hiểu/kiểm chẩn đoán đã báo cáo. |
| [src/models/nucle_logistic.py](../../src/models/nucle_logistic.py) | 5 | Có | Implementation LR/LBFGS và convergence diagnostics. |
| [src/models/nucle_tree.py](../../src/models/nucle_tree.py) | 4 | Có | Implementation cây Gini và prediction. |
| [src/reporting/__init__.py](../../src/reporting/__init__.py) | 0 | Có | Giữ dấu mốc package; không phải code thừa. |
| `src/reporting/build_ablation_notebook.py` | 0 | Có | Dựng notebook 04 từ artifact validation; có entry point độc lập. |
| `src/reporting/build_bert_diagnostics_report.py` | 0 | Có | Generator lịch sử; còn xuất vào docs/ và ghi đề xuất thử BERT của mốc cũ. |
| `src/reporting/build_contextual_validation_report.py` | 0 | Có | Generator lịch sử; còn xuất vào docs/ và ghi trạng thái trước test. |
| `src/reporting/build_evaluation_tasks_report.py` | 0 | Không | Generator report task hiện hành; có dead helper/import/biến cần dọn. |
| `src/reporting/package_final_reproducibility.py` | 0 | Không | Đóng gói và kiểm restore; có chức năng riêng, README mẫu còn trạng thái cũ. |
| `src/reporting/update_validation_documentation.py` | 0 | Có | Updater một lần cho trạng thái tuần 1; không còn phù hợp workspace hiện tại. |
| [src/training/__init__.py](../../src/training/__init__.py) | 0 | Có | Giữ dấu mốc package; không phải code thừa. |
| [src/training/bert_span_classifier.py](../../src/training/bert_span_classifier.py) | 4 | Có | Training/tokenization benchmark BERT lịch sử; nhiều diagnostic và test còn dùng. |
| [src/training/nucle_ablation.py](../../src/training/nucle_ablation.py) | 3 | Có | Sáu variants, masking và helper sparse/remap được final selection dùng. |
| [src/training/nucle_benchmark.py](../../src/training/nucle_benchmark.py) | 5 | Có | Grid LR/Tree và loader/matrix dùng ở các bước khác. |
| [src/training/nucle_final_selection.py](../../src/training/nucle_final_selection.py) | 8 | Có | 52 fits, aliases, encoder IO và freeze; evaluation/reporting nhập nhiều helper. |

## Tests và notebooks

Cả 12 file test có nhiệm vụ hiện hành. test_model.py và test_training.py của demo cũ đã được xóa ở commit được review; không còn nằm trong danh mục code.

| File | Vai trò |
|---|---|
| [tests/test_bert_readout.py](../../tests/test_bert_readout.py) | Mask, gradient, CLS equivalence và padding. |
| [tests/test_bert_span_classifier.py](../../tests/test_bert_span_classifier.py) | Crop, offsets, marker và không rò nhãn/correction. |
| [tests/test_boundary_challenges.py](../../tests/test_boundary_challenges.py) | Regression cho hai boundary cases có văn bản NUCLE local; skip nếu không có fixture. |
| [tests/test_datadownload.py](../../tests/test_datadownload.py) | Retry, bỏ qua file có sẵn và xóa tải dở bằng mock, không tải mạng. |
| [tests/test_final_evaluation.py](../../tests/test_final_evaluation.py) | Pair metric, bootstrap group, encoder IO, alias/family removal và lock. |
| [tests/test_nucle_benchmark.py](../../tests/test_nucle_benchmark.py) | Gini, metric, không tải test khi train và ghi timing. |
| [tests/test_nucle_features.py](../../tests/test_nucle_features.py) | Feature leakage, train-only encoder và offset. |
| [tests/test_nucle_preparation.py](../../tests/test_nucle_preparation.py) | Segmentation, alignment, provenance, input hash và no-overwrite. |
| [tests/test_nucle_quality.py](../../tests/test_nucle_quality.py) | Quality flags, duplicates, overlap và CSV integrity. |
| [tests/test_nucle_understanding.py](../../tests/test_nucle_understanding.py) | SGML/schema, edit type, invalid offsets và đường dẫn. |
| [tests/test_preparation_alignment.py](../../tests/test_preparation_alignment.py) | Contract của bộ so sánh v1/v2, khác test dựng dataset. |
| [tests/test_sentence_boundaries.py](../../tests/test_sentence_boundaries.py) | Luật biên câu v2 và tính bất biến tokenization. |

| Notebook | Vai trò |
|---|---|
| [01_data_understanding.ipynb](../../notebooks/01_data_understanding.ipynb) | Đọc và trình bày thống kê annotation nguồn. |
| [02_nucle_data_quality.ipynb](../../notebooks/02_nucle_data_quality.ipynb) | Đọc và trình bày audit dữ liệu. |
| [03_data_visualization.ipynb](../../notebooks/03_data_visualization.ipynb) | Trình bày benchmark validation. |
| [04_feature_ablation_report.ipynb](../../notebooks/04_feature_ablation_report.ipynb) | Trình bày ablation/masking validation. |

Chúng là giao diện đọc kết quả, không thay thế pipeline trong src. Không xóa chỉ vì source có hàm tính cùng metric hoặc script tạo notebook.

## Thứ tự dọn đề xuất

1. Dọn dead helper/list/import trong build_evaluation_tasks_report.py và import thừa của năm module ngoài lock còn lại; đối chiếu report ở output riêng.
2. Đánh dấu updater cũ ngừng dùng trong tài liệu/điểm vào. Giữ snapshot code đã khóa và checksum; chưa di chuyển/xóa file đã khóa.
3. Tách việc nâng cấp generator/template hiện hành khỏi các generator lịch sử. Chỉ hợp nhất hai verifier sau khi bảo toàn đầy đủ assertion và consumers.
4. Khi cần giảm dung lượng local, kiểm lại bundle gốc trước khi dọn bản export/restore; không gộp việc đó với xóa corpus/checkpoint hoặc bằng chứng thí nghiệm.

Giới hạn: phân tích tĩnh không chứng minh không có consumer bên ngoài repository hoặc lời gọi động. Đã kiểm syntax cho 58 file Python và 20 code cells; không có lỗi parse. Đây là lượt review, chưa sửa implementation nên không chạy lại bộ test/thí nghiệm. Kết quả 99 test pass của lượt commit trước không được tính là một lần chạy mới ở đây.
