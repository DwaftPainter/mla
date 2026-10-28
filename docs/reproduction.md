# Hướng dẫn chạy, kiểm tra và tái lập

Bổ sung28/10/2026: BERT mở rộng đã có test/CI. Xem [hướng dẫn train](bert_small_training_guide.md), [phân tích cuối](../reports/bert_final_analysis/analysis_report.md) và [bàn giao](../reports/summaries/final/handover_checklist.md). Tái tính evidence vào output mới bằng `.venv/Scripts/python.exe -m src.evaluation.bert_final_analysis --output reports/bert_final_analysis_recheck`; không chạy scorer hoặc thay lock cũ.

Xuất tài liệu dùng môi trường riêng để không đổi môi trường huấn luyện:

```powershell
uv venv .cache/reporting_env --python .venv/Scripts/python.exe --system-site-packages --cache-dir .cache/uv
uv pip install --python .cache/reporting_env/Scripts/python.exe --cache-dir .cache/uv -r requirements-reporting.txt
.cache/reporting_env/Scripts/python.exe -m src.reporting.build_bert_delivery --documents-only
.cache/reporting_env/Scripts/python.exe -m src.reporting.build_nucle_slides
.cache/reporting_env/Scripts/python.exe -m src.evaluation.verify_bert_delivery
```

Các lệnh export mặc định đọc bản Markdown hiện hành, giữ nguyên phần văn bản đã biên tập rồi cập nhật Word/PDF. Chỉ dùng `--refresh-evidence` khi muốn tạo lại bản nháp kỹ thuật từ số liệu; tùy chọn này sẽ thay phần diễn giải đã sửa. Các bản trước biên tập được lưu trong documentation_archive. Ghi chú AI và quyết định review là đầu vào riêng, không tự sinh lại từ phép tính metric. PDF và slide PDF được kiểm tra trong môi trường xuất; Word/PowerPoint có thể phân trang hoặc điều chỉnh chữ khác khi mở trên máy khác.

Tài liệu vận hành hiện hành, đối chiếu code/artifact ngày 23/10/2026. [Project guide](note_project_guide.md) giải thích thuật toán/công thức; [status.md](status.md) ghi tiến độ. Các lệnh dùng PowerShell tại thư mục gốc repository.

## 1. Chọn đúng mức công việc

| Mục đích | Cần gì | Có trong Git? |
|---|---|---|
| Đọc kết quả | Markdown/DOCX/JSON/CSV tổng hợp | Có; bắt đầu ở [reports](../reports/README.md) |
| Unit tests | Môi trường nền; thêm transformers cho hai module BERT | Có code; fixture boundary chứa NUCLE chỉ local |
| Reload checkpoint và kiểm metric | Checkpoint, encoded input, predictions, scoring manifests và các input trong lock | Chỉ một phần; phần còn lại local |
| Dựng lại từ raw và huấn luyện | NUCLE đúng hash, review logs, lịch sử selection/predictions, môi trường và output riêng | Chưa đủ input chỉ bằng clone repo |

Nguồn SGML tại `data_set/raw/nucle3.2.sgml`, SHA256 `8441a60060730bdc6af1e79a064c68441ecb24a66da91ea2aab0061a64a130ac`. Cần quyền sử dụng NUCLE. Downloader không tự tạo hoặc thay thế quyền sử dụng/review log.

## 2. Môi trường

Lượt chạy đã kiểm dùng Python 3.13.7, spaCy 3.8.7, `en_core_web_sm` 3.8.0. Tagger/parser là hai component trong cùng model package. `.python-version` ghi 3.13; khoảng Python cho phép trong pyproject không có nghĩa mọi patch/platform đã tái lập giống byte.

Trong checkout mới:

```powershell
uv sync --locked --python 3.13.7
uv pip install --python .venv/Scripts/python.exe -r requirements-bert.txt
.venv/Scripts/python.exe -m pytest tests -q
```

Lệnh cài thứ hai chỉ cần cho BERT/DOCX; pin `transformers==4.57.6`, `python-docx==1.2.0`. Dependencies bắc cầu của phần bổ sung không được lock nền khóa hết. Đối chiếu versions trong [pipeline_reproduction.json](../reports/evaluation_followup/pipeline_reproduction.json), cùng `reports/contextual_model_validation/environment.txt` (local) cho lượt BERT. Không cài lại mù quáng lên môi trường nghiên cứu đã có.

Baseline/final LR dùng CPU 10 threads; sensitivity và BERT dùng 4 threads. Đã quan sát 2/2.368 validation predictions LR khác nhau khi đổi 4/10 threads; seed giống nhau chưa đủ bảo đảm tái tạo số học.

## 3. Đầu vào và output từng bước

Đây là bản đồ chạy trong workspace đầy đủ input, không phải danh sách lệnh tự động cho clone trống. Path trong bảng tính từ root.

| Entry point trong src | Input cần có | Output / cách ghi |
|---|---|---|
| `data/data_download.py` | Quyền truy cập nguồn | SGML; bỏ qua nếu file đã có |
| `data/data_understanding.py` | SGML | CSV toàn annotation và report; có thể ghi lại file, dùng `--csv`, `--report` mới |
| `data/data_quality.py` | SGML, CSV | Audit CSV, duplicate evidence, JSON/Markdown; có thể ghi lại; dùng cả `--output-dir` và `--report` mới |
| `data/prepare_nucle.py` | SGML, CSV, audit CSV, overlap review log | Samples/sentences/provenance/manifest; `--policy v1/v2`, `--output` phải chưa tồn tại |
| `data/compare_preparation.py` | Hai preparation, source và boundary decisions lịch sử | Diff/regression/review packet; `--old`, `--new`, `--output` riêng |
| `data/group_split_nucle.py` | Source, preparation, duplicate evidence | Essay groups/split; `--prepared`, `--duplicate-report`, `--output`, `--seed` |
| `features/nucle_pos.py` | Preparation và split | POS baseline; `--prepared`, `--split`, `--output` riêng |
| `features/nucle_features.py` | Preparation, split, NLP package | Features/targets/encoder fit train; `--prepared`, `--split`, `--output` riêng |
| `training/nucle_benchmark.py` | Feature artifact và split | Grid Full LR/Tree; `--features`, `--split`, `--output` riêng |
| `training/nucle_ablation.py` | Features, split, preparation chuẩn cho masking | Sáu variants; có `--variant`; features/split/output có thể chỉ định, masking vẫn đọc preparation chuẩn |
| `evaluation/nucle_span_analysis.py` | Dataset/split và predictions validation cũ | Span/cặp nhãn/POS tables; `--output` mới |
| `training/nucle_final_selection.py` | Data/features/split và summary/predictions sáu variants cũ | 52 fit rồi `--freeze`; path cố định, từ chối nếu reports/final_evaluation đã tồn tại |
| `evaluation/nucle_final_evaluation.py` | Toàn bộ lock và artifact local tương ứng | Mode `score`, `bootstrap`, `replay`; score từ chối nếu test đã bắt đầu/chấm |

`data/parse_nucle.py` là thư viện parser/DataFrame; chạy file riêng không xuất CSV. Điểm vào xuất CSV hiện là data_understanding.py.

Preparation cần `data_set/interim/nucle_overlap_review_log_all.csv`; comparison cần `data_set/interim/nucle_boundary_decisions.jsonl`. Đây là quyết định nghiên cứu lịch sử, không được thay bằng file rỗng. Tái lập giữ chúng cố định, không tái tạo adjudication của con người.

Ví dụ rebuild preparation sau khi đủ input:

```powershell
.venv/Scripts/python.exe src/data/prepare_nucle.py --policy v1 --output data_set/processed/nucle_preparation_v1_rebuild
.venv/Scripts/python.exe src/data/prepare_nucle.py --policy v2 --output data_set/processed/nucle_preparation_v2_rebuild
.venv/Scripts/python.exe src/data/compare_preparation.py --old data_set/processed/nucle_preparation_v1_rebuild --new data_set/processed/nucle_preparation_v2_rebuild --output reports/nucle_preparation_comparison_rebuild
```

Output phải chưa tồn tại. Muốn train với preparation mới phải truyền đúng input qua từng bước; không trộn feature mới với split/encoder cũ. Runner ablation/final có path cố định nên dùng workspace riêng đúng bố cục.

## 4. Khóa, test và kiểm chứng

[lock.json](../reports/final_evaluation/lock.json) và [lock.sha256](../reports/final_evaluation/lock.sha256) ghi cấu hình/hash trước test: 74 input hashes, gồm 40 source files, protocol, môi trường nền, data, encoder và checkpoint. Model fit trên train; validation chọn cấu hình; test được mở sau khóa.

20 tên pipeline gồm 18 tên LR/Tree + Majority/POS. Full−D là alias của S+L+P+M cho từng model, nên có **16 checkpoint LR/Tree riêng** và hai baseline. 52 fit gồm 40 grid fit Full−P/M và 12 refit cấu hình lịch sử.

Các lệnh kiểm trên workspace đầy đủ:

```powershell
.venv/Scripts/python.exe src/evaluation/nucle_final_evaluation.py replay
.venv/Scripts/python.exe src/evaluation/verify_final_results.py
.venv/Scripts/python.exe src/evaluation/review_saved_evidence.py
```

Chúng không chọn lại model nhưng có ghi `checkpoint_replay.json`, `verification.json` và output trong reports/evaluation_followup. Dùng workspace riêng nếu cần giữ bằng chứng cũ. review_audit_evidence.py đọc thêm corpus/private packet, kiểm parser và ghi margin/queue; không phải unit test không cần dữ liệu.

Các module final không có CLI `--output` chung. Không truyền `--help` cho mọi script: nucle_final_selection.py chỉ nhận nhánh `--freeze`, trường hợp khác đi vào run. Chỉ dùng `--help` với script thực sự có argparse.

## 5. Đã tái lập đến đâu?

[reproduce_evaluation_pipeline.py](../src/evaluation/reproduce_evaluation_pipeline.py) đã chạy trong môi trường riêng: SGML → Understanding/audit → preparation v2 → grouping/split → NLP/features → 52 fit → test replay. Data outputs, 16 checkpoint LR/Tree và 20 bộ prediction theo tên pipeline khớp; xem [bằng chứng](../reports/evaluation_followup/pipeline_reproduction.json) và [diễn giải](../reports/evaluation_followup/review_report.md).

Script yêu cầu đúng môi trường `.cache/final_replay_env`, output `.cache/evaluation_pipeline_reproduction` chưa tồn tại và đủ input lịch sử. Lệnh từng dùng:

```powershell
.cache/final_replay_env/Scripts/python.exe src/evaluation/reproduce_evaluation_pipeline.py
```

Workspace hiện đã có output nên script sẽ dừng theo thiết kế. Thử lại cần workspace/output riêng và môi trường đúng phiên bản; không xóa kết quả chuẩn để vượt guard.

Chưa rerun toàn grid sáu variants lịch sử hoặc BERT trong môi trường sạch. Review logs/config cũ vẫn là input cố định. Bộ 13 tests của lượt tái lập là tập con; lượt kiểm trước push code có 99 tests pass, bản export thiếu fixture NUCLE có 96 pass/1 module skip. Các số này thuộc các phạm vi kiểm khác nhau, không phải kết quả chạy mới ở lượt sửa tài liệu này.

Lock lưu path theo Windows. Chưa xác nhận chạy nguyên trạng trên Linux/macOS. `.gitattributes` giữ byte/line endings; không sửa hash để vượt kiểm tra.

## 6. Reporting và bảo trì

| Script | Cách dùng hiện tại |
|---|---|
| `reporting/build_evaluation_tasks_report.py` | Xuất vào reports/summaries/evaluation, ghi lại DOCX/Markdown; cần artifact local. |
| `reporting/build_ablation_notebook.py` | Dựng và ghi lại notebook 04 từ artifact validation. |
| `reporting/build_contextual_validation_report.py`, `build_bert_diagnostics_report.py` | Generator lịch sử với path/nội dung cũ, không dùng cập nhật trạng thái hiện hành. |
| `reporting/update_validation_documentation.py` | Updater một lần đã lỗi thời; ngừng dùng cho bảo trì tài liệu. |
| `reporting/package_final_reproducibility.py` | Đóng gói snapshot/kiểm restore; template README còn mốc trước raw reproduction. Zip cũ không đại diện toàn bộ repo mới nhất. |

Đọc [code review](../reports/maintenance/code_redundancy_review.md) trước khi dọn. Sửa code mới không được dùng để sửa ngược hash/bằng chứng cũ. Ý tưởng phát sinh sau khi xem test cần thiết kế đánh giá mới.
