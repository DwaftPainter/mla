# NUCLE data pipeline

Dự án xây pipeline dữ liệu NUCLE và benchmark phân loại năm nhãn `ArtOrDet`, `Prep`, `Nn`, `Vform`, `SVA`. Mỗi annotation được giữ như một quan sát riêng; pipeline không tự sửa nhãn, span hoặc correction. Group-aware split, feature extraction S/L/P/M/D, POS baseline và benchmark LR/Tree đã chạy trên train/validation; test chưa được chấm.

Tài liệu chính:

| File | Nội dung |
| --- | --- |
| [docs/status.md](docs/status.md) | Trạng thái, kết quả kiểm chứng và bước tiếp theo |
| [docs/proposal.md](docs/proposal.md) | Câu hỏi nghiên cứu và phạm vi |
| [docs/preparation.md](docs/preparation.md) | Schema, segmentation, offset, alignment và artifact preparation |
| [docs/experiment_protocol.md](docs/experiment_protocol.md) | Quy trình feature, split, tuning và đánh giá |

Corpus được cấp quyền đặt tại `data_set/raw/nucle3.2.sgml`. Môi trường dùng Python 3.12 hoặc 3.13.7; cài dependency bằng `uv sync --locked`.

## Các bước dữ liệu

```powershell
py src/data/data_understanding.py
py src/data/data_quality.py
py src/data/prepare_nucle.py
```

Data Understanding xuất toàn bộ annotation vào `data_set/interim/nucle_errors.csv` và báo cáo vào `reports/nucle_data_understanding.md`. Data Quality đối chiếu CSV với nguồn, tạo audit flags, báo cáo Markdown/JSON, annotation audit và duplicate candidates trong `reports/`. Các flag là ứng viên cần xem xét, không khẳng định annotation sai.

Preparation mặc định chạy policy v2 với spaCy 3.8.7 và tạo `data_set/processed/nucle_preparation_v2/`. Đầu vào gồm SGML, annotation CSV, audit CSV và review log `data_set/interim/nucle_overlap_review_log_all.csv`. Artifact chuẩn đã có trong môi trường làm việc này; không chạy lại nếu không cần rebuild. Script từ chối ghi đè. Nếu cần rebuild có chủ đích, chỉ định một output mới và cập nhật downstream manifests.

Dataset gồm `samples.jsonl`, `sentences.jsonl`, `provenance.jsonl`, `rejected.jsonl`, `boundary_challenges.jsonl`, `input_groups.jsonl`, `summary.json`, `manifest.json` và `manifest.sha256`. Manifest ghi checksum của nguồn và output. V2 hiện giữ 15.788/15.788 mẫu mục tiêu; hai boundary challenge vẫn nằm trong dữ liệu chính, được gắn cờ `unresolved` và kiểm tra hồi quy trong `tests/test_boundary_challenges.py`.

Group split cục bộ nằm tại `data_set/split/nucle_group_split/` với seed `20261013`: 11.052 train, 2.368 validation, 2.368 test. Grouping policy, decisions, per-essay groups, sample split manifest, summary và output hashes được lưu cùng thư mục; không chứa sentence text. Benchmark hiện tại fit trên train và chọn cấu hình bằng validation; test chưa được chấm.

POS tagger và baseline diagnostic đã khóa ở spaCy `3.8.7` + `en_core_web_sm 3.8.0`. Report hiện có tại `reports/nucle_pos_baseline_v1/`; script chỉ đọc train/validation, không chấm test. Nếu cần tạo lại, chọn một output mới vì script không ghi đè:

```powershell
uv run --locked python src/features/nucle_pos.py
```

Kết quả hiện tại nằm ở `reports/nucle_pos_baseline_v1/`. Khi chạy lại, chọn `--output` là một thư mục mới vì script không ghi đè.

Feature extraction và benchmark đã chạy. Khi dựng lại từ đầu trong môi trường có đủ input, các entry point là:

```powershell
py src/features/nucle_features.py
py src/training/nucle_benchmark.py
```

Artifact feature chuẩn nằm tại `data_set/features/nucle_features_v1/`; benchmark mới nhất tại `reports/nucle_feature_benchmark_solver_v2/`. LR tốt nhất trên validation đạt Macro-F1 0.8067; Decision Tree đạt 0.7879. Báo cáo chi tiết và biểu đồ nằm trong `notebooks/03_data_visualization.ipynb`. Lệnh benchmark mặc định từ chối ghi đè output đã có.

Bằng chứng so sánh v1/v2 hiện có tại `reports/nucle_preparation_v2_comparison/`. Script xác minh manifest/output checksums, metadata nguồn, span offsets, sentence/token alignment và tạo diff cùng regression evidence. Chỉ chạy lại khi có thay đổi preparation; script từ chối ghi đè artifact hiện có.

## Kiểm thử

Chạy các test dữ liệu:

```powershell
uv run python -B -m pytest -p no:cacheprovider tests/test_nucle_understanding.py tests/test_nucle_quality.py tests/test_nucle_preparation.py tests/test_preparation_alignment.py tests/test_boundary_challenges.py tests/test_sentence_boundaries.py tests/test_datadownload.py -q
```

Các artifact trong `data_set/` và báo cáo trong `reports/` được tạo cục bộ; các file trong `tests/` cũng được giữ ngoài Git theo cấu hình repo. Git theo dõi code, notebook và proposal. Ba notebook trong `notebooks/` phân tích dữ liệu, chất lượng và kết quả benchmark; notebook không thay đổi corpus.
