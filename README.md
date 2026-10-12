# NUCLE data pipeline

Dự án hiện tập trung vào đọc corpus, kiểm tra chất lượng annotation và tạo dữ liệu câu/span cho năm nhãn `ArtOrDet`, `Prep`, `Nn`, `Vform`, `SVA`. Mỗi annotation được giữ như một quan sát riêng; pipeline không tự sửa nhãn, span hoặc correction, và chưa thực hiện train/validation/test split.

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
uv run python -B -m src.data.data_understanding
uv run python -B -m src.data.data_quality
uv run python -B -m src.data.prepare_nucle
```

Data Understanding xuất toàn bộ annotation vào `data_set/interim/nucle_errors.csv` và báo cáo vào `reports/nucle_data_understanding.md`. Data Quality đối chiếu CSV với nguồn, tạo audit flags, báo cáo Markdown/JSON, annotation audit và duplicate candidates trong `reports/`. Các flag là ứng viên cần xem xét, không khẳng định annotation sai.

Preparation mặc định chạy policy v2 với spaCy 3.8.7 và tạo `data_set/processed/nucle_preparation_v2/`. Đầu vào gồm SGML, annotation CSV, audit CSV và review log `data_set/interim/nucle_overlap_review_log_all.csv`. Nếu chạy lại, chọn một thư mục output mới vì pipeline không ghi đè artifact có sẵn:

```powershell
uv run python -B -m src.data.prepare_nucle --output data_set/processed/nucle_preparation_v2_rerun
```

Dataset gồm `samples.jsonl`, `sentences.jsonl`, `provenance.jsonl`, `rejected.jsonl`, `boundary_challenges.jsonl`, `input_groups.jsonl`, `summary.json`, `manifest.json` và `manifest.sha256`. Manifest ghi checksum của nguồn và output. V2 hiện giữ 15.788/15.788 mẫu mục tiêu; hai boundary challenge vẫn nằm trong dữ liệu chính, được gắn cờ `unresolved` và kiểm tra hồi quy trong `tests/test_boundary_challenges.py`.

Để kiểm tra checksum và so sánh v1/v2 trên toàn bộ mẫu:

```powershell
uv run python -B -m src.data.compare_preparation --output reports/nucle_preparation_v2_comparison_rerun
```

Lệnh so sánh cần thư mục output mới. Nó xác minh manifest/output checksums, metadata nguồn, span offsets, sentence/token alignment và ghi diff cùng regression evidence vào `reports/`.

## Kiểm thử

Chạy các test dữ liệu:

```powershell
uv run python -B -m pytest -p no:cacheprovider tests/test_nucle_understanding.py tests/test_nucle_quality.py tests/test_nucle_preparation.py tests/test_preparation_alignment.py tests/test_boundary_challenges.py tests/test_sentence_boundaries.py tests/test_datadownload.py -q
```

Các artifact trong `data_set/` và báo cáo trong `reports/` được tạo cục bộ; các file trong `tests/` cũng được giữ ngoài Git theo cấu hình repo. Git theo dõi code, notebook và tài liệu proposal/trạng thái/protocol đã công bố. Hai notebook trong `notebooks/` phân tích dữ liệu và hiển thị biểu đồ; notebook không thay đổi corpus.
