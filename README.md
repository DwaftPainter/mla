# NUCLE grammatical error-type classification

Phân loại **một lỗi tại vị trí đã biết trong câu tiếng Anh gốc** thành `ArtOrDet`, `Prep`, `Nn`, `Vform` hoặc `SVA`. Mỗi annotation là một sample, nên một câu có thể tạo nhiều sample. Project không tự tìm vị trí lỗi hoặc sinh câu sửa.

Đầu vào gồm `sentence`, `error_start`, `error_end`, `error_span`. Offsets là chỉ số ký tự Python, đầu có tính/cuối không tính: `sentence[start:end] == error_span`. LR/Tree nhận vector đặc trưng S/L/P/M/D; BERT-small nhận câu có marker span. Correction, nhãn và IDs không vào predictors. Thuật toán và công thức: [project guide](docs/note_project_guide.md).

## Trạng thái và kết quả

Đã hoàn thành preparation v1/v2, grouped split, baseline/ablation validation, khóa cấu hình, chấm test, bootstrap và tái lập **52 fit của đợt đánh giá** từ dữ liệu nguồn. BERT-small đã thử trên validation và chẩn đoán trên train; dừng thử thêm, chưa có điểm test BERT. Final report dự án và slide tổng kết chưa chốt.

| Model | Macro-F1 validation | Macro-F1 test | CI95% test |
|---|---:|---:|---|
| Majority | 0.1184 | 0.1184 | [0.1104; 0.1279] |
| POS rule | 0.7048 | 0.7106 | [0.6766; 0.7365] |
| LR Full | 0.8067 | 0.8163 | [0.7876; 0.8382] |
| Tree Full | 0.7879 | 0.7996 | [0.7693; 0.8246] |

Cấu hình chọn trên validation; giữ model fit train, không refit train+validation. Test có 2.368 samples thuộc 144 essays/97 connected groups; CI dùng 1.000 paired group-bootstrap, không đo biến thiên do seed huấn luyện. Bảng đủ chữ số: [reviewed_overall_results.csv](reports/evaluation_followup/reviewed_overall_results.csv). Kiểm chứng và giới hạn: [report rà soát](reports/evaluation_followup/review_report.md).

LR vốn có thông tin dependency từ parse toàn câu. Kết quả BERT chưa thành công không chứng minh ngữ cảnh vô ích. Audit có hỗ trợ AI; 17 ca vẫn chờ chuyên gia, chưa sửa gold.

## Đọc tài liệu theo nhu cầu

| Cần làm gì | Điểm bắt đầu |
|---|---|
| Xem đã làm gì và còn gì | [Trạng thái](docs/status.md), [kế hoạch](note.md) |
| Cài môi trường, chạy test, hiểu điều kiện tái lập | [Hướng dẫn chạy và tái lập](docs/reproduction.md) |
| Hiểu code, preparation v1/v2, feature và công thức | [Project guide](docs/note_project_guide.md) |
| Tra thiết kế, protocol và related work | [Mục lục docs](docs/README.md), [reference.md](reference.md) |
| Đọc báo cáo DOCX/Markdown và các mốc kết quả | [Mục lục reports](reports/README.md) |
| Xem code thừa và phạm vi dọn | [Code review](reports/maintenance/code_redundancy_review.md) |

## Cài và kiểm thử trong checkout mới

Các lệnh sau dành cho PowerShell tại thư mục gốc. Môi trường thực nghiệm dùng Python 3.13.7; `.python-version` chỉ ghi 3.13 nên chỉ định patch version khi cần đối chiếu.

```powershell
uv sync --locked --python 3.13.7
.venv/Scripts/python.exe -m pytest tests -q
```

Để chạy test BERT hoặc tạo DOCX, cài phần bổ sung vào môi trường đó:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-bert.txt
```

`uv.lock` khóa môi trường nền; `requirements-bert.txt` là phần bổ sung, không được lock nền khóa toàn bộ dependencies bắc cầu. Xem [tests/README.md](tests/README.md) về fixture riêng và các trường hợp skip. Các test BERT dùng vocabulary/model nhỏ tự tạo, không tải checkpoint pretrained.

## Luồng xử lý và phạm vi repository

```text
SGML có quyền sử dụng + review logs
  → Understanding / quality audit
  → preparation v2 → connected-group split
  → POS baseline / train-only features
  → LR và Tree: validation benchmark + ablation
  → selection và freeze → test → bootstrap / audit
  → kiểm chứng độc lập, tái lập và report task
```

Git chứa source, tests, notebooks, tài liệu, 5 báo cáo DOCX và các bảng/hình tổng hợp đã rà soát. Corpus `data_set/`, review inputs chứa nguồn, encoder/vocabulary, checkpoint, predictions từng sample và private audit packet giữ local. Clone repo đủ để đọc kết quả và chạy phần lớn unit tests, **chưa đủ để chạy lại toàn bộ nghiên cứu**.

Điều kiện đầu vào và cách ghi output của từng script ở [hướng dẫn tái lập](docs/reproduction.md). Có 40 file source đã khóa checksum; `.gitattributes` giữ nguyên byte để bảo toàn kiểm chứng. Notebook 01–04 là báo cáo các mốc dữ liệu/validation, không phải pipeline huấn luyện hay bảng test mới nhất.
