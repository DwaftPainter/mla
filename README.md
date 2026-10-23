# NUCLE grammatical error-type classification

Pipeline xây dựng dữ liệu và benchmark phân loại năm nhãn `ArtOrDet`, `Prep`, `Nn`, `Vform`, `SVA` trên annotation có span lỗi đã biết. Mỗi annotation là một mẫu riêng. Hiện đã có preparation v1/v2, group-aware split, feature S/L/P/M/D, POS baseline, LR/Tree benchmark và validation ablation/masking. Test đã chấm sau khóa cấu hình; đã kiểm tái lập đợt đánh giá từ dữ liệu nguồn. Final report và slide tổng kết để sau.

Mục lục: [tài liệu phương pháp](docs/README.md) · [báo cáo và artifact](reports/README.md).

## Tài liệu chính

| File | Vai trò |
| --- | --- |
| [docs/proposal.md](docs/proposal.md) | Câu hỏi nghiên cứu và phạm vi |
| [docs/methodology/preparation.md](docs/methodology/preparation.md) | Schema, segmentation, offset và alignment |
| [docs/methodology/experiment_protocol.md](docs/methodology/experiment_protocol.md) | Split, feature, tuning và đánh giá |
| [docs/status.md](docs/status.md) | Kết quả hiện tại và việc tiếp theo |
| [note_project_guide.md](docs/note_project_guide.md) | Hướng dẫn chạy pipeline và train |

Đặt corpus được cấp quyền tại `data_set/raw/nucle3.2.sgml`. Dùng Python 3.12 hoặc 3.13.7; cài môi trường bằng `uv sync --locked`. Các lệnh dưới đây chạy từ thư mục gốc project.

## Pipeline và entry points

```powershell
py src/data/data_understanding.py
py src/data/data_quality.py
py src/data/prepare_nucle.py
py src/data/group_split_nucle.py
py src/features/nucle_pos.py
py src/features/nucle_features.py
py src/training/nucle_benchmark.py --output reports/nucle_feature_benchmark_my_run
py src/training/nucle_ablation.py --output reports/nucle_feature_ablation_my_run
py src/reporting/build_ablation_notebook.py
```

Preparation chuẩn v2 nằm ở `data_set/processed/nucle_preparation_v2/`; split chuẩn ở `data_set/split/nucle_group_split/`; feature chuẩn ở `data_set/features/nucle_features_v1/`. Những artifact này đã được dựng trong workspace. Script từ chối ghi đè output đã tồn tại; chỉ chạy lại bước tạo artifact khi có chủ đích và cung cấp đường dẫn output mới. Corpus, dữ liệu dẫn xuất và artifact riêng tư giữ local. Code, tests, tài liệu, báo cáo để đọc và các bảng kết quả được chọn có trong Git; xem phạm vi ở reports/README.md.

`src/models/nucle_logistic.py` và `src/models/nucle_tree.py` chứa implementation của classifier. `src/training/nucle_benchmark.py` và `nucle_ablation.py` điều phối fit, validation selection và lưu summary/predictions. `src/evaluation/metrics.py` định nghĩa metric NUCLE; `src/reporting/build_ablation_notebook.py` dựng notebook từ report cục bộ.

## Kết quả và kiểm thử

Validation benchmark hiện tại: LR Macro-F1 0.8067, Tree 0.7879, POS rule 0.7048 và Majority 0.1184. Báo cáo benchmark ở `notebooks/03_data_visualization.ipynb`; ablation/masking ở `notebooks/04_feature_ablation_report.ipynb`. Không dùng holdout test để chọn model.

```powershell
py -B -m pytest -p no:cacheprovider --basetemp=.pytest-tmp tests -q
```

`tests/` được quản lý bằng Git; xem tests/README.md về dependency BERT và fixture NUCLE chỉ có local. `data_set/`, checkpoint, encoder, private audit packet và backup lịch sử bị ignore. `.gitattributes` giữ nguyên byte của các file để checksum đã khóa không bị thay đổi bởi chuyển đổi xuống dòng. Notebook 01–04 lần lượt ghi nhận hiểu dữ liệu, chất lượng, phân tích benchmark và ablation; notebook là báo cáo, không phải entry point huấn luyện.
