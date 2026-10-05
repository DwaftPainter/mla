# NUCLE — Data Understanding và Data Quality

Phạm vi hiện tại chỉ gồm hai bước dữ liệu. Mỗi annotation lỗi là một quan sát riêng; năm nhãn mục tiêu là ArtOrDet, Prep, Nn, Vform và SVA. Không tự gộp/xóa annotation vì chồng lấn, nghi trùng hoặc cùng span khác nhãn.

Dùng Python 3.12, chạy `uv sync --locked` để cài dependency. Đặt nguồn được cấp quyền tại `data_set/raw/nucle3.2.sgml`; utility tải tùy chọn là `python -m src.data.data_download`.

```powershell
uv run python -m src.data.data_understanding
uv run python -m src.data.data_quality
```

| Bước | Chức năng | Đầu ra |
| --- | --- | --- |
| Data Understanding | Đọc SGML, bảo toàn offset, mô tả schema, thống kê nhãn/span/edit và xuất toàn bộ annotation. | `data_set/interim/nucle_errors.csv`, `docs/nucle_data_understanding.md` |
| Data Quality | Đối chiếu CSV với SGML; kiểm tra missing, offset, no-op, nhãn khác cùng span, overlap, nghi trùng, chất lượng văn bản và mất cân bằng lớp. | `docs/nucle_data_quality.md`, `reports/nucle_data_quality.json`, `reports/nucle_annotation_audit.csv`, `reports/nucle_duplicate_content.json` |

`src/data` gồm hai entrypoint `data_understanding.py` và `data_quality.py`; `nucle_sgml.py`, `parse_nucle.py`, `constants.py` và `data_download.py` là helper đọc dữ liệu/đường dẫn/thu thập nguồn. `parse_nucle` không còn là bước CLI riêng.

Hai notebook tương ứng trong `notebooks/` sử dụng cùng code, mặc định chỉ đọc và hiển thị. Các cờ quality là ứng viên cần review theo annotation_id, chưa phải lỗi ngữ nghĩa đã xác nhận. Correction rỗng khác với thiếu thẻ correction. Hash văn bản trùng không chứng minh leakage và không phát hiện hết paraphrase.

Pipeline này không làm sạch/xóa mẫu, tách câu, POS, feature engineering, tạo sample câu/span, nhóm chia tập hay train/validation/test. Tên file nucle3.2.sgml không xác định release; quality lưu SHA256 và mặc định `release=unverified`, có thể truyền `--release` sau khi xác minh.

```powershell
uv run python -B -m pytest -p no:cacheprovider tests/test_nucle_understanding.py tests/test_nucle_quality.py tests/test_datadownload.py -q
```

Configs, models, training, evaluation, inference và checkpoints vẫn hoãn theo phạm vi trước; test demo model/training chưa được coi là chạy được.

Dependency được quản lý bằng `pyproject.toml` và `uv.lock`. Khi thay đổi dependency, cập nhật lockfile rồi đồng bộ môi trường:

```powershell
uv lock
uv sync --locked
```

Git theo dõi code, test, notebook không chứa output và duy nhất `docs/proposal.md` trong docs. Corpus, artifact và các báo cáo dữ liệu được giữ hoặc tạo lại tại máy.
