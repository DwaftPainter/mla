# Chạy kiểm thử

Chạy từ root trong checkout/môi trường đã chuẩn bị theo [hướng dẫn tái lập](../docs/reproduction.md):

```powershell
uv sync --locked --python 3.13.7
.venv/Scripts/python.exe -m pytest tests -q
```

Hai module BERT cần transformers trong requirements-bert.txt:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-bert.txt
.venv/Scripts/python.exe -m pytest tests -q
```

## Phạm vi và skip

| Nhóm | Cần gì / hành vi |
|---|---|
| Parser, quality, preparation, feature, benchmark, final evaluation | Dữ liệu tổng hợp; spaCy và model package theo môi trường nền. Không tải corpus. |
| test_datadownload.py | Mock HTTP để kiểm retry/partial file; không tải mạng. |
| test_bert_span_classifier.py | Vocabulary WordPiece tự tạo; không cần pretrained checkpoint. Skip module nếu thiếu transformers. |
| test_bert_readout.py | BERT config nhỏ khởi tạo ngẫu nhiên để kiểm gradient/readout. Skip module nếu thiếu transformers. |
| test_boundary_challenges.py | Fixture local tests/fixtures/nucle_boundary_challenges.jsonl chứa các đoạn NUCLE được cấp quyền; skip cả module nếu không có. |

Khi công bố code ở 52349bd: workspace đầy đủ có **99 pass**; bản export không có fixture NUCLE có **96 pass, 1 module skip**, cùng môi trường có transformers. `reports/evaluation_followup/tests.xml` chỉ lưu 13 tests của lượt tái lập trước đó; không phải báo cáo của toàn suite. Các số này ghi lại lượt đã kiểm, không thay cho kết quả chạy trên máy mới.

Tests không chứng minh parser đúng ngữ nghĩa, không thay review chuyên gia và không rerun toàn bộ thí nghiệm. Không tái phân phối fixture NUCLE để tránh skip.
