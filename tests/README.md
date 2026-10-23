# Chạy kiểm thử

Tại thư mục gốc, cài môi trường bằng `uv sync --locked`, rồi chạy:

```powershell
.venv/Scripts/python.exe -m pytest tests -q
```

Các test dùng dữ liệu tổng hợp. Nhóm BERT cần dependencies bổ sung trong `requirements-bert.txt`; nếu chưa cài `transformers`, pytest báo skip hai module BERT. Tokenizer test dùng vocabulary tự tạo, không cần tải checkpoint.

Riêng `test_boundary_challenges.py` kiểm các đoạn nguồn NUCLE đã được review. Fixture `tests/fixtures/nucle_boundary_challenges.jsonl` giữ local theo quyền sử dụng corpus; module này báo skip khi clone repository mà chưa có fixture. Workspace nghiên cứu có fixture sẽ chạy đủ các ca đó. Không tải hoặc công bố corpus qua test.
