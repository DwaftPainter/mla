# Tái lập các task đánh giá NUCLE

Gói này gồm source, protocol, configs/lock metadata, bảng tổng hợp và báo cáo. Không chứa corpus, sentences/correction, encoder vocabulary, checkpoint hay private audit packet. Không phải bộ dữ liệu phân phối lại. Không tự upload.

## Bằng chứng đã kiểm

- Full/Mask train-only,12 refit cũ khớp mọi validation prediction.
- Lock có trước test;20 tên pipeline gồm aliases,18 predictors riêng.
- Reload checkpoint từ encoded test trong môi trường hiện tại và fresh venv khớp;20 metric và75 expanded-group checks khớp.
- Restore zip vào .cache/evaluation_tasks_bundle_restore và đối chiếu SHA256 từng file.
- Chưa chạy lại toàn grid hoặc spaCy từ raw trong fresh venv. Không gọi checkpoint replay là full training reproduction.

## Input cần có cho tái lập đầy đủ

NUCLE được cấp quyền đúng source hash; CSV Understanding, audit/review logs preparation; preparation v2 và split đã khóa; feature artifact; báo cáo/predictions validation sáu variants cũ. Hash cụ thể nằm ở reports/final_evaluation/lock.json. Các file bị loại khỏi zip phải được cung cấp qua nơi lưu local đã được phép. Không tạo lại split tùy ý, không sửa gold sau test.

## Môi trường

Môi trường nền: pyproject.toml + uv.lock, Python3.13.7, spaCy3.8.7, en_core_web_sm3.8.0. Tagger wheel: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl . Run thực tế còn có requirements-bert.txt; full versions ở environment.txt. Fresh inference chỉ cần NumPy2.5.3/PyTorch2.14.0+cpu và dependencies, xem clean_environment.json. Torch CPU lấy từ https://download.pytorch.org/whl/cpu; NumPy từ PyPI, tránh trộn index resolver không chủ ý.

## Chạy lại trên workspace đã có artifact

    .venv\Scripts\python.exe src/evaluation/nucle_final_evaluation.py replay
    .venv\Scripts\python.exe src/evaluation/verify_final_results.py
    .venv\Scripts\python.exe src/reporting/build_evaluation_tasks_report.py

Không chạy lại mode score lên output đã có; scorer từ chối. Replay chỉ kiểm bất biến, không model selection. Bootstrap từ saved predictions có thể tái tạo bằng mode bootstrap với cùng weights/seed. Khi thay môi trường, không sửa lock hoặc weights chỉ để cho kiểm tra pass.

## Tái tạo selection trong checkout/thư mục riêng

Chỉ khi đủ input có quyền truy cập và reports/final_evaluation chưa tồn tại:

    .venv\Scripts\python.exe src/training/nucle_final_selection.py
    .venv\Scripts\python.exe -m pytest tests/test_final_evaluation.py tests/test_nucle_features.py tests/test_nucle_benchmark.py -q -p no:cacheprovider
    .venv\Scripts\python.exe src/training/nucle_final_selection.py --freeze
    .venv\Scripts\python.exe src/evaluation/nucle_final_evaluation.py score
    .venv\Scripts\python.exe src/evaluation/nucle_final_evaluation.py bootstrap

Full−D phải tái sử dụng S+L+P+M. Test hiện đã được xem nên lượt tái tạo chỉ là reproduction của thiết kế cũ, không phải test mới độc lập. Script audit tạo packet trước; ghi chú AI đọc packet là artifact review, không thể tái lập bằng một câu lệnh như metric.

## Đọc artifact

selection_summary.json/fit_progress.json:52 fits và config; lock.json/lock.sha256:đóng băng trước test; test_predictions.csv:sample×pipeline (giữ local, ngoài zip); test_metrics.json:counts/F1; bootstrap.json:paired CI; manual_error_review.json:40 ghi chú text-free; report task DOCX/Markdown:kết quả các task vừa chạy; final report dự án để sau. Full_minus_D alias không được cộng thành lượt fit độc lập.
