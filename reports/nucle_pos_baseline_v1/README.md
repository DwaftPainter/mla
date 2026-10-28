# Majority và luật POS — validation

Majority và POS rule là hai đối chứng ban đầu của dự án. Majority luôn chọn lớp phổ biến nhất trong train; POS rule kết hợp các ánh xạ từ loại cố định với bảng tra nhãn học từ train. Thư mục này lưu kết quả validation và các tệp cần để kiểm tra lại hai đối chứng đó.

| File | Vai trò |
|---|---|
| [summary.json](summary.json) | Contract POS, phiên bản tagger và metric |
| `validation_predictions.csv` (local) | Dự đoán validation đã lưu |
| [pos_by_label_train.csv](pos_by_label_train.csv) | Thống kê POS theo nhãn train |

Luật fit từ train. Đây là snapshot validation; test nằm ở final_evaluation.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

POS baseline dùng span UPOS signature; ADP→Prep, DET→ArtOrDet, còn lại majority theo signature học train. Majority chung cũng học train; không dùng lexical hoặc context ngoài span trong baseline này.

Đây là validation. Bảng POS test nằm ở final_evaluation; không thay summary cũ bằng điểm test.

Entry point: [src/features/nucle_pos.py](../../src/features/nucle_pos.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
