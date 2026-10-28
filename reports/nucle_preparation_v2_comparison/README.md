# So sánh preparation v1/v2

Phần so sánh này kiểm tra hai cách chuẩn bị dữ liệu v1 và v2 có làm thay đổi câu, ngữ cảnh hoặc cách biểu diễn span hay không. ID, nhãn và offset nguồn được giữ để có thể đối chiếu từng mẫu. Đây là kiểm tra dữ liệu; mức ảnh hưởng lên mô hình được đánh giá trong một thí nghiệm riêng.

| File | Vai trò |
|---|---|
| [comparison_summary.json](comparison_summary.json) | Tổng hợp khác biệt dữ liệu và input |
| [fresh_selection.json](fresh_selection.json) | IDs của mẫu review bổ sung |

Các JSONL còn lại chứa trường hợp regression/review và có thể có câu nguồn; giữ local. Đây là so sánh dữ liệu, không phải kết quả test classifier.

[Trở về mục lục reports](../README.md).

## Cách đọc và điều kiện chạy

So sánh cùng 15.788 annotations: 544 input câu đổi, 533 sentence IDs đổi, 30 context đổi và 2 token span đổi; các tập này chồng lấp. Giữ source offsets/nhãn.

Đây là đối chiếu dữ liệu/cơ học. Sensitivity classifier v1/v2 ở contextual_model_validation là bước khác. Packet có text nguồn giữ local.

Entry point: [src/data/compare_preparation.py](../../src/data/compare_preparation.py). Các output/inputs còn lại có thể chỉ local; xem [hướng dẫn tái lập](../../docs/reproduction.md) trước khi chạy. Trạng thái chung: [status](../../docs/status.md).
