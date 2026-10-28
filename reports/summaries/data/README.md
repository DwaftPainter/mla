# Dữ liệu và preprocessing

Nhóm báo cáo dữ liệu mô tả cách đọc NUCLE, kiểm tra chất lượng annotation, giữ đúng offset và chia train/validation/test theo nhóm bài viết. Các tệp thống kê giúp đối chiếu quy mô và các cờ cần xem lại; chúng chưa phải kết quả phân loại của mô hình.

| Tài liệu | Vai trò |
|---|---|
| [data_preprocessing_report.docx](data_preprocessing_report.docx) | Nguồn, schema, offsets, preparation v1/v2, grouping/split, code và công thức; có phần bổ sung theo mốc. |
| [nucle_data_understanding.md](nucle_data_understanding.md) | Thống kê annotation nguồn trước preparation. |
| [nucle_data_quality.md](nucle_data_quality.md) | Audit nguồn trước dựng dữ liệu/mô hình. |

## Bổ sung khi đọc báo cáo

Policy hiện hành là preparation v2. V1 là đối chứng segmentation; 15.788 target samples vẫn giữ nguyên. V2 thay input câu của 544 mẫu, không đổi source offsets/nhãn; sensitivity classifier là thí nghiệm riêng, xem nhóm contextual. Schema/offset được sở hữu bởi [preparation.md](../../../docs/methodology/preparation.md), còn thuật toán từng bước ở [project guide](../../../docs/note_project_guide.md).

DOCX giữ mốc trước đánh giá test; câu “test chưa chấm” trong đó không mô tả trạng thái hiện hành. Các tên docs/preparation.md hoặc report cũ được nhắc trong nội dung là đường dẫn lịch sử; dùng link trong mục lục này hoặc [mapping](../../organization/README.md).

Offset là chỉ số ký tự trên paragraph sau parse SGML, không phải byte trong markup. Không có span rỗng không có nghĩa không có lỗi thiếu từ. Không strip/chuẩn hóa text predictor để làm đẹp dữ liệu; chuẩn hóa tìm duplicate là bước riêng.

Artifact nguồn/private review logs giữ local. [So sánh v1/v2](../../nucle_preparation_v2_comparison/README.md) công bố bảng tổng hợp; điều kiện rebuild ở [hướng dẫn tái lập](../../../docs/reproduction.md). Fixture boundary chứa nguồn NUCLE không có trong Git.
