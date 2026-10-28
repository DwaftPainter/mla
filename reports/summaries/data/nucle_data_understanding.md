# Tổng quan dữ liệu NUCLE

## 1. Quy mô dữ liệu

- Tổng số annotation: **44,912**
- Số bài viết có annotation: **1,397**
- Số loại lỗi trong nguồn: **28**
- Số annotation thuộc năm nhãn nghiên cứu: **15,788**
- Offset không hợp lệ: **0**
- Annotation đi qua nhiều đoạn văn: **4**

## 2. Phân bố năm loại lỗi được nghiên cứu

| error_type | count | percentage | essays_with_class |
| --- | --- | --- | --- |
| ArtOrDet | 6640 | 42.06 | 1178 |
| Prep | 2413 | 15.28 | 954 |
| Nn | 3768 | 23.87 | 1027 |
| Vform | 1443 | 9.14 | 690 |
| SVA | 1524 | 9.65 | 746 |

## 3. Các dạng chỉnh sửa trong tập năm nhãn

| edit_type | count |
| --- | --- |
| replacement | 12821 |
| deletion | 2967 |

## 4. Số lỗi mục tiêu trong mỗi bài viết

- Số bài có ít nhất một lỗi mục tiêu: **1,355**
- Số lỗi trung bình mỗi bài: **11.65**
- Trung vị số lỗi mỗi bài: **9.00**
- Nhỏ nhất: **1**
- Lớn nhất: **62**

## 5. Độ dài đoạn lỗi

Độ dài dưới đây được tính theo ký tự, chỉ trên những span hợp lệ nằm trong một đoạn văn.
Số mẫu mục tiêu không nằm trong phép tính: **0**.
Span rỗng chỉ được tính khi việc dựng lại đoạn văn bản từ offset đã thành công:

- Trung bình: **5.86**
- Trung vị: **5.00**
- Nhỏ nhất: **1**
- Lớn nhất: **156**
- Span rỗng: **0**

## 6. Cách hiểu các thống kê

Mỗi annotation được tính là một mẫu có nhãn riêng, nên số mẫu không bằng số câu độc lập. Bước thống kê giữ lại toàn bộ annotation và offset nguồn, kể cả những trường hợp cần rà soát chất lượng. Các cờ về dữ liệu thiếu, span chồng lấp và nội dung trùng được kiểm tra ở bước data_quality tiếp theo. Phần này chưa tách câu, gán POS, trích đặc trưng, lọc mẫu hoặc chia train/validation/test.
