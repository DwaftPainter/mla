# Bổ sung kiểm tra hai span trong cùng câu

Đây là kiểm tra bổ sung sau ba lượt học thuộc; ghi trước khi chạy, không sửa protocol chẩn đoán ban đầu và không thay lựa chọn cấu hình.

Chỉ dùng train. Với mỗi câu có hai annotation khác nhãn và khác vị trí, lấy cặp đầu theo sample ID; sắp theo hash câu rồi shuffle seed20261023, chọn tối đa50 câu. Dùng checkpoint CLS của lượt BERT thất bại trước đây, không fit thêm.

Báo số cặp model đổi prediction, số cặp cả hai đúng và cosine giữa hidden CLS/mean span. Hai nhãn gold khác nhau không bảo đảm model phải đúng; đây là stress test có chủ đích, không đại diện toàn train. Không đưa câu nguồn vào report.

Cosine của span mean được đo trên encoder cũ; không dùng classifier chưa được huấn luyện cho mean span để so accuracy. So hai cách biểu diễn chỉ mô tả độ thay đổi, không chứng minh hiệu quả tổng quát hóa.
