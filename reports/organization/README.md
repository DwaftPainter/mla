# Tổ chức tài liệu — 23/10/2026

[layout_manifest.json](layout_manifest.json) lưu ánh xạ 13 file trước/sau, checksum và các file điều hướng/code đã cập nhật. `before_layout/` giữ bản sao trước sửa để phục hồi; đây là bản lưu lịch sử, không phải tài liệu hiện hành. [layout_verification.json](layout_verification.json) ghi kết quả kiểm tra sau sắp xếp.

- Chuyển phương pháp vào docs/methodology, literature vào docs/literature.
- Chuyển các báo cáo DOCX/Markdown để đọc vào reports/summaries theo chủ đề; bỏ hậu tố `.updated` của báo cáo LR/Tree.
- Bốn protocol đã đăng ký, 40 file code đã khóa, checkpoint, predictions và dữ liệu kết quả giữ nguyên.
- Các thư mục artifact giữ đường dẫn ổn định; không tạo junction/symlink hoặc bản sao báo cáo hoạt động để giả lập di chuyển.
- Snapshot cũ trong documentation_archive hoặc reproducibility zip không được sửa lại. Dùng manifest ánh xạ khi đọc tên file cũ trong nội dung lịch sử.

Các script báo cáo/kiểm tra hiện hành đã dùng đường dẫn mới. Các generator lịch sử bị khóa giữ code gốc; nếu chạy lại trong workspace tái lập, đầu ra của chúng có thể mang đường dẫn cũ và cần đối chiếu manifest này. Không chạy lại model chỉ để sắp xếp tài liệu.
