# Trạng thái dự án

Cập nhật ngày 23/10/2026, sau các lượt thí nghiệm, tái lập và rà soát code/tài liệu. Đây là nguồn trạng thái hiện hành; [note.md](../note.md) giữ kế hoạch, [reproduction.md](reproduction.md) hướng dẫn chạy. Các báo cáo cũ ghi trạng thái tại mốc của chúng.

## Đã hoàn thành

| Phần việc | Bằng chứng và giới hạn |
|---|---|
| Dữ liệu | 44.912 annotations nguồn; 15.788 samples năm nhãn. Preparation v2 và grouping/split đã khóa. [Đặc tả](methodology/preparation.md). |
| Validation | POS/Majority, LR/Tree Full, sáu variants và masking; encoder fit train, chọn bằng validation. [Báo cáo cổ điển](../reports/summaries/classical/README.md). |
| BERT-small | Bốn lượt fine-tuning validation; ba seed ở learning rate chọn. Diagnostics chỉ train: học thuộc, gradient, span/marker. Dừng thử thêm; không có BERT test. [Báo cáo](../reports/summaries/contextual/README.md). |
| Selection và freeze | 40 grid fit Full−P/M + 12 refit cũ; Full−D reuse S+L+P+M. [Protocol trước chạy](final_evaluation_protocol.md), [lock](../reports/final_evaluation/lock.json). |
| Test và CI | 20 tên pipeline, gồm 18 LR/Tree có 2 aliases, cùng Majority/POS; 16 checkpoint LR/Tree riêng. Test 2.368 samples/144 essays/97 groups; 1.000 paired bootstrap. [Report task](../reports/summaries/evaluation/README.md). |
| Audit lỗi | 40 ca theo ID sau prediction; review có hỗ trợ AI. Kiểm nguồn/offset/parser đã khớp; 17 ca chờ chuyên gia. [Queue](../reports/evaluation_followup/expert_review_queue.csv). |
| Tái lập | Raw → preparation v2 → grouping → NLP/features → 52 fit → test replay trong môi trường riêng; data/checkpoints/predictions khớp. [Report kiểm chứng](../reports/evaluation_followup/review_report.md). |
| Kiểm số liệu | 280 model–slice metrics, 40 CI F1, 30 CI chênh lệch; bảng Markdown/DOCX và CSV đã đối chiếu. [Evidence](../reports/evaluation_followup/evidence_review.json). |
| Bàn giao | Code, docs, 5 DOCX và aggregate results đã đưa lên Git; corpus/checkpoint/private artifacts local. [Phạm vi](../reports/README.md). |
| Bảo trì | Rà 46 source, 12 test, 4 notebook; xác định updater cũ và dead code, chưa xóa/sửa implementation. [Code review](../reports/maintenance/code_redundancy_review.md). |

## Các câu hỏi nghiên cứu

| Câu hỏi | Tình trạng bằng chứng |
|---|---|
| RQ1: LR/Tree so với Majority | Đã có benchmark test, từng lớp, CI và paired differences; POS là đối chứng bổ sung. |
| RQ2: Đóng góp feature | Đã có chín tên variant/model, chuỗi tăng dần, masking, non-lexical và Full−P/M/D; Full−D là alias. |
| RQ3: Nhóm nhầm và đặc điểm liên quan | Đã có confusion/slices/audit; diễn giải ngôn ngữ vẫn bị giới hạn bởi review có hỗ trợ AI. |

Đã đủ đầu ra thí nghiệm trong phạm vi chốt để tổng hợp các RQ; không đồng nghĩa final report dự án đã được viết. Kết luận và giới hạn theo từng RQ ở [report rà soát](../reports/evaluation_followup/review_report.md).

## Chưa hoàn thành hoặc không thuộc kết quả hiện có

- Chưa adjudication độc lập 17 ca; nếu chưa có chuyên gia, giữ nguyên gold và công bố giới hạn audit.
- Chưa rerun toàn grid lịch sử sáu variants/BERT trong môi trường sạch; review logs/config lịch sử vẫn là input cố định.
- Chưa có BERT test, BERT-base hoặc bằng chứng mọi mô hình ngữ cảnh đều kém LR.
- Chưa xác nhận nơi sao lưu corpus/private artifacts độc lập ngoài workspace. Push code/docs không thay thế việc này.
- Final report dự án và slide tổng kết để sau theo phạm vi người dùng đã chốt. Bản nháp trong documentation_archive không phải bản nộp.

Test đã mở: không đổi feature/config/seed/nhãn theo điểm test rồi báo lại như đánh giá độc lập. Việc dọn code còn là đề xuất bảo trì; 40 source files đã nằm trong lock.

## Cách đọc số lượng tests

13 tests trong bằng chứng tái lập là tập con chạy trong môi trường riêng. Lượt kiểm toàn repo trước commit 52349bd có 99 pass; bản export thiếu fixture NUCLE có 96 pass/1 module skip. Chúng khác phạm vi, không mâu thuẫn và không phải số lần chạy mới trong lượt chỉnh tài liệu.
