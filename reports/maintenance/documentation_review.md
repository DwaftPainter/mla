# Rà soát docs, README và báo cáo

Ngày 23/10/2026, đối chiếu với implementation và artifact của commit nền `52349bd`. Mục tiêu là giúp người đọc phân biệt thiết kế, kết quả đã chạy, điều kiện tái lập và công việc còn lại. Đây là báo cáo bảo trì tài liệu, không phải final report hoặc một lượt thí nghiệm mới.

## 1. Phạm vi đọc và đối chiếu

Đã rà 48 tài liệu có sẵn: 43 Markdown và 5 DOCX, gồm cả [code review](code_redundancy_review.md) từ lượt trước chưa commit. Đọc nội dung các DOCX và đối chiếu với README, code, manifests, metrics và báo cáo followup. Không coi tên file hoặc sự có mặt của file là đủ xác nhận nội dung.

Bổ sung ba Markdown: [runbook](../../docs/reproduction.md), [mục lục bảo trì](README.md) và báo cáo này. Sau cập nhật có 46 Markdown/5 DOCX trong phạm vi tài liệu công bố. Các bản sao trong `.cache`, `before_layout`, `documentation_archive` và zip tái lập là snapshot local, không tính thành tài liệu đang bảo trì.

| Nhóm đã rà | Cách xử lý |
|---|---|
| README gốc, docs/README, status, note | Làm rõ trạng thái, đường đọc, kết quả chính và việc còn lại. |
| Project guide, methodology và preparation | Đối chiếu class/function, input, split, phiên bản và phạm vi thực nghiệm. |
| Proposal và bốn protocol trước chạy | Đọc như thiết kế tại mốc cũ; giữ nguyên, dẫn tới trạng thái hiện hành từ mục lục. |
| Reference và literature | Đồng bộ trạng thái đã dùng/dự kiến, cải thiện bố cục, bỏ cách diễn đạt biến ghi chú Excel thành task còn lại. |
| README các thư mục artifact | Ghi rõ mốc validation/test, script tạo, input local và nguồn kết quả mới hơn. |
| Năm DOCX và các Markdown kết quả | Giữ nguyên snapshot; bổ sung lời giải thích trong README cùng nhóm. |
| Reproducibility và organization | Phân biệt manifest/zip lịch sử với docs hiện hành; không sửa manifest cũ. |
| Tests và code review | Làm rõ điều kiện chạy/skip, phạm vi từng mốc test và script đã ngừng dùng. |

## 2. Những điểm đã sửa

| Vấn đề | Thông tin được làm rõ | Nguồn hiện hành |
|---|---|---|
| README thiếu đường chạy và bảng kết quả mới nhất | Đầu vào là câu gốc + span đã biết; bảng validation/test và CI; setup, liên kết và phạm vi Git. | [README](../../README.md) |
| Hướng dẫn còn nói fresh environment chỉ kiểm inference | Followup đã dựng lại raw → preparation v2 → grouping → features → 52 fit → test replay. Chưa chạy lại toàn grid lịch sử/BERT. | [Runbook](../../docs/reproduction.md), [evidence](../evaluation_followup/pipeline_reproduction.json) |
| Tên encoder sai và dễ hiểu nhầm giới hạn vocabulary | Class đúng là `TrainOnlyEncoder`; Full 16.180 chiều, 500 áp dụng pooled n-gram; lexical slots có vocabulary riêng; numeric không z-score. | [Project guide](../../docs/note_project_guide.md), [implementation audit](../../docs/methodology/implementation_data_audit.md) |
| Preparation còn ghi split chưa khóa | Split đã khóa seed 20261013; bổ sung samples/essays/groups từng split và phân biệt tổng nguồn với tập có nhãn mục tiêu. | [Preparation](../../docs/methodology/preparation.md), project guide |
| Dễ đếm 20 pipeline thành 20 checkpoint riêng | 18 tên LR/Tree gồm 2 alias Full−D, tương ứng 16 checkpoint riêng; thêm Majority/POS. Đợt final là 40 grid fit + 12 refit = 52. | [Final evaluation](../final_evaluation/README.md), runbook |
| Ngày dự kiến dễ bị đọc thành ngày chạy thực tế | Lịch hai tuần là kế hoạch ban đầu; các task thực nghiệm đã chạy trong các phiên 23/10. Final report/slide chưa chốt. | [Note](../../note.md), [status](../../docs/status.md) |
| DOCX diagnostics còn khuyến nghị thử BERT tiếp | Đây là đề xuất ở mốc chẩn đoán; quyết định sau đó là dừng. Benchmark chưa thành công không chứng minh ngữ cảnh vô ích; LR đã dùng dependency. | [README contextual](../summaries/contextual/README.md) |
| Các mốc 13, 99 và 96 tests dễ gây mâu thuẫn | 13 là tập con trong lần tái lập; 99 là suite local trước commit nền; 96 pass/1 module skip là bản export thiếu fixture NUCLE. | [Tests README](../../tests/README.md) |
| Clone Git dễ bị hiểu là đủ input để tái lập toàn bộ | Corpus, review inputs, checkpoint, encoder và predictions từng sample giữ local. Runbook chỉ rõ điều kiện đầu vào và script có thể ghi lại output. | Runbook, [reports index](../README.md) |
| Updater/generator cũ dễ được dùng để cập nhật docs mới | Ghi updater đã ngừng dùng, generator lịch sử còn path/nội dung cũ; không chạy lại để đồng bộ tài liệu. | Runbook, [code review](code_redundancy_review.md) |

## 3. Cách đọc năm báo cáo DOCX

| Báo cáo | Phạm vi giữ nguyên | Phần cập nhật đọc kèm |
|---|---|---|
| [Data preprocessing](../summaries/data/data_preprocessing_report.docx) | Các bước dữ liệu tại mốc báo cáo; không phải báo cáo test. | [Data README](../summaries/data/README.md), preparation/project guide cho v1/v2 và split hiện hành. |
| [Modeling/ablation](../summaries/classical/modeling_ablation_results_report.docx) | Baseline, feature, ablation và masking validation. | [Classical README](../summaries/classical/README.md), nhóm evaluation cho test/CI. |
| [Contextual validation](../summaries/contextual/contextual_model_validation_report.docx) | Sensitivity, audit và BERT-small trên validation. | [Contextual README](../summaries/contextual/README.md), quyết định dừng ở status. |
| [BERT diagnostics](../summaries/contextual/bert_learning_diagnostics_report.docx) | Học thuộc train nhỏ, gradient/span/marker và khuyến nghị tại mốc đó. | Contextual README phân biệt accuracy với ngưỡng loss và quyết định sau chẩn đoán. |
| [Evaluation tasks](../summaries/evaluation/evaluation_tasks_report.docx) | Test/CI/audit và mức tái lập ở mốc báo cáo task. | [Evaluation README](../summaries/evaluation/README.md) và [followup](../evaluation_followup/review_report.md) cho rà soát/tái lập bổ sung. |

Không sửa nội dung DOCX để biến kết quả validation thành test hoặc biến ghi chép sau chạy thành protocol trước chạy. Báo cáo followup đã đối chiếu bảng DOCX/Markdown; bản DOCX được giữ nguyên byte để bằng chứng đối chiếu đó còn áp dụng.

## 4. Kiểm tra trước bàn giao

Kết quả kiểm trên tập 166 file công bố: **377 liên kết nội bộ hợp lệ, 0 liên kết đích thiếu; 74/74 input hashes và checksum lock khớp**. Bốn hàng test F1/CI trong README khớp artifact; 10 entry points có đầy đủ flags được mô tả. Không có thay đổi ở code, tests, notebook, DOCX, protocol hoặc các file kết quả đã đối chiếu.

- Kiểm tất cả Markdown links nội bộ trong tập file công bố; đích phải tồn tại và có trong repository, hoặc là thư mục hợp lệ. Không coi URL ngoài là đã được kiểm trực tuyến.
- Đối chiếu bốn hàng test macro-F1/CI trong README với `test_metrics.json` và `bootstrap.json`; giữ nguyên điểm và khoảng tin cậy gốc.
- Đọc khai báo CLI của 10 entry points argparse để kiểm flags trong runbook; đọc riêng nhánh điều khiển của các script final không có argparse. Không gọi `--help` hàng loạt, vì một số script đi thẳng vào chạy nếu không nhận đúng mode.
- Kiểm checksum lock và toàn bộ 74 input hashes; giữ nguyên bốn protocol, năm DOCX, 46 source files, 12 test files và bốn notebook.
- Giữ nguyên các JSON/CSV/ảnh kết quả đã công bố và Markdown báo cáo thí nghiệm. Giữ kiểu xuống dòng cũ ở những dòng không sửa để giảm diff không liên quan.
- Kiểm whitespace/diff và phạm vi file trước khi stage; commit gồm tài liệu và code-review report của lượt trước, không gồm implementation cleanup.

Không chạy lại unit tests hoặc thí nghiệm trong lượt sửa tài liệu. Số 99/96/13 tests là bằng chứng của các lượt trước, đã ghi đúng phạm vi; không được báo thành kết quả kiểm mới.

## 5. Giới hạn còn lại

17 ca audit vẫn chờ chuyên gia; không đổi gold. Corpus/private artifacts chưa có nơi sao lưu độc lập được xác nhận. Raw reproduction giữ cố định review logs và cấu hình lịch sử; chưa bao phủ toàn bộ lịch sử tuning hoặc BERT. Chưa xác nhận pipeline nguyên trạng trên Linux/macOS.

Lượt này không bổ sung bài báo mới và không kiểm lại metadata/URL học thuật trên web. Danh mục tham khảo là danh mục làm việc, không tự trở thành bibliography cuối. Những giới hạn nghiên cứu này vẫn được giữ trong status/report; chỉnh tài liệu không giải quyết thay chúng.
