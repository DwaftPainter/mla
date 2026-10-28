# Báo cáo để đọc

Nếu cần đọc toàn bộ kết quả, hãy bắt đầu từ báo cáo tổng kết ở nhóm 0. Các nhóm tiếp theo giải thích từng giai đoạn dữ liệu, mô hình, chẩn đoán và đánh giá. Kết quả BERT mở rộng ngày 28/10 được tách riêng ở nhóm 5 để không lẫn với đợt BERT ban đầu.

| Thứ tự | Nhóm | Mốc nội dung |
|---|---|---|
| 0 | [Tổng kết Markdown](final/bao_cao_tong_ket_NUCLE.md), [DOCX](final/bao_cao_tong_ket_NUCLE.docx) | Tổng hợp kết quả đã lưu đến 23/10, cập nhật báo cáo 26/10/2026 |
| 1 | [Dữ liệu](data/README.md) | Nguồn, preprocessing, offset, grouping/split |
| 2 | [Mô hình cổ điển](classical/README.md) | Đặc trưng, LR/Tree, ablation/masking validation |
| 3 | [Mô hình ngữ cảnh](contextual/README.md) | BERT validation và chẩn đoán; thử nghiệm chưa thành công |
| 4 | [Đánh giá và rà soát](evaluation/README.md) | Test, CI, audit và kiểm tái lập mới nhất |
| 5 | [BERT mở rộng](../bert_final_analysis/analysis_report.md) | Kết quả ngày28/10: primary test0,8146; mean ba seed0,8182 ±0,0038 |

Đọc nhanh [trạng thái](../../docs/status.md), [bàn giao PDF/slide](final/handover_checklist.md) và [kết quả BERT mới](../bert_final_analysis/analysis_report.md). Các báo cáo cũ giữ mốc thí nghiệm của chúng; câu “test chưa chấm” không mô tả trạng thái BERT hiện tại. Tên/MSSV, đóng góp thực tế và xác nhận nộp bài vẫn cần nhóm hoàn thiện.

## Thứ tự mốc và cách đọc bổ sung

1. Dữ liệu/preprocessing và LR/Tree validation: báo cáo phương pháp cùng kết quả trước test.
2. Contextual validation: BERT-small chưa thành công và sensitivity v1/v2.
3. BERT diagnostics: chứng minh học được train nhỏ, không đo validation/test. Đợt mở rộng28/10 đã chạy và báo riêng ở nhóm 5.
4. Evaluation tasks: đã có test/CI/audit, tại mốc mới kiểm checkpoint replay.
5. Evaluation followup: bổ sung tái lập raw → 52 fit → test và kiểm độc lập bằng chứng.

Năm DOCX theo giai đoạn và DOCX tổng kết được đọc theo mốc tạo. README từng nhóm bổ sung scope, đường dẫn và trạng thái; không thay điểm hay bảng lịch sử. Các tên file cũ trong DOCX được đối chiếu bằng [organization](../organization/README.md). Report task và review khác với báo cáo tổng kết ở `final/`.
