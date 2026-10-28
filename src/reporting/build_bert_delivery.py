"""Build updated Vietnamese reports and handover artifacts from saved evidence.

Uses a separate export environment (requirements-reporting.txt). No training.
The pre-update final Markdown/DOCX are archived once before replacement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

from src.evaluation.bert_final_analysis import ROOT, csv_read, csv_write, dump, read, sha
from src.reporting.document_export import export_docx, export_pdf

FINAL = ROOT / "reports/summaries/final"
ANALYSIS = ROOT / "reports/bert_final_analysis"
EXPERIMENT = ROOT / "reports/bert_small_experiment"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def replace_table(text, number, replacement):
    pattern = rf"(^Bảng {number}\. [^\n]+\n\n)(?:\|[^\n]*\n)+"
    text, count = re.subn(pattern, lambda m: m[1] + replacement + "\n", text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Cannot find table {number}")
    return text


def require_replace(text, old, new):
    if old not in text:
        raise ValueError(f"Missing report anchor: {old[:80]}")
    return text.replace(old, new, 1)


def build_report():
    overall = csv_read(ANALYSIS / "overall_results.csv")
    summary = read(ANALYSIS / "summary.json")
    bootstrap = read(ANALYSIS / "bootstrap.json")
    audit = read(ANALYSIS / "audit_summary.json")
    grid = csv_read(ANALYSIS / "validation_grid.csv")
    seeds = summary["test_seeds"]
    backup = FINAL / "documentation_archive/before_bert_delivery_20261028"
    backup.mkdir(parents=True, exist_ok=True)
    for filename in ("bao_cao_tong_ket_NUCLE.md", "bao_cao_tong_ket_NUCLE.docx", "source_manifest.json"):
        if (FINAL / filename).exists() and not (backup / filename).exists():
            shutil.copy2(FINAL / filename, backup / filename)
    text = (backup / "bao_cao_tong_ket_NUCLE.md").read_text(encoding="utf-8-sig")
    text = require_replace(text,
        "Cập nhật ngày 26/10/2026. Báo cáo sử dụng kết quả thực nghiệm và kiểm chứng đã lưu đến ngày 23/10/2026. Lượt cập nhật này tổng hợp tài liệu, không huấn luyện lại hoặc chấm test mới. Ngôn ngữ: tiếng Việt.",
        "Cập nhật ngày 28/10/2026. Kết quả cổ điển và chẩn đoán ban đầu thuộc đợt 23/10; BERT-small mở rộng đã được người dùng train, khóa cấu hình và chấm test ngày 28/10. Lượt tổng hợp này chỉ đọc predictions/checkpoints đã lưu, tính bootstrap và xuất báo cáo, không train hoặc gọi lại model scorer lên test. Ngôn ngữ: tiếng Việt.")
    text = require_replace(text,
        "BERT-small đạt validation macro-F1 trung bình 0,1984 ± 0,0030 trên ba seed và chưa được chấm test. Các phép chẩn đoán cho thấy mô hình có gradient và có thể học thuộc tập nhỏ, nhưng chưa chứng minh khả năng tổng quát hóa tốt.",
        "BERT-small đợt ban đầu chỉ đạt validation macro-F1 0,1984 ± 0,0030. Đợt mở rộng so CLS/span mean với LR encoder/head riêng và early stopping chọn span mean: validation trung bình 0,8183 ± 0,0011; test trung bình 0,8182 ± 0,0038 qua ba seed. Seed chính đã khóa đạt test macro-F1 0,8146 và accuracy 0,8438; BERT−LR là −0,0017, CI95% [−0,0184; 0,0182], chưa đủ bằng chứng mô hình nào tốt hơn. Mean/std seed khác với bootstrap CI.")
    text = text.replace("thí nghiệm BERT nhỏ có kết quả âm được báo cáo đầy đủ", "BERT-small gồm kết quả âm ban đầu và đợt mở rộng có khóa lựa chọn được báo cáo đầy đủ")
    text = text.replace("Audit ngôn ngữ có hỗ trợ AI; 17 ca còn chờ chuyên gia.",
                        "Audit ngôn ngữ có hỗ trợ AI; hàng đợi hợp nhất sau đợt mới có 22 IDs còn chờ chuyên gia.")
    text = text.replace("Ưu tiên bằng chứng: reviewed_overall_results.csv", "Ưu tiên bằng chứng BERT mở rộng: reports/bert_final_analysis (verification, summary, bootstrap, bảng/hình) và delivery_manifest.json ngày28/10. Với kết quả cổ điển, ưu tiên reviewed_overall_results.csv")
    text = text.replace("Không có BERT test.", "Đợt ban đầu không có BERT test; kết quả test của đợt mở rộng ngày 28/10 được trình bày ở mục 9.2.")
    text = text.replace("# 9. Thử nghiệm BERT-small và chẩn đoán", "# 9. BERT-small: đợt ban đầu, chẩn đoán và đợt mở rộng")
    text = text.replace("Nguồn: docs/contextual_model_validation_report.md; bert run artifacts", "Nguồn: reports/summaries/contextual/contextual_model_validation_report.docx; bert run artifacts đợt cũ")
    text = text.replace("docs/bert_learning_diagnostics_report.md", "reports/summaries/contextual/bert_learning_diagnostics_report.docx")
    text = text.replace("Hình 7. BERT-small và đối chứng trên cùng validation; không phải điểm test.", "Hình 7. Đợt BERT ban đầu và đối chứng trên validation; không phải điểm test hoặc đợt mở rộng.")
    text = text.replace("Hình 2. Kết quả test và khoảng tin cậy từ 1.000 paired group-bootstrap.", "Hình 2. Các mô hình cổ điển và CI từ 1.000 paired group-bootstrap; hình10 bổ sung BERT.")
    text = text.replace("Hình 3. So sánh F1 từng nhãn trên test.", "Hình 3. F1 từng nhãn của các mô hình cổ điển; hình12 bổ sung BERT.")
    text = replace_table(text, 11, table(
        ["Mô hình", "Val F1", "Test F1", "CI95% test", "Accuracy", "Macro P", "Macro R"],
        [[row["model"], f"{float(row['validation_macro_f1']):.4f}", f"{float(row['test_macro_f1']):.4f}",
          f"[{float(row['ci95_low']):.4f}; {float(row['ci95_high']):.4f}]", f"{float(row['accuracy']):.4f}",
          f"{float(row['macro_precision']):.4f}", f"{float(row['macro_recall']):.4f}"] for row in overall]))
    text = text.replace("Nguồn: reports/evaluation_followup/reviewed_overall_results.csv\n\n![Hình 2.",
                        "Nguồn: reports/bert_final_analysis/overall_results.csv. BERT dùng seed chính 20261013 đã khóa; mean test ba seed là 0,8182 ± 0,0038, không dùng seed tốt nhất để thay đại diện.\n\n![Hình 2.", 1)
    old_differences = csv_read(ROOT / "reports/final_evaluation/paired_differences.csv")
    old_rows = []
    for row in old_differences:
        if row.get("slice") == "all" and row.get("contrast") in ("LR Full - Majority", "Tree Full - Majority", "LR Full - Tree Full", "LR Full - POS"):
            old_rows.append([row["contrast"], f"{float(row['difference']):.4f}", f"[{float(row['ci95_low']):.4f}; {float(row['ci95_high']):.4f}]"])
    if not old_rows:
        # Values read directly from the immutable original bootstrap JSON.
        old = read(ROOT / "reports/final_evaluation/bootstrap.json")
        old_rows = [[r["contrast"], f"{r['difference']:.4f}", f"[{r['ci95'][0]:.4f}; {r['ci95'][1]:.4f}]"]
                    for r in old["differences"] if r["slice"] == "all" and r["contrast"] in
                    ("LR Full - Majority", "Tree Full - Majority", "LR Full - Tree Full", "LR Full - POS")]
    new_rows = [[r["contrast"], f"{r['difference']:.4f}", f"[{r['ci95'][0]:.4f}; {r['ci95'][1]:.4f}]"] for r in bootstrap["differences"]]
    text = replace_table(text, 12, table(["So sánh", "Δ macro-F1", "CI95%"], old_rows + new_rows))
    text = require_replace(text, "LR Full dẫn đầu benchmark chính, hơn Tree", "LR Full dẫn đầu các model cổ điển và gần BERT seed chính; chưa đủ bằng chứng phân biệt BERT−LR/Tree vì CI chứa 0. Trong nhóm cổ điển, LR hơn Tree")
    text = replace_table(text, 14, table(["Nhãn", "Majority", "POS rule", "LR Full", "Tree Full", "BERT primary"],
        [[label] + [f"{summary['metrics'][name]['per_label'][label]['f1']:.4f}" for name in ("Majority", "POS", "LR_03", "Tree_03", "BERT")]
         for label in ("ArtOrDet", "Prep", "Nn", "Vform", "SVA")]))
    text = text.replace("Bảng 14. F1 từng lớp của bốn đối chứng", "Bảng 14. F1 từng lớp của các mô hình chính")
    text = text.replace("LR Full là mô hình chính mạnh nhất: test F1 0,8163; hơn Tree/POS với paired CI dương.",
                        "LR Full test F1 0,8163; hơn Tree/POS với paired CI dương. BERT primary 0,8146, CI BERT−LR chứa 0; chưa đủ bằng chứng mô hình nào tốt hơn.")
    text = text.replace("Kết quả BERT âm là một kết quả thực nghiệm cần giữ, không phải lý do bỏ ngữ cảnh.",
                        "Kết quả BERT âm của đợt đầu cần giữ cùng kết quả đợt mở rộng: span mean đạt test gần LR, chưa có bằng chứng vượt LR. Điều này cho thấy kết luận phụ thuộc recipe đã khảo sát.")
    text = text.replace("chưa xác nhận giải pháp.", "đợt mở rộng kiểm readout và tối ưu hóa trong cùng grid; vẫn chưa cô lập nguyên nhân của mọi khác biệt giữa hai đợt.")
    text = text.replace("| BERT-small chưa thành công | Chưa kiểm lợi ích contextual mạnh hơn | So CLS/span_mean cùng ngân sách, theo dõi learning curve; chưa dùng test cũ để tune. |",
                        "| Một BERT nhỏ và test đã dùng trong nghiên cứu cổ điển | Kết quả gần LR không chứng minh mọi contextual model tương đương | External test mới; đăng ký trước thiết kế và không tune theo test đã xem. |")
    text = text.replace("BERT-small chỉ có kết quả validation và chẩn đoán train, chưa có điểm test.",
                        "BERT-small đã hoàn tất 12 run mới, khóa span mean bằng validation và chấm test ba seed; primary test F1 0,8146, mean 0,8182 ± 0,0038.")
    text = text.replace("Nguồn: Code hiện có; danh mục từng file và checksum trong source_manifest.json",
                        "Nguồn: Code hiện có; source_manifest.json giữ snapshot 26/10. delivery_manifest.json ghi nguồn/output của bản cập nhật 28/10; không viết lại hash lịch sử.")
    audit_section = f"""## 8.1. Audit bổ sung BERT và LR sau test

Trên cùng 2.368 mẫu, hai model cùng đúng {summary['error_overlap']['both_correct']} ca; BERT đúng riêng {summary['error_overlap']['BERT_only_correct']}, LR đúng riêng {summary['error_overlap']['LR_only_correct']}; cùng sai {summary['error_overlap']['both_wrong']}. Đây là thống kê toàn test, khác với tần suất của mẫu audit có chủ đích.

Mẫu mới gồm 40 ca: 12 BERT sai/LR đúng (ưu tiên sáu gold SVA), 12 chiều ngược, 10 cùng sai, sáu controls cùng đúng. IDs được chọn bằng seed 20261028 trước khi đọc câu/correction. Cả 40 gold/span khớp SGML; có 40 ghi chú hỗ trợ AI, còn cột review/contribution của nhóm để trống. Nhóm cả hai sai được chọn từ bucket sắp xếp nên có 9 ArtOrDet và 1 Nn; không dùng mẫu này ước lượng prevalence.

Nhận xét nổi bật: SVA cần xét chủ ngữ mệnh đề hoặc động từ phối hợp; missing-word annotation có thể neo vào danh từ, động từ hoặc mảnh từ; substring không luôn là token độc lập. Một số nhãn/correction cần chuyên gia xem thêm, gồm ID 23037, 34539, 16809, 30366, 32801, 32874. Không đổi gold, không quy nguyên nhân cho attention/POS/truncation khi chưa có phép kiểm tương ứng. Control partial-token ID 38155 được cả hai model dự đoán đúng, tránh suy mọi partial-token đều gây lỗi.

Nguồn: reports/bert_final_analysis/audit_summary.json; private/error_review_with_ai.csv giữ local do chứa corpus. AI review không phải adjudication độc lập của chuyên gia.

"""
    text = text.replace("# 9. BERT-small:", audit_section + "# 9. BERT-small:", 1)
    extension = f"""## 9.2. Đợt mở rộng ngày 28/10/2026

Giữ split 11.052/2.368/2.368, pretrained/revision và cách giữ trọn span. Grid bốn cấu hình: CLS/span mean × head LR 1e−4/1e−3; encoder LR 5e−5; mỗi cấu hình ba seed đã khai báo. Tối đa 20 epochs, patience 4, min_delta 0,0001; batch32, max_length128, AdamW, dropout0,1, weight_decay0,01, warmup10%, clip1. Pooler/classifier dùng head LR; embeddings/transformer dùng encoder LR. Mọi run khởi tạo từ pretrained, không dùng checkpoint học thuộc 100 mẫu. Train/validation metrics được tính ở eval mode; checkpoint chọn macro-F1 validation cao nhất, hòa giữ epoch trước.

Bảng 33. Grid BERT mở rộng, mean/std validation qua ba seed

{table(['Readout', 'Encoder LR', 'Head LR', 'Val F1 mean', 'Sample std'], [[r['readout'], r['encoder_lr'], r['head_lr'], f"{float(r['validation_macro_f1_mean']):.4f}", f"{float(r['validation_macro_f1_sample_std']):.4f}"] for r in grid])}

Chọn span mean/head1e−4 bằng mean validation 0,8183 ± 0,0011. Không chọn seed tốt nhất. Primary seed cố định 20261013, checkpoint epoch18; train F1 0,9309, validation F1 0,8173. Khoảng cách train–validation vẫn tồn tại; model cuối là best checkpoint thay vì trạng thái optimizer cuối.

![Hình 9. Đường học train/validation của primary BERT; vạch epoch18 là checkpoint chọn trước test.](../../bert_final_analysis/figures/primary_learning_curve.png)

Bảng 34. Ba seed BERT đã khóa trên cùng test

{table(['Seed', 'Epoch chọn', 'Val F1', 'Test F1', 'Test accuracy'], [[r['seed'], r['best_epoch'], f"{r['validation_macro_f1']:.4f}", f"{r['test_macro_f1']:.4f}", f"{r['accuracy']:.4f}"] for r in seeds])}

Mean test macro-F1 0,8182 ± 0,0038. Seed 20261015 đạt 0,8222 nhưng không được chọn thay primary sau khi xem test. Primary 0,8146, CI95% [0,7919;0,8350]; BERT−LR −0,0017, CI [−0,0184;0,0182]; BERT−Tree +0,0149, CI [−0,0049;0,0378]; BERT−POS +0,1040, CI [0,0785;0,1343]. Chưa đủ bằng chứng BERT hơn LR/Tree; có bằng chứng hơn POS trong phép so pointwise này.

![Hình 10. Benchmark năm mô hình; BERT dùng primary seed, cùng 1.000 group resamples.](../../bert_final_analysis/figures/benchmark_ci.png)

![Hình 11. CI paired của BERT primary trừ LR/Tree/POS; đường 0 giúp đọc sự bất định.](../../bert_final_analysis/figures/paired_differences_focus.png)

![Hình 12. F1 từng lớp: BERT cải thiện bốn lớp so với LR nhưng SVA thấp hơn.](../../bert_final_analysis/figures/per_class_f1.png)

![Hình 13. Confusion LR và BERT primary trên cùng test; hàng gold, cột prediction.](../../bert_final_analysis/figures/confusion_comparison.png)

SVA của BERT có F1 0,7392 so LR 0,8106; Nn của BERT 0,8098 so LR 0,7781. Accuracy BERT 0,8438 cao hơn LR 0,8357 không đồng nghĩa macro-F1 cũng cao hơn. Không dùng mean ba seed BERT để thay thế phép so primary BERT với một seed LR như một kiểm định tương đương.

Nguồn: reports/bert_final_analysis/overall_results.csv, seed_results.csv, bootstrap.json và verification.json. Selection BERT có trước scoring BERT; đợt mở rộng diễn ra sau khi test cổ điển đã được xem, nên đây là mở rộng trên cùng test, không phải một test độc lập hoàn toàn mới. Không thay cấu hình sau khi xem BERT test; nghiên cứu tiếp nên dùng external test mới.

"""
    text = text.replace("# 10. Kiểm chứng, tái lập và độ tin cậy", extension + "# 10. Kiểm chứng, tái lập và độ tin cậy", 1)
    text = text.replace("# 11. Thảo luận, trả lời RQ và kết luận",
        "## 10.1. Kiểm chứng đợt BERT và giới hạn workspace hiện tại\n\n61 kiểm tra selection/code/input/checkpoint/predictions của đợt BERT mới đều qua. Tái tính bootstrap dùng đúng weights 1.000×97 của đợt cổ điển và tái tạo đúng CI cổ điển; không gọi model scorer lên test. Bộ 23 tests tổng hợp kiểm tokenization/readout, train→lock→test, resume và group-bootstrap đã qua ở lượt chuẩn bị bàn giao. Đây khác với replay raw→52 fit lịch sử.\n\nLock cổ điển hiện còn 69/74 file khớp: bốn generator reporting bị xóa và CSV validation đổi padding; CSV test cũng lệch scoring hash. Chuẩn hóa CSV trong bộ nhớ tái tạo đúng hash gốc, bốn source có bản đúng hash trong zip. Metric cổ điển vẫn khớp nhưng không tuyên bố verify_lock toàn workspace hiện tại pass. Xem reports/maintenance/reports_check.md; không sửa lock lịch sử.\n\n# 11. Thảo luận, trả lời RQ và kết luận", 1)
    text = text.replace("src/reporting/build_final_project_report.py", "src/reporting/build_bert_delivery.py")
    text = text.replace("Báo cáo hiện chưa có xác nhận phân công theo tên;", "Mẫu xác nhận và bằng chứng ở reports/summaries/final/handover_checklist.md; báo cáo hiện chưa có xác nhận phân công theo tên;")
    text = text.replace("Có 17 ca chờ chuyên gia; chưa đổi gold.",
                        "Có 17 ca chờ chuyên gia từ đợt cũ và sáu ID ưu tiên từ audit BERT mới (có thể trùng); hàng đợi hợp nhất nằm trong private/expert_review_queue.csv. Chưa đổi gold.")
    counter = iter(range(1, 100))
    text = re.sub(r"^Bảng \d+\.", lambda _: f"Bảng {next(counter)}.", text, flags=re.MULTILINE)
    path = FINAL / "bao_cao_tong_ket_NUCLE.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_analysis_report():
    s = read(ANALYSIS / "summary.json")
    b = read(ANALYSIS / "bootstrap.json")
    rows = csv_read(ANALYSIS / "overall_results.csv")
    content = """# Kết quả BERT-small mở rộng và so sánh cuối

Ngày 28/10/2026. Chỉ đọc bằng chứng đã lưu; không train lại, không chạy lại inference test, không thay gold hoặc seed đại diện.

## Phương pháp và kiểm chứng

12 run: CLS/span mean × head LR 1e−4/1e−3 × ba seed, encoder LR5e−5; tối đa20 epochs/patience4. Chọn cấu hình theo mean best-validation F1; primary seed cố định20261013. 61 kiểm tra hash/metric đều qua. Dùng cùng 2.368 test IDs, gold và 97 connected groups với LR/Tree/POS. CI dùng đúng 1.000 paired group resamples của đợt cổ điển; std seed không phải CI.

## Benchmark primary

""" + table(["Model", "Val F1", "Test F1", "Accuracy", "CI95%"],
        [[r["model"], f"{float(r['validation_macro_f1']):.4f}", f"{float(r['test_macro_f1']):.4f}", f"{float(r['accuracy']):.4f}",
          f"[{float(r['ci95_low']):.4f}; {float(r['ci95_high']):.4f}]"] for r in rows])
    content += "\n\nMean test BERT qua ba seed = 0,8182 ± 0,0038; primary test = 0,8146. Không thay primary bằng seed test cao nhất.\n\n"
    content += table(["Contrast", "Δ F1", "CI95%"], [[r["contrast"], f"{r['difference']:.4f}", f"[{r['ci95'][0]:.4f}; {r['ci95'][1]:.4f}]"] for r in b["differences"]])
    content += """

CI BERT−LR và BERT−Tree chứa 0: chưa đủ bằng chứng mô hình nào tốt hơn. CI BERT−POS dương. Đây là pointwise intervals, một split; đợt BERT mở rộng sau khi test cổ điển đã được xem, không phải external test mới.

![Benchmark với CI](figures/benchmark_ci.png)

![F1 từng lớp](figures/per_class_f1.png)

## Phân tích lỗi

Hai model cùng đúng 1.860 ca; BERT đúng riêng138, LR đúng riêng119, cùng sai251. BERT primary SVA F1 0,7392 so LR0,8106; Nn 0,8098 so LR0,7781. Không quy nguyên nhân cho attention/POS chỉ từ sự khác biệt prediction.

Audit40: 12 BERT-sai/LR-đúng (ít nhất sáu gold SVA ưu tiên), 12 chiều ngược, 10 cùng sai, sáu controls. Source gold/span khớp40/40; đã có40 nhận xét AI riêng. Cột review/contribution của nhóm để trống. Nhóm cùng-sai chứa9 ArtOrDet/1 Nn do thứ tự bucket; không suy prevalence toàn corpus. Sáu ID ưu tiên chuyên gia:23037,34539,16809,30366,32801,32874; không tự đổi gold.

CSV có câu/correction giữ local: private/error_review_with_ai.csv. error_review_template.csv giữ bản trống. Ghi chú AI không phải adjudication độc lập; correction nguồn có thể cần đọc cùng annotation khác. Control partial-token dự đoán đúng giúp tránh quy mọi substring thành nguyên nhân gây lỗi.

## Bằng chứng và tái chạy kiểm tra

- verification.json: kết quả kiểm hash và metric.
- bootstrap.json, paired_differences.csv: CI/model và paired contrasts.
- overall_results.csv, per_class_metrics.csv, validation_grid.csv, seed_results.csv: bảng tổng hợp.
- audit_summary.json: phạm vi chọn và trạng thái review; corpus nằm riêng dưới private/.
- source_manifest.json: snapshot lúc tạo analysis; delivery_manifest.json: snapshot cuối sau review/report.

Chạy lại analysis vào output mới bằng `python -m src.evaluation.bert_final_analysis --output reports/bert_final_analysis_recheck`. Script từ chối ghi đè. AI notes là quyết định review được lưu, không thể tự tạo lại như một metric bằng lệnh tính toán.

Không gọi verify_lock cổ điển là pass trên workspace hiện tại: bốn source bị xóa và CSV đã đổi định dạng; chuẩn hóa tái tạo đúng hash gốc, metric/CI vẫn khớp. Manifest lịch sử giữ nguyên.
"""
    (ANALYSIS / "analysis_report.md").write_text(content, encoding="utf-8")
    (ANALYSIS / "README.md").write_text("# BERT-small: kết quả cuối ngày 28/10/2026\n\nĐọc [analysis_report.md](analysis_report.md); các bảng JSON/CSV và figures là aggregate, private/ chứa corpus/review từng mẫu và giữ local.\n\nBERT primary test F1 0,8146; mean ba seed 0,8182 ± 0,0038. CI BERT−LR chứa 0; không chọn seed theo test.\n", encoding="utf-8")


def merge_expert_queue():
    old = csv_read(ROOT / "reports/evaluation_followup/expert_review_queue.csv")
    mapping = {row["annotation_id"]: {**row, "origin": "classical_audit"} for row in old}
    notes = {r["annotation_id"]: r for r in read(ANALYSIS / "private/ai_review.json")["notes"]}
    packet = {r["annotation_id"]: r for r in read(ANALYSIS / "private/audit_packet.json")}
    for annotation_id in read(ANALYSIS / "audit_summary.json")["expert_priority_ids"]:
        if annotation_id in mapping:
            mapping[annotation_id]["origin"] += ";bert_audit"
        else:
            mapping[annotation_id] = {"annotation_id": annotation_id, "gold": packet[annotation_id]["gold"],
                "current_category": notes[annotation_id]["category"], "question": notes[annotation_id]["note"],
                "status": "pending_independent_human_review", "expert_decision": "", "origin": "bert_audit"}
    csv_write(ANALYSIS / "private/expert_review_queue.csv", list(mapping.values()))
    return len(mapping)


def write_handover(queue_count):
    content = f"""# Bàn giao, xác nhận đóng góp và khai báo AI

Ngày28/10/2026. Bộ kỹ thuật đã hoàn tất train/validation selection/test, kiểm chứng predictions và paired bootstrap; chưa có xác nhận nộp bài. Tên/MSSV và phân công dưới đây cần nhóm điền từ công việc thực tế, không suy ngược từ module hoặc nhật ký AI.

## File sử dụng

- [Báo cáo Markdown](bao_cao_tong_ket_NUCLE.md), [DOCX](bao_cao_tong_ket_NUCLE.docx), [PDF](bao_cao_tong_ket_NUCLE.pdf).
- [Slide PowerPoint](NUCLE_presentation.pptx), [slide PDF](NUCLE_presentation.pdf), [ghi chú thuyết trình](speaker_notes.md).
- [Bảng/hình và kiểm chứng BERT](../../bert_final_analysis/analysis_report.md).
- Review có source giữ local tại reports/bert_final_analysis/private/error_review_with_ai.csv; manual review/contribution vẫn trống. Hàng đợi hợp nhất có {queue_count} IDs chờ người đánh giá; không tự xác nhận bằng AI.

## Xác nhận thành viên và đóng góp thực tế

| Thành viên | Họ tên/MSSV | Phần việc thực tế | Bằng chứng code/log/review | Tự kiểm tra và xác nhận |
|---|---|---|---|---|
| 1 | CHƯA CUNG CẤP | Cần điền | Cần dẫn file/ID cụ thể | Chưa xác nhận |
| 2 | CHƯA CUNG CẤP | Cần điền | Cần dẫn file/ID cụ thể | Chưa xác nhận |
| 3 | CHƯA CUNG CẤP | Cần điền | Cần dẫn file/ID cụ thể | Chưa xác nhận |

Phân chia tham khảo đã thống nhất về nội dung: phần1 dữ liệu + features; phần2 mô hình/huấn luyện; phần3 đánh giá/phân tích. Đây không phải xác nhận ba thành viên thực sự đã làm các phần đó.

## Khai báo hỗ trợ AI để nhóm rà soát

Công cụ hỗ trợ trong phiên làm việc này: Codex (OpenAI). Phạm vi: giải thích phương pháp, tạo/sửa code pipeline và tests, tính toán từ predictions đã lưu, tạo bảng/hình, ghi chú audit có nhãn AI, cập nhật tài liệu và xuất bản bàn giao. Người dùng tự chạy train và test BERT. Không ghi nhận phiên AI là đóng góp cá nhân của thành viên hoặc adjudication độc lập.

Nhóm cần kiểm tra và xác nhận thêm tên công cụ/model khác đã sử dụng, phần nội dung giữ lại, cách từng thành viên tự kiểm tra kết quả, và chính sách môn học. Không tự xác nhận khai báo này thay cho sinh viên. Các quan sát source/correction trong audit là nhận xét hỗ trợ AI; gold giữ nguyên.

## Kiểm trước nộp

- [x] Khóa cấu hình/primary seed trước BERT test; cùng split, không refit train+validation.
- [x] Giữ metric primary riêng với mean/std ba seed và bootstrap CI.
- [x] Giữ kết quả âm BERT đợt đầu cùng kết quả mở rộng; nêu test cổ điển đã được xem trước mở rộng.
- [x] Có source/output hashes, script kiểm chứng, bảng/hình và artifacts review.
- [ ] Nhóm điền tên/MSSV, đóng góp thực tế và chấp nhận khai báo AI.
- [ ] Người chuyên môn hoàn tất các review còn pending hoặc nhóm giữ rõ giới hạn.
- [ ] Đối chiếu mẫu/rubric, số trang, font, quy định slide và hạn nộp trên LMS.
- [ ] Kiểm PDF/slide trên máy dùng trình bày và thực hiện nộp bài; chưa được thực hiện thay người dùng.

Mỗi thành viên nên tự giải thích một sample từ source→span→features/encoding→prediction→gold, cách chia group tránh leakage, vì sao dùng macro-F1, khác nhau giữa std seed và CI bootstrap, và một case model sai đã tự review. Không tune thêm bằng test đã xem.
"""
    (FINAL / "handover_checklist.md").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-only", action="store_true", help="Export current edited Markdown; slides built separately")
    parser.add_argument("--refresh-evidence", action="store_true", help="Explicitly regenerate the technical draft from evidence; replaces edited prose")
    args = parser.parse_args()
    markdown = FINAL / "bao_cao_tong_ket_NUCLE.md"
    if args.refresh_evidence or not markdown.exists():
        focused_differences()
        write_analysis_report()
        queue_count = merge_expert_queue()
        write_handover(queue_count)
        markdown = build_report()
    export_docx(markdown, FINAL / "bao_cao_tong_ket_NUCLE.docx")
    export_pdf(markdown, FINAL / "bao_cao_tong_ket_NUCLE.pdf")
    export_docx(ANALYSIS / "analysis_report.md", FINAL / "BERT_comparison_report.docx")
    export_pdf(ANALYSIS / "analysis_report.md", FINAL / "BERT_comparison_report.pdf")
    evaluation = ROOT / "reports/summaries/evaluation/evaluation_tasks_report"
    export_docx(evaluation.with_suffix(".md"), evaluation.with_suffix(".docx"))
    print("Exported the current edited reports. Existing review decisions and source evidence were left unchanged.")


def focused_differences():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [r for r in read(ANALYSIS / "bootstrap.json")["differences"] if "Majority" not in r["contrast"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    for index, row in enumerate(data):
        ax.plot(row["ci95"], [index, index], color="#087f8c", linewidth=3)
        ax.scatter(row["difference"], index, color="#c45b39", zorder=3)
        ax.annotate(f"{row['difference']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]",
                    (row["difference"], index), xytext=(0, 11), textcoords="offset points", fontsize=9, ha="center")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(yticks=range(len(data)), yticklabels=[r["contrast"] for r in data],
           xlabel="Difference in test macro-F1; paired group-bootstrap 95% CI", ylim=(-.6, len(data)-.4))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout(); fig.savefig(ANALYSIS / "figures/paired_differences_focus.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
