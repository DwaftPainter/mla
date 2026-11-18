create a folder have : 
README.md — List group members and contributions. Give setup instructions, the run order, and the script or notebook that reproduces a main result.
Report/Final_Report.pdf — Explain question, data and labels, baseline, model choices, evaluation, results, error analysis, and limitations.
Code/ and Results/ — Include scripts or notebooks, dependencies, configurations, split or seed information, and saved outputs needed to check the report. Use relative paths.
Data/ — Include the data have permission to redistribute. In DATA_README.md, document the source, version, licence, and preparation steps. For restricted data, document how authorised users can obtain it and include an approved sample if available. Remove credentials and any private data that are not authorised to share.
AI_Log/AI_Use_Log.md — Record where AI was used and what personally checked or corrected. If a language model is part of the pipeline, include the model version, settings, prompts, and relevant saved outputs. 
Video/ will be used to store video 
Write the report, README, and AI log in English. Use the same final project version in the report, code, results, and videos.


create script for Individual video requirements in english:

1. Dữ liệu và đặc trưng - Quân
Đọc NUCLE/SGML; kiểm tra annotation, offsets; preparation v1/v2; chia train–validation–test; chạy spaCy; trích xuất S/L/P/M/D; fit encoder trên train và chuyển features thành vector
output: Dữ liệu đã chia, vector đặc trưng, encoder và mô tả features
2. Xây dựng và huấn luyện mô hình - Duy
Majority, POS rule, LR, Tree; tuning trên validation; huấn luyện các cấu hình ablation/masking; thử nghiệm và chẩn đoán BERT-small
output: Checkpoint, cấu hình được chọn, dự đoán và kết quả validation
3. Đánh giá và phân tích kết quả - Công
Đánh giá test; metrics, confusion matrix, bootstrap CI; so sánh mô hình và ablation; audit 18 ca validation/40 ca test; kiểm chứng tái lập; tổng hợp kết luận và giới hạn
output: Bảng/hình kết quả, phân tích lỗi và báo cáo



Briefly introduce the problem and identify contribution. 
Show a real example from the final project and explain how it moves from data to a prediction or result. 
Use a chart, table, or annotated example to explain what the data shows.
Demonstrate the project using its interface, notebook, or saved outputs, and interpret one result, error, or limitation.
scripts must include technical explanation, evidence and demonstration, personal contribution, and production quality.

