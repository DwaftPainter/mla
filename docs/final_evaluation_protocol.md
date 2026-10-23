# Protocol bổ sung và đánh giá cuối NUCLE

Ghi trước khi chạy bổ sung/chấm test, ngày 23/10/2026. Thực hiện theo yêu cầu tuần 2; BERT dừng thử thêm. Bản này thay phạm vi chỉ LR/POS trong đề xuất cũ, bổ sung quy tắc thực thi cho experiment_protocol.md. Kết quả chưa tồn tại tại thời điểm ghi protocol.

## 1. Ablation và tái tạo checkpoint trước test

- Cùng preparation v2, split seed 20261013, năm nhãn theo thứ tự ArtOrDet, Prep, Nn, Vform, SVA. Dùng nguyên train 11052, validation 2368, không refit train+validation.
- Tái sử dụng sáu variant đã chọn ở reports/nucle_feature_ablation_v1. Artifact cũ chỉ có summary/predictions, không checkpoint. Refit cấu hình được chọn, kiểm prediction validation với artifact gốc; không chọn lại hyperparameter theo lần refit. Ghi mọi bất đồng và dừng trước test nếu không tái tạo được.
- Full−D đồng nhất S+L+P+M: tái sử dụng feature columns, checkpoint và predictions; báo là alias, không gọi lượt fit độc lập. Full−P và Full−M: 8 cấu hình LR + 12 Tree mỗi variant, tổng 40 fit mới. Chín tên variant nhưng chỉ tám bộ feature riêng biệt, 18 tên pipeline LR/Tree nhưng 16 checkpoint riêng biệt.
- LR C=[0.01,0.1,1,10], balanced=[false,true]; Tree depth=[4,8,12], min_leaf=[5,20], balanced=[false,true]. Max macro-F1 validation, chỉ LR convergence=true theo code hiện hành; hòa giữ cấu hình đầu trong thứ tự grid. Không có eligible thì dừng báo lỗi trước test.
- LR dùng code LBFGS hiện hành (max_iter1000/max_eval1250, tolerance_grad1e-6/change1e-9), torch CPU10 threads để khớp baseline đã chẩn đoán; seed20261013. Tree deterministic. Lưu thời gian và termination, không gọi change_tolerance là gradient-converged.
- Encoder Full được khôi phục từ artifact train-only và xác minh transform train/validation. Masked encoder fit riêng chỉ train, loại lexical span trước fit, giữ parse gốc/context. Kiểm masked feature names và predictions khớp lượt cũ. Không sửa extractor/model lịch sử.
- Bỏ P/M/D chỉ bỏ nhóm feature có tên tương ứng; D vẫn chứa head/subject POS/morph, nên không gọi Full−P là loại hết thông tin từ loại, hay Full−M là loại hết thông tin hình thái.

## 2. Khóa và chấm test

- Trước mở test: lưu manifest hash source/split/preparation/encoder/model/config/code/protocol, checkpoint và validation replay. Hash manifest lock riêng, kiểm lại trước scoring. Chỉ khóa sau khi mọi replay đúng và mọi LR chọn đủ điều kiện.
- 20 tên kết quả: Majority, POS, 9 variants × LR/Tree; Full−D là alias nên có 18 predictor riêng biệt. Benchmark RQ1 chính: LR Full, Tree Full, Majority; POS bổ sung. Không thay bằng variant thắng trên test.
- Dùng test partition 2368 từ manifest, không thay labels hoặc regroup. Test Full features đã được transform bởi encoder train-only trước đây; có thể tái sử dụng sau kiểm hashes. Masked test chỉ transform bằng encoder đã khóa. Không fit bất cứ tham số nào trên test.
- Một đợt prediction test, lưu predictions để tính metric/bootstrap/phân tích sau đó. Reload/replay chỉ kiểm định tính tái lập, không lựa chọn lại model.
- Metric toàn tập: accuracy, macro precision/recall/F1, từng lớp P/R/F1/support, confusion true rows/pred columns; thứ tự năm nhãn cố định, zero_division=0.
- Span bins theo số span_token_indices preparation: empty/single_token/multi_token. Macro-F1 năm lớp cho span bins. Rỗng support0 → N/A.
- Cặp gold Vform/SVA và Nn/ArtOrDet: giữ prediction năm lớp, tính macro-F1 trên hai nhãn gold trong cặp; ngoài cặp vẫn gây FN. Báo thêm outside-pair predictions. Tập POS định trước: mỗi cặp giao primary_upos VERB/AUX/NOUN/DET, không tự gọi đây là noun phrase đã xác nhận. Báo support và so LR Full/POS/Tree Full/Majority.

## 3. Bootstrap và phân tích lỗi

- 1000 bootstrap, seed20261023, lấy có hoàn lại đúng G connected group IDs từ toàn test; giữ mọi annotation trong group, kể cả nhóm nhiều essay. Dùng cùng trọng số group cho mọi model và subset.
- CI percentile 2.5/97.5% cho macro-F1 và paired differences: LR Full−Majority, Tree Full−Majority, LR Full−Tree Full, LR Full−POS; mỗi họ LR/Tree: ba bước incremental và Full−masked/nonlexical/Full−P/Full−M/Full−D.
- Trên subsets khó báo LR Full−POS CI nếu có ít nhất 20 sample, 10 groups và cả hai nhãn gold; báo ngưỡng/support, không bỏ subset khỏi bảng metric vì CI thiếu điều kiện.
- Mọi bootstrap nonempty giữ label set cố định, thiếu lớp vẫn tính F1=0 khi denominator0; ghi số lượt thiếu lớp. Nếu subset empty trong một replicate thì bỏ replicate đó, ghi số lần và không báo CI nếu <950 replicate hợp lệ. Không bootstrap annotation độc lập.
- Nhiều CI là pointwise, không kiểm định đồng thời đã hiệu chỉnh, không khẳng định nhân quả từ ablation hoặc hệ số.
- Audit định trước tối đa 30 lỗi + 10 đúng (trong giới hạn kế hoạch 50/10), round-robin theo gold và predicted label của LR, thêm lỗi Tree khi LR đúng, ưu tiên phủ multi-token/cặp khó/bất đồng POS. Chọn bằng seed20261023, lưu IDs và membership trước đọc correction. Audit có hỗ trợ AI, không phải adjudication độc lập; không ước lượng tỷ lệ nguyên nhân trên corpus từ mẫu chủ đích. Text/correction chỉ ở private packet local.

## 4. Báo cáo và tái lập

Report DOCX mới theo chức năng, có câu trả lời riêng RQ1–RQ3, bảng validation/test/CI, công thức, file code, 40 fit bổ sung, 12 refit cũ, alias Full−D, convergence, parser/annotation limits và BERT âm. Lưu CSV/JSON/plots và sáu slide. Kiểm metric từ predictions độc lập, model reload, môi trường sạch nếu dependencies khả dụng; gói code/protocol/hash/hướng dẫn, không corpus/checkpoint. Phân biệt replay inference với tái lập toàn bộ tuning. Không push/upload tự động.
