# Tự huấn luyện và đánh giá BERT-small

Pipeline: `src/training/bert_small_experiment.py`. Người dùng đã chạy xong đợt này ngày28/10/2026: span mean/head1e−4 được chọn; primary test F1 0,8146, mean ba seed0,8182 ±0,0038. Đọc [kết quả đã lưu](../reports/bert_final_analysis/analysis_report.md). Các lệnh dưới đây là hướng dẫn tái chạy trong output mới, không phải yêu cầu train/test lại hoặc chọn lại theo test đã xem. Không sửa scorer/kết quả LR/Tree.

## 1. Mục tiêu và dữ liệu

Phân loại một span lỗi đã biết thành ArtOrDet, Prep, Nn, Vform hoặc SVA. BERT nhận câu gốc có `[SPAN_START]`/`[SPAN_END]`; không dùng correction, nhãn hoặc IDs làm predictors. S/L/P/M/D vẫn là đầu vào riêng của LR/Tree, không nối vào BERT trong thiết kế này.

| Tập | Số mẫu mặc định | Cách sử dụng |
|---|---:|---|
| Train | 11.052 | Học trọng số; tính metric train sau mỗi epoch |
| Validation | 2.368 | Chọn checkpoint và cấu hình |
| Test | 2.368 | Chỉ tạo dataloader sau khi có selection lock |

Giữ preparation v2 và grouped split hiện hành. Các bước check/train đọc file JSONL chung để lọc partition nhưng không tokenize, giữ text/label test trong dataset hoặc chấm test. File chung được hash để kiểm tra toàn vẹn. Không thay đổi split để tìm điểm cao hơn. Bài kiểm tra 100 mẫu cũ chỉ là chẩn đoán, không phải tập train của pipeline này. Mọi run mới khởi tạo từ pretrained gốc, không tiếp tục checkpoint đã học thuộc 100 mẫu.

## 2. Chuẩn bị môi trường và input local

Môi trường project hiện có có thể dùng ngay nếu đã cài dependencies. Với môi trường mới:

```powershell
uv sync --locked --python 3.13.7
uv pip install --python .venv/Scripts/python.exe -r requirements-bert.txt
```

Input cần có trên máy:

- `data_set/processed/nucle_preparation_v2/samples.jsonl`.
- `data_set/split/nucle_group_split/split_manifest.csv`.
- `reports/contextual_model_validation/model_source.json` và thư mục pretrained mà trường `path` trỏ tới.

Checkpoint mặc định là `google/bert_uncased_L-2_H-128_A-2`, revision đã ghi trong model_source: 2 layers, hidden size 128, 2 heads. Script chỉ load file local, không tự tải model hoặc corpus. Nếu chuyển project sang máy khác, copy cả input/model local và sửa đường dẫn trong bản config/model_source riêng cho máy đó. Clone Git không chứa đủ corpus/checkpoint.

## 3. Chọn cấu hình trước khi chạy

Config sẵn có: [configs/bert_small_training.json](../configs/bert_small_training.json).

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `readouts` | `cls`, `span_mean` | Pooled CLS hoặc trung bình token span sau encoder |
| `encoder_learning_rates` | `[0.00005]` | LR embeddings/transformer |
| `head_learning_rates` | `[0.0001, 0.001]` | LR pooler và classifier |
| `seeds` | 20261013, 20261014, 20261015 | Mọi cấu hình chạy cùng ba seed |
| `max_epochs` | 20 | Trần epochs cho mỗi run |
| `patience` | 4 | Dừng sau 4 epoch không cải thiện đáng kể |
| `min_delta` | 0.0001 | Mức tăng macro-F1 để reset patience |
| `batch_size` | 32 | Số mẫu/batch |
| `max_length` | 128 | Giữ nguyên toàn span; crop context nếu cần |
| `class_weight` | `none` | Có thể dùng `balanced`, trọng số chỉ tính từ train |
| `device` | `cpu` | Có thể chọn `cuda` hoặc `auto` nếu PyTorch hỗ trợ |
| `threads` | 4 | Số CPU threads |

Grid mặc định có **4 cấu hình × 3 seed = 12 run**, tối đa **240 epochs tổng cộng**, có early stopping. Đây là không gian thử mới, không bảo đảm tốt hơn kết quả cũ. Cân nhắc thời gian CPU trước khi bắt đầu. Không cần GPU để chạy; `auto` không tự cài bản PyTorch có CUDA.

Trong `span_mean`, token span vẫn có ngữ cảnh nhờ encoder BERT; chỉ bước lấy vector đầu ra tập trung vào span. Với span rỗng, dùng biểu diễn marker bắt đầu. Toàn bộ encoder được fine-tune; pooler/classifier dùng LR head. Các tham số bias/LayerNorm không chịu weight decay; các tham số còn lại dùng weight_decay cấu hình. AdamW, gradient clipping và linear warmup/decay được dùng trong từng run.

Nếu muốn thử nhanh một cấu hình trước:

```powershell
Copy-Item configs/bert_small_training.json configs/bert_small_pilot.json
```

Sửa bản pilot thành `readouts: ["span_mean"]`, `head_learning_rates: [0.001]`, `seeds: [20261013]`, `max_epochs: 3`, `patience: 2`. Dùng thư mục output pilot riêng. Không thay đổi cấu hình giữa một đợt đang chạy; scheduler phụ thuộc max_epochs. Khi chạy đợt đầy đủ, khởi tạo lại pretrained trong output mới. Pilot một seed không cung cấp độ lệch chuẩn qua seed; một epoch không cho biết chắc khả năng hội tụ.

## 4. Kiểm tra đầu vào trước khi train

```powershell
.venv/Scripts/python.exe -m src.training.bert_small_experiment check --config configs/bert_small_training.json
```

Lệnh này kiểm tra số mẫu/nhãn, IDs/split, offsets và marker; tokenize train/validation; chạy forward hai mẫu cho mỗi readout; báo device, số tham số, số mẫu bị crop và số run dự kiến. Không cập nhật trọng số, không chấm test. Group/essay không được qua nhiều split nếu metadata tương ứng có trong split manifest. Pipeline dùng split đã khóa của project; không dựng lại grouping.

Nếu check lỗi, sửa đường dẫn/cấu hình/input trước khi train. Nếu span không vừa max_length, script dừng thay vì âm thầm cắt span. Giữ quy tắc input giống nhau cho mọi seed và test.

## 5. Train và theo dõi train/validation

```powershell
.venv/Scripts/python.exe -m src.training.bert_small_experiment train --config configs/bert_small_training.json --output reports/bert_small_experiment
```

Script in tiến độ mỗi epoch, ví dụ định dạng `train_F1=... val_F1=... best=... patience=...`. Không có ngưỡng tự động bắt buộc train accuracy đạt 100%. Sau mỗi epoch:

1. Đánh giá toàn bộ train và validation bằng `model.eval()`/không gradient.
2. Lưu metric, recall từng nhãn và số dự đoán từng nhãn.
3. Lưu checkpoint khi validation macro-F1 tăng; hòa giữ epoch trước.
4. Cập nhật patience bằng min_delta; dừng ở patience hoặc max_epochs.

Checkpoint tốt nhất dùng **mọi mức tăng thực sự** của validation macro-F1, còn min_delta chỉ điều khiển patience. Lệnh không chấm test và chưa tự khóa lựa chọn. Cuối mỗi run, script nạp checkpoint vừa lưu và xác nhận dự đoán validation khớp từng sample.

Artifacts trong mỗi thư mục run:

| Artifact | Dùng để làm gì |
|---|---|
| `history.csv` | Mở Excel: train/validation loss, accuracy, macro-F1, recall và prediction count từng lớp |
| `history.json` | Metric đầy đủ từng epoch, gồm confusion matrix |
| `learning_curves.png` | Ba biểu đồ train/validation: loss, accuracy, macro-F1 |
| `best/` | Model, tokenizer và readout được chọn theo validation |
| `best_metrics.json` | Metric train/validation tại checkpoint tốt nhất |
| `validation_predictions.csv` | ID, gold, prediction và xác suất năm nhãn |
| `last.pt` | Model/optimizer/scheduler/RNG để resume ở ranh giới epoch |
| `summary.json` | Epoch tốt nhất, lý do dừng, hashes và kết quả checkpoint replay |

`loss` trong history là cross-entropy **không class weight**, tính lại trong eval để so train/validation. `mean_batch_training_objective` là trung bình loss batch dùng để backprop, có thể có class weight và dropout; không đồng nhất hai giá trị. Early stopping dựa vào validation macro-F1, không dựa vào loss train.

Nếu bị gián đoạn:

```powershell
.venv/Scripts/python.exe -m src.training.bert_small_experiment train --config configs/bert_small_training.json --output reports/bert_small_experiment --resume
```

Resume bỏ qua run hoàn tất và nạp trạng thái ở cuối epoch gần nhất của run dang dở. Phần epoch chưa hoàn tất được chạy lại. Config, data, source, code và môi trường phải giống manifest; nếu bạn muốn đổi chúng, dùng thư mục output mới. Không xóa checkpoint kết quả cũ để chạy đè. Khi một run đã xong chỉ có `last.pt` dùng để tiếp tục trạng thái optimizer, còn `best/` dùng cho đánh giá cuối.

## 6. Xác định model có học và có khái quát hóa không

| Quan sát | Cách diễn giải |
|---|---|
| Train loss giảm; train F1/recall nhiều lớp tăng | Có bằng chứng model học trên train |
| Train và validation đều thấp | Có thể underfit/tối ưu hóa chưa phù hợp; kiểm tra cấu hình và input |
| Train cao nhưng validation thấp hoặc giảm | Khoảng cách khái quát hóa/overfitting; không giải quyết bằng train vô hạn |
| Đoán gần như toàn ArtOrDet/Nn | Xem prediction counts và recall Prep/Vform/SVA; accuracy chung có thể che vấn đề |
| Validation đạt đỉnh trước epoch cuối | Dùng `best/`, không dùng `last.pt` làm model cuối |

Không có quy tắc train accuracy phải bằng 100% trên 11.052 mẫu. Không cam kết BERT phải vượt LR. Có thể điều chỉnh thiết kế sau khi xem validation, nhưng phải ghi nhận các đợt đã thử, dùng config/output mới và hoàn tất điều chỉnh trước test. Nếu thử class_weight=balanced, xác định rõ đây là một variant mới; không lựa chọn sau khi nhìn test.

## 7. Khóa lựa chọn bằng validation

Sau khi toàn grid hoàn tất và kiểm tra learning curves:

```powershell
.venv/Scripts/python.exe -m src.training.bert_small_experiment select --experiment reports/bert_small_experiment
```

Script bắt buộc đủ mọi run/seed khai báo, lấy **mean validation macro-F1 của checkpoint tốt nhất qua các seed** để chọn cấu hình. Hòa chọn cấu hình xuất hiện trước trong config. Seed đại diện cố định là seed đầu, không chọn seed có điểm cao nhất. Nếu config chỉ có một seed, sample_std là null, không phải 0.

Sinh `selection.json` và `selection.sha256`, ghi danh sách checkpoint/hashes. Sau lock, lệnh train/resume từ chối thay đổi đợt này. Nếu tiếp tục phát triển trước test, tạo đợt mới; không sửa tay selection lock.

## 8. Chạy test sau khi chốt

```powershell
.venv/Scripts/python.exe -m src.training.bert_small_experiment test --selection reports/bert_small_experiment/selection.json --output reports/bert_small_experiment/test
```

Chỉ lúc này mới tokenize/tạo dataloader test. Script kiểm selection hash, code/data và checkpoint hashes; dùng toàn bộ seed của cấu hình được chọn; không backprop, không refit train+validation, không chọn lại checkpoint theo test. Output có:

- `summary.json`: test macro-F1/accuracy từng seed và mean/sample std.
- `*_test_metrics.json`: precision/recall/F1 từng nhãn và confusion matrix từng seed.
- `*_test_predictions.csv`: predictions/xác suất theo ID để audit và paired comparison.
- `per_class_metrics.csv`: bảng từng lớp mở trong Excel.
- `test_started.json`: ghi nhận thời điểm bước test đã được khởi chạy theo lock và audit input.

Lệnh từ chối ghi đè thư mục test đã tồn tại. Nếu mất điện/lỗi hệ thống làm test dở dang, giữ output dở làm bằng chứng và chạy lại **cùng selection/checkpoint** vào output mới; không điều chỉnh thiết kế bằng các điểm test đã xem. Lệnh không thể ngăn ai đó tự tạo một thư mục khác, nên quy tắc không tune theo test vẫn là trách nhiệm nghiên cứu.

Để so với LR/Tree, dùng cùng split và cùng định nghĩa macro-F1. Primary BERT run là seed đầu đã khóa. Mean/std qua seed đo biến thiên huấn luyện; **không phải bootstrap CI**. Pipeline này xuất predictions sẵn cho phân tích paired group-bootstrap nhưng chưa tự tính CI hoặc ghi đè bảng so sánh cuối của dự án. Nếu bổ sung BERT vào bảng kết quả cuối, làm bước bootstrap/comparison riêng sau khi có test outputs, giữ nguyên phương pháp group-bootstrap đã dùng cho LR/Tree.

## 9. Kiểm thử code

```powershell
.venv/Scripts/python.exe -m pytest tests/test_bert_small_experiment.py tests/test_bert_span_classifier.py tests/test_bert_readout.py -q -p no:cacheprovider
```

Tests dùng BERT/vocabulary tổng hợp rất nhỏ; kiểm lifecycle train→lock→test, resume khớp trạng thái không gián đoạn, cách chọn theo mean seed, chặn checkpoint bị sửa và việc không tokenize test khi train. Không tải mạng và không huấn luyện corpus nghiên cứu.

Các kết quả BERT cũ và quyết định dừng trong snapshot trước đây vẫn phản ánh đúng giai đoạn đó. Đợt mới chỉ có kết quả sau khi bạn chạy các lệnh trên; không sửa protocol lịch sử để coi đây là thiết kế đã đăng ký trước từ đợt cũ.
