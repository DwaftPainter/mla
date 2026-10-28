"""Apply the documented language edit without changing tables or evidence.

The original documents are kept in reports/documentation_archive. Numerical
tables, formulas, citations, corpus examples and executable snippets are retained.
"""
from __future__ import annotations

from pathlib import Path
import re

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

# These changes make the terminology consistent in prose. Code/file names are
# protected separately; formal references and numerical tables are not edited.
PHRASES = [
    ("gold span", "đoạn lỗi được gán sẵn"),
    ("gold label", "nhãn gốc"),
    ("class weighting", "trọng số lớp"),
    ("sample standard deviation", "độ lệch chuẩn mẫu"),
    ("confidence interval", "khoảng tin cậy"),
    ("early stopping", "dừng sớm"),
    ("full training reproduction", "tái lập toàn bộ quá trình huấn luyện"),
    ("independent expert adjudication", "đánh giá độc lập của chuyên gia"),
]


def polish(text):
    if not text.strip() or text.lstrip().startswith(("Nguồn:", "[REF", "http://", "https://")):
        return text
    # Keep commands and code examples byte-for-byte. Language edits belong in
    # narrative passages; translating individual English words can break code
    # and produce unnatural Vietnamese in otherwise English sentences.
    if text.startswith(("    ", "\t")) or re.match(r"\s*(?:def |class |return |from |import |for |if |with |assert |\.venv|python |py |uv )", text):
        return text
    text = re.sub(r"\b(Ngày|ngày|phần|Phần|hình|Hình|bảng|Bảng|nhóm|Nhóm|mẫu|có|còn|qua|đúng|sai|riêng|tối đa|epoch|epochs|seed|batch|dropout|patience|update|updates)(?=\d)", r"\1 ", text)
    text = re.sub(r"\b(LR|CLS|Tree|BERT|SVA|POS)(?=\d)", r"\1 ", text)
    text = re.sub(r"±\s*(?=\d)", "± ", text)
    saved = []

    def protect(match):
        saved.append(match[0])
        return f"§{len(saved)-1}§"

    text = re.sub(r"`[^`]+`|https?://[^\s)]+|(?:[A-Za-z]:)?(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.\\/-]+|[A-Za-z0-9_.-]+\.(?:py|json|jsonl|csv|md|docx|png|pdf|pptx|txt|xml|npy|pt|zip)\b", protect, text)
    for old, new in PHRASES:
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text, flags=re.IGNORECASE)
    for index, value in enumerate(saved):
        text = text.replace(f"§{index}§", value)
    return text


def polish_markdown(path):
    original = path.read_text(encoding="utf-8-sig")
    lines = original.splitlines()
    fenced = False
    references = False
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            result.append(line)
            continue
        if re.match(r"^#+\s+Tài liệu tham khảo", stripped, re.IGNORECASE):
            references = True
        if fenced or references or stripped.startswith(("|", "![")) or re.search(r"[Σ∂∇]|^\w+\([^)]*\)\s*=", stripped):
            result.append(line)
        else:
            result.append(polish(line))
    updated = "\n".join(result) + "\n"
    if updated != original:
        path.write_text(updated, encoding="utf-8")
    return sum(a != b for a, b in zip(lines, result))


LEGACY_EDITS = {
    "contextual/contextual_model_validation_report.docx": {
        1: "Báo cáo ghi lại đợt thực nghiệm ngày 23/10/2026, chỉ sử dụng tập train và validation. Nội dung gồm rà soát dữ liệu, so sánh hai cách tiền xử lý và thử BERT-small. Đợt này chưa đánh giá test, tính bootstrap hay khóa mô hình cuối; các kết quả mở rộng về sau được trình bày trong báo cáo tổng kết.",
        3: "Trên validation, LR Full đạt macro-F1 0.8067, còn BERT-small ở seed chính đạt 0.2003; chênh lệch BERT−LR là -0.6064. BERT được thử ở đây là kiến trúc nhỏ gồm 2 layer, hidden size 128, chạy trên CPU. Khi lặp lại learning rate được chọn với ba seed, kết quả là 0.1984 ± 0.0030. Giá trị sau dấu ± là độ lệch chuẩn mẫu giữa các seed, không phải khoảng tin cậy.",
        8: "Hai phiên bản tiền xử lý được so sánh trên cùng 11.052 mẫu train và 2.368 mẫu validation, giữ nguyên ID và nhãn. Có 381 mẫu train và 96 mẫu validation thay đổi câu đầu vào hoặc vị trí lỗi trong câu. Kiểm tra cho thấy đoạn văn bản tại các offset và số token giao với span đều khớp. Không có đầu vào trùng hoàn toàn giữa train và validation, dù phép kiểm này chưa loại trừ mọi trường hợp gần trùng về nghĩa.",
        9: "Để đo độ nhạy với tiền xử lý, các mô hình được huấn luyện lại với cấu hình đã chọn, không tinh chỉnh thêm. Cả hai lượt LR đều dừng vì mức thay đổi đã nhỏ, trước giới hạn tối ưu hóa. Do dữ liệu train cũng thay đổi, dự đoán trên một mẫu validation giữ nguyên đầu vào vẫn có thể khác. Vì chênh lệch nhỏ và chưa có khoảng tin cậy, kết quả này chưa đủ để khẳng định v2 cải thiện mô hình.",
        13: "Phần phân tích định tính xem xét 18 lỗi được chọn có chủ đích: ba ID đầu ở mỗi hướng nhầm Vform/SVA và Nn/ArtOrDet, sau đó thêm sáu ca có nhiều token. Bản sửa chỉ được đọc sau khi có dự đoán và không đi vào mô hình. Các nhận xét có hỗ trợ AI, chưa được chuyên gia đánh giá độc lập; vì vậy không dùng mẫu này để ước lượng tỷ lệ nguyên nhân trong toàn corpus. Nhãn gốc được giữ nguyên.",
        14: "Một điểm dễ bỏ sót là lỗi thiếu determiner vẫn có thể được gán vào một đoạn danh từ không rỗng. Do đó, không có span rỗng không có nghĩa dữ liệu không chứa lỗi thiếu từ. Ngoài ra, một bản sửa có thể thay đổi nhiều yếu tố cùng lúc, trong khi annotation chỉ có một nhãn. Những trường hợp này cần được đọc trong ngữ cảnh trước khi quy mọi sai khác cho mô hình.",
        18: "Mỗi đầu vào BERT là câu gốc có [SPAN_START] và [SPAN_END] đánh dấu đoạn lỗi, cùng [CLS] và [SEP] ở hai đầu. Ba phần trái, span và phải được tokenize riêng. Với giới hạn 128 token, toàn bộ span được giữ trước, rồi phần dung lượng còn lại dành cho ngữ cảnh gần hai phía. Embedding của marker, classifier và encoder đều được fine-tune; nhãn, bản sửa và ID không nằm trong đầu vào dự đoán.",
        29: "Kết quả theo độ dài span dùng macro-F1 trên năm lớp. Với các cặp nhãn hoặc nhóm POS, F1 được tính lại trên hai nhãn của cặp nhưng dự đoán vẫn giữ đủ năm lớp. Các nhóm ít mẫu cần được đọc cùng support trong per_class_metrics.csv. Span rỗng không có mẫu nên ghi N/A. POS NOUN do parser gán cũng chưa đủ để xác nhận cả đoạn là một cụm danh từ.",
        34: "Ở seed chính, BERT gần như chỉ dự đoán ArtOrDet và Nn, không dự đoán Vform hoặc SVA. Vì vậy, accuracy chỉ cao hơn Majority một ít chưa thể hiện khả năng phân biệt đủ năm lớp. Dung lượng nhỏ, cách lấy pooled CLS và thời gian tối ưu hóa đều là những khả năng cần kiểm tra, chưa phải nguyên nhân đã được chứng minh. Phần chẩn đoán tiếp theo nên xem mô hình có học được trên train và checkpoint có tái tạo đúng dự đoán hay không, trước khi thử span pooling hoặc mô hình lớn hơn.",
        35: "Tại thời điểm báo cáo này, LR Full, Tree S+L+P+M và hai mô hình đối chứng POS/Majority là các ứng viên cho bước lựa chọn tiếp theo. BERT-small được giữ lại như một kết quả thử nghiệm có cấu hình, checkpoint và seed rõ ràng. Đây chưa phải quyết định khóa cuối cùng. Điểm cao hơn của một lượt huấn luyện lại trong phép kiểm độ nhạy cũng không được dùng để thay kết quả đối chứng đã lưu.",
        36: "Các kết quả ở đây chỉ phản ánh validation, vốn đã được dùng trong quá trình phát triển và lựa chọn mô hình. Đợt này chưa có test hoặc khoảng tin cậy. Mẫu 18 ca có chủ đích và nhận xét có hỗ trợ AI nên không đại diện toàn corpus. BERT còn có kiến thức từ dữ liệu pretraining ngoài NUCLE; nghiên cứu chưa kiểm tra khả năng trùng dữ liệu pretraining và cũng không so trực tiếp điểm số với các bài có thiết lập khác.",
    },
    "contextual/bert_learning_diagnostics_report.docx": {
        1: "Báo cáo ngày 23/10/2026 tập trung vào một câu hỏi: BERT-small có thực sự học được từ dữ liệu train hay không? Các phép kiểm chỉ dùng train, không bổ sung điểm validation hoặc test và không thay kết quả đã công bố. Thiết kế được ghi trước ba lượt học thuộc; phép thử hai span trong cùng câu có thêm protocol riêng trước khi chạy.",
        3: "BERT-small học được nhãn của 100 mẫu train. Cả ba cấu hình đều đạt accuracy 100% trên tập nhỏ này, và các phép kiểm chưa phát hiện đường gradient bị đứt. Với learning rate 5e-5, mô hình dự đoán đúng toàn bộ mẫu nhưng loss vẫn cao hơn ngưỡng chặt đã đặt. Learning rate 1e-3 đạt cả hai điều kiện nhanh hơn. Như vậy, điểm validation thấp trước đó không đồng nghĩa mô hình hoàn toàn không thể học bài toán.",
        4: "Kết quả này gợi ý nên thử cách tối ưu hóa khác trên cùng BERT nhỏ và so sánh CLS với span mean trong điều kiện kiểm soát. Chưa có căn cứ phải chuyển ngay sang BERT-base. Đây là hướng kiểm tra tiếp theo, chưa phải kết quả validation mới: tại mốc này, LR đạt 0,8067 còn BERT cũ đạt 0,1984 ± 0,0030.",
        7: "Tập chẩn đoán gồm 100 mẫu, được chọn bằng seed 20261023 với 20 mẫu cho mỗi nhãn, trước khi xem dự đoán. Các lượt dùng cùng đầu vào, nhãn, thứ tự minibatch và trọng số khởi tạo. Cách chọn cân bằng này giúp kiểm tra khả năng học thuộc, nhưng dễ hơn bài toán khái quát hóa: mô hình có thể nhớ từng câu mà chưa học được quy luật ngữ pháp hoặc vị trí lỗi.",
        10: "Sau một bước AdamW, embedding của hai marker, trọng số query trong encoder và classifier đều thay đổi. Điều này cho thấy các thành phần được kiểm đang nhận cập nhật, thay vì bị đóng băng ngoài ý muốn. Tuy nhiên, nó chưa bảo đảm toàn bộ cấu hình huấn luyện đã tối ưu hoặc không còn lỗi ở nơi khác.",
        12: "Ở cách đọc CLS, classifier nhận biểu diễn CLS của layer cuối qua dense và tanh; nó không đọc trực tiếp các hidden state tại span ở layer cuối. Vì vậy, gradient trực tiếp ở các vị trí span của layer này bằng 0 là điều phù hợp với kiến trúc. Gradient vẫn đi qua self-attention tới các layer trước và embedding, nên không thể dựa vào giá trị đó để kết luận encoder không học từ span.",
        15: "Để kiểm tra phản ứng với vị trí span, phép thử giữ thứ tự từ trong 100 câu train và chuyển hai marker sang một vùng không giao span cũ, có cùng số token. Checkpoint CLS cũ được dùng cho phép thử này. Vì cả quan hệ vị trí marker và token đều thay đổi, đây là phép kiểm độ nhạy, không phải một tập câu mới có nhãn đúng để chấm accuracy.",
        16: "CLS thay đổi dự đoán ở 28/100 mẫu khi dời marker. Mô hình có phản ứng với vị trí đánh dấu, nhưng phản ứng đó chưa chứng minh nó xác định đúng loại lỗi. Kết quả span mean trong bảng chỉ đổi cách đọc encoder cũ, chưa huấn luyện lại classifier cho cách đọc mới, nên không dùng để xếp hạng hai mô hình. Việc thêm padding chỉ làm logits thay đổi tối đa 3.58e-07, phù hợp với sai số số học nhỏ.",
        18: "Từ 1594 câu train đủ điều kiện, phép thử chọn 50 câu có hai span khác vị trí và khác nhãn. Checkpoint CLS cũ cho dự đoán khác nhau ở 7/50 cặp, nhưng chỉ đúng cả hai mục tiêu ở 2/50 cặp. Kết quả cho thấy phản ứng với marker chưa đồng nghĩa phân biệt tốt hai lỗi trong cùng câu. Đây là mẫu có chủ đích trên train, không phải accuracy trên validation.",
        24: "Trong thí nghiệm 100 mẫu, hai lượt CLS chỉ khác learning rate; so sánh span mean với CLS ở 1e-3 cũng giữ cùng pooler và classifier. Tuy nhiên, so với lần huấn luyện toàn bộ train trước đó, thí nghiệm nhỏ còn khác số lần lặp mỗi mẫu, dropout, weight decay, lịch learning rate và phân bố lớp. Vì thế, chưa thể quy toàn bộ thất bại cũ cho riêng learning rate.",
    },
    "classical/modeling_ablation_results_report.docx": {
        1: "Đặc trưng, huấn luyện và phân tích kết quả",
        0: "Báo cáo trình bày kết quả validation tại mốc 20/10/2026. Các bổ sung ngày 23/10/2026 và kết quả BERT về sau được đọc cùng README của nhóm báo cáo; trạng thái hiện tại nằm trong docs/status.md. Những câu nói test chưa được đánh giá trong tài liệu này thuộc thời điểm thực nghiệm ban đầu.",
        4: "Báo cáo tổng hợp mã nguồn và các kết quả thực nghiệm đã lưu, không thực hiện lại huấn luyện hoặc đánh giá test. Các điểm số bên dưới là validation, trừ những chỗ ghi rõ train. Thông tin test_scored=false phản ánh phạm vi của các tệp được đối chiếu. Các phần giải thích nguyên nhân được trình bày như giả thuyết khi chưa có phép kiểm trực tiếp.",
        8: "Sau tiền xử lý, dữ liệu gồm 15.788 mẫu thuộc năm nhãn, chia theo nhóm bài viết thành 11.052 mẫu train, 2.368 validation và 2.368 test. Encoder chỉ học từ train; validation dùng để chọn cấu hình. Trên cùng cách chia này, báo cáo so sánh Majority, POS rule, Logistic Regression và Decision Tree, rồi xem xét sáu biến thể đặc trưng bằng cùng lưới tham số cho từng loại mô hình.",
        9: "LR Full đạt macro-F1 0,8067, cao nhất trong các biến thể LR đã thử. Với Tree, kết quả Full là 0,7879, thấp hơn một chút so với S+L+P+M đạt 0,7929. Hai đối chứng POS rule và Majority lần lượt đạt 0,7048 và 0,1184. Việc thêm POS tạo mức tăng lớn nhất trong chuỗi thí nghiệm. Ngược lại, bỏ toàn bộ thông tin từ vựng chỉ làm giảm nhẹ điểm chung, dù ảnh hưởng ở từng lớp khác nhau.",
        10: "Các kết quả này giúp lựa chọn mô hình trên validation. Ở thời điểm báo cáo, chưa có khoảng tin cậy để đánh giá các chênh lệch nhỏ, nên chưa coi thứ hạng validation là kết luận cuối cùng trên test.",
        12: "Bộ trích xuất nhận câu gốc và vị trí ký tự của đoạn lỗi đã được gán. spaCy 3.8.7 cùng en_core_web_sm 3.8.0 cung cấp từ loại, hình thái và quan hệ phụ thuộc; thành phần NER được tắt. Trước khi lấy đặc trưng, mã kiểm tra câu và đoạn văn bản tại các offset có khớp nguồn hay không, rồi chọn những token giao với span. Vị trí lỗi được cho trước, không do mô hình tự tìm.",
        19: "TrainOnlyEncoder xây dựng từ vựng và các giá trị phân loại từ tập train. Đặc trưng số giữ nguyên giá trị; đặc trưng phân loại được mã hóa one-hot, còn sự xuất hiện unigram/bigram dùng multi-hot. Các trường hợp chưa gặp, thiếu dữ liệu và không áp dụng có marker riêng. Mã hiện tại không chuẩn hóa z-score cho đặc trưng số.",
        25: "Majority luôn dự đoán ArtOrDet vì đây là lớp đông nhất trong train. POS rule dùng hai ánh xạ cố định ADP → Prep và DET → ArtOrDet. Với các POS signature khác, nó tra nhãn phổ biến nhất đã học từ train; trường hợp chưa gặp hoặc thiếu thông tin quay về Majority. Đối chứng này chỉ dùng POS của span, không dùng POS ngữ cảnh.",
        40: "LR Full cao hơn Tree Full 0,0188 trên validation. Một cách giải thích có thể là LR kết hợp được nhiều tín hiệu thưa, trong khi Tree chịu giới hạn về độ sâu và kích thước lá. Tuy nhiên, bảng điểm chưa kiểm chứng cơ chế này; kết quả chỉ áp dụng cho dữ liệu và lưới tham số đã thử.",
        47: "Thêm nhiều nhóm đặc trưng không phải lúc nào cũng cho điểm cao nhất. LR chọn Full, còn Tree đạt 0,7929 với S+L+P+M, cao hơn Full khoảng 0,0051. Hai biến thể Tree cũng khác depth và class_weight, nên chênh lệch này chưa đủ để kết luận thông tin dependency làm mô hình kém đi.",
    },
    "data/data_preprocessing_report.docx": {
        1: "Báo cáo chuẩn bị dữ liệu",
        0: "Báo cáo ghi lại quy trình dữ liệu tại mốc 20/10/2026. Các kiểm tra bổ sung ngày 23/10/2026 và kết quả mô hình về sau được đọc cùng README của nhóm báo cáo; docs/status.md ghi trạng thái hiện tại. Những giới hạn thực nghiệm trong tài liệu này thuộc thời điểm báo cáo ban đầu.",
        4: "Báo cáo mô tả cách dữ liệu được chuẩn bị trong mã nguồn hiện có. Các con số được đối chiếu từ summary, manifest và báo cáo so sánh đã lưu; đây không phải một lần chạy lại toàn bộ quy trình tại ngày viết báo cáo.",
        8: "Tiền xử lý chuyển dữ liệu SGML thành các mẫu dùng để phân loại loại lỗi tại vị trí đã biết. Mỗi annotation thuộc phạm vi nghiên cứu tạo một mẫu gồm câu gốc, vị trí bắt đầu và kết thúc lỗi, đoạn văn bản tương ứng và nhãn. Một câu có nhiều annotation có thể tạo nhiều mẫu, nên cần giữ mối liên hệ với bài viết nguồn khi chia tập.",
        9: "Năm nhãn được giữ nguyên từ trường TYPE của corpus: ArtOrDet, Prep, Nn, Vform và SVA. Bản sửa và ghi chú được lưu riêng, không đưa vào mô hình. Cách tổ chức này vừa giữ được nguồn gốc dữ liệu để kiểm tra, vừa tránh để thông tin của đáp án xuất hiện trong đầu vào dự đoán.",
        12: "Nguồn dữ liệu có 1.397 bài viết và 44.912 annotation. Trước khi tạo các tệp đầu ra, bước preparation kiểm tra SHA-256 của nguồn, đối chiếu CSV với SGML và kiểm tra các đầu vào audit/review. Manifest lưu lại những thông tin cần để xác định dữ liệu và mã đã dùng trong mỗi lượt chạy.",
        15: "Offset được tính theo vị trí ký tự Python trong đoạn văn mà parser trả về, với khoảng nửa mở [start, end): lấy ký tự đầu và không lấy ký tự cuối. Nó không phải vị trí byte trong file SGML. Vì parser có xử lý xuống dòng theo quy tắc cố định, các lần đối chiếu phải dùng cùng cách đọc nguồn.",
    },
}


def edit_legacy_docx():
    counts = {}
    for name, replacements in LEGACY_EDITS.items():
        path = REPORTS / "summaries" / name
        original_path = REPORTS / "documentation_archive/before_language_edit_20261028/summaries" / name
        doc = Document(original_path if original_path.exists() else path)
        changed = 0
        references = False
        for index, paragraph in enumerate(doc.paragraphs):
            before = paragraph.text
            if "tài liệu tham khảo" in before.casefold() and paragraph.style.name.startswith("Heading"):
                references = True
            if paragraph.style.name.startswith("Heading") and "phụ lục" in before.casefold():
                references = False
            code_font = any(run.font.name in ("Consolas", "Courier New") for run in paragraph.runs)
            if not before.strip() or references or code_font or paragraph._p.xpath(".//w:drawing") or paragraph._p.xpath(".//m:oMath"):
                continue
            if index in replacements:
                after = replacements[index]
            elif re.search(r"[Σ∂∇]|^\w+\([^)]*\)\s*=|^B\.\d+", before) or paragraph.style.name.lower().startswith("code"):
                continue
            else:
                after = polish(before)
            if after != before:
                paragraph.text = after
                changed += 1
        doc.save(path)
        counts[name] = changed
    return counts


def main():
    counts = {}
    for path in REPORTS.rglob("*.md"):
        if any(part in {"documentation_archive", "before_layout", "private"} for part in path.parts):
            continue
        counts[path.relative_to(ROOT).as_posix()] = polish_markdown(path)
    counts.update(edit_legacy_docx())
    from src.evaluation.bert_final_analysis import dump
    dump(ROOT / ".cache/report_editorial/automatic_edits.json", counts)
    print("Applied terminology/spacing edit to", sum(v > 0 for v in counts.values()), "documents.")


if __name__ == "__main__":
    main()
