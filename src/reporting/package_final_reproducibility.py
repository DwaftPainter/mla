"""Package code/protocol/aggregate evidence and verify a local restore; no corpus."""
from pathlib import Path
import sys,json,zipfile,importlib.metadata as metadata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.training.nucle_final_selection import OUT,dump,sha
from src.evaluation.nucle_final_evaluation import verify_lock

README='''# Tái lập các task đánh giá NUCLE

Gói này gồm source, protocol, configs/lock metadata, bảng tổng hợp và báo cáo. Không chứa corpus, sentences/correction, encoder vocabulary, checkpoint hay private audit packet. Không phải bộ dữ liệu phân phối lại. Không tự upload.

## Bằng chứng đã kiểm

- Full/Mask train-only,12 refit cũ khớp mọi validation prediction.
- Lock có trước test;20 tên pipeline gồm aliases,18 predictors riêng.
- Reload checkpoint từ encoded test trong môi trường hiện tại và fresh venv khớp;20 metric và75 expanded-group checks khớp.
- Restore zip vào .cache/evaluation_tasks_bundle_restore và đối chiếu SHA256 từng file.
- Chưa chạy lại toàn grid hoặc spaCy từ raw trong fresh venv. Không gọi checkpoint replay là full training reproduction.

## Input cần có cho tái lập đầy đủ

NUCLE được cấp quyền đúng source hash; CSV Understanding, audit/review logs preparation; preparation v2 và split đã khóa; feature artifact; báo cáo/predictions validation sáu variants cũ. Hash cụ thể nằm ở reports/final_evaluation/lock.json. Các file bị loại khỏi zip phải được cung cấp qua nơi lưu local đã được phép. Không tạo lại split tùy ý, không sửa gold sau test.

## Môi trường

Môi trường nền: pyproject.toml + uv.lock, Python3.13.7, spaCy3.8.7, en_core_web_sm3.8.0. Tagger wheel: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl . Run thực tế còn có requirements-bert.txt; full versions ở environment.txt. Fresh inference chỉ cần NumPy2.5.3/PyTorch2.14.0+cpu và dependencies, xem clean_environment.json. Torch CPU lấy từ https://download.pytorch.org/whl/cpu; NumPy từ PyPI, tránh trộn index resolver không chủ ý.

## Chạy lại trên workspace đã có artifact

    .venv\\Scripts\\python.exe src/evaluation/nucle_final_evaluation.py replay
    .venv\\Scripts\\python.exe src/evaluation/verify_final_results.py
    .venv\\Scripts\\python.exe src/reporting/build_evaluation_tasks_report.py

Không chạy lại mode score lên output đã có; scorer từ chối. Replay chỉ kiểm bất biến, không model selection. Bootstrap từ saved predictions có thể tái tạo bằng mode bootstrap với cùng weights/seed. Khi thay môi trường, không sửa lock hoặc weights chỉ để cho kiểm tra pass.

## Tái tạo selection trong checkout/thư mục riêng

Chỉ khi đủ input có quyền truy cập và reports/final_evaluation chưa tồn tại:

    .venv\\Scripts\\python.exe src/training/nucle_final_selection.py
    .venv\\Scripts\\python.exe -m pytest tests/test_final_evaluation.py tests/test_nucle_features.py tests/test_nucle_benchmark.py -q -p no:cacheprovider
    .venv\\Scripts\\python.exe src/training/nucle_final_selection.py --freeze
    .venv\\Scripts\\python.exe src/evaluation/nucle_final_evaluation.py score
    .venv\\Scripts\\python.exe src/evaluation/nucle_final_evaluation.py bootstrap

Full−D phải tái sử dụng S+L+P+M. Test hiện đã được xem nên lượt tái tạo chỉ là reproduction của thiết kế cũ, không phải test mới độc lập. Script audit tạo packet trước; ghi chú AI đọc packet là artifact review, không thể tái lập bằng một câu lệnh như metric.

## Đọc artifact

selection_summary.json/fit_progress.json:52 fits và config; lock.json/lock.sha256:đóng băng trước test; test_predictions.csv:sample×pipeline (giữ local, ngoài zip); test_metrics.json:counts/F1; bootstrap.json:paired CI; manual_error_review.json:40 ghi chú text-free; report task DOCX/Markdown:kết quả các task vừa chạy; final report dự án để sau. Full_minus_D alias không được cộng thành lượt fit độc lập.
'''

def run():
    verify_lock();base=OUT/'reproducibility';base.mkdir(exist_ok=True)
    (base/'README.md').write_text(README,encoding='utf-8')
    packages=sorted((d.metadata['Name'],d.version) for d in metadata.distributions() if d.metadata.get('Name'))
    (OUT/'environment.txt').write_text('\n'.join(f'{n}=={v}' for n,v in packages)+'\n',encoding='utf-8')
    files=list((ROOT/'src').rglob('*.py'))
    files +=[ROOT/p for p in ['pyproject.toml','uv.lock','requirements-bert.txt','docs/final_evaluation_protocol.md','reports/summaries/evaluation/evaluation_tasks_report.docx','reports/summaries/evaluation/evaluation_tasks_report.md','note.md','docs/note_project_guide.md','reference.md','tests/test_final_evaluation.py','tests/test_nucle_features.py','tests/test_nucle_benchmark.py']]
    files +=[OUT/n for n in ['selection_summary.json','fit_progress.json','lock.json','lock.sha256','encoder_verification.json','bootstrap.json','test_metrics.json','overall_results.csv','per_class_results.csv','per_class_deltas.csv','paired_differences.csv','verification.json','checkpoint_replay.json','clean_environment.json','environment.txt','manual_error_review.json','benchmark_ci.png','confusion_matrices.png']]
    files.append(base/'README.md')
    # Explicit allowlist only: no wildcard into data, checkpoints, or private packets.
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}
    dump(base/'bundle_manifest.json',manifest);files.append(base/'bundle_manifest.json')
    archive=base/'code_protocol_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    restore=(ROOT/'.cache/evaluation_tasks_bundle_restore').resolve();restore.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            dest=(restore/name).resolve();assert dest.is_relative_to(restore)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    for name,h in manifest.items():assert sha(restore/name)==h,name
    dump(base/'restore_verification.json',{'archive_sha256':sha(archive),'files_verified':len(manifest),'restore_directory':str(restore.relative_to(ROOT)),'all_hashes_match':True,'contains_corpus_or_checkpoints':False})
    inventory={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    dump(OUT/'artifact_hashes.json',inventory)
    print('BUNDLE RESTORE VERIFIED',len(manifest),'files',archive.stat().st_size,'bytes')
if __name__=='__main__':run()
