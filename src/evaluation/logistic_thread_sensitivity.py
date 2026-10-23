"""Replay fixed LR on saved features at 4/10 threads; validation only."""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.training.nucle_benchmark import load_rows,matrix_for,FEATURE_DIR,SPLIT_DIR,LABELS
from src.models.nucle_logistic import fit_logistic
from src.evaluation.metrics import scores
from src.evaluation.preprocessing_sensitivity_audit import csvread
def run():
    features,targets,_,tr,va=load_rows(FEATURE_DIR,SPLIT_DIR)
    dim=len(json.loads((FEATURE_DIR/'encoder.json').read_text())['feature_names'])
    x=matrix_for(tr,features,dim); xv=matrix_for(va,features,dim)
    y=[LABELS.index(targets[s]['error_type']) for s in tr]
    old={r['sample_id']:r['best_lr'] for r in csvread(ROOT/'reports/nucle_feature_benchmark_solver_v2/validation_predictions.csv')}
    audit={r['sample_id']:r['lr'] for r in csvread(ROOT/'reports/contextual_model_validation/audit/v2_validation_predictions.csv')}
    results=[]
    for threads in [4,10]:
        torch.set_num_threads(threads); w,b,d=fit_logistic(x,y,0.1,False)
        with torch.no_grad(): pred={s:LABELS[i] for s,i in zip(va,(torch.sparse.mm(xv,w)+b).argmax(1).tolist())}
        results.append({'threads':threads,'macro_f1':scores([targets[s]['error_type'] for s in va],[pred[s] for s in va])['macro_f1'],'disagree_original':sum(pred[s]!=old[s] for s in va),'disagree_sensitivity':sum(pred[s]!=audit[s] for s in va),'convergence':d})
    (ROOT/'reports/contextual_model_validation/audit/thread_diagnostic.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
if __name__=='__main__': run()
