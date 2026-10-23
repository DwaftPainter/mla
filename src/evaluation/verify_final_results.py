"""Independent saved-prediction checks; no fitting or model selection."""
from pathlib import Path
import sys,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.training.nucle_final_selection import OUT,read,dump,LABELS,sha
from src.evaluation.nucle_final_evaluation import load_predictions,verify_lock
from src.evaluation.metrics import scores

def run():
    lock=verify_lock();metadata,pred=load_predictions();reported=read(OUT/'test_metrics.json');b=read(OUT/'bootstrap.json')
    assert sha(OUT/'test_predictions.csv')==read(OUT/'test_scoring_complete.json')['predictions_sha256']
    y=[r['error_type'] for r in metadata]
    for name,p in pred.items():
        m=scores(y,p);assert abs(m['macro_f1']-reported['all']['models'][name]['macro_f1'])<1e-12
        assert m['confusion_matrix_true_rows_predicted_columns']==reported['all']['models'][name]['confusion']
    weights=np.load(OUT/'bootstrap_group_weights.npy');g=b['groups']
    assert np.array_equal(weights,np.random.default_rng(b['seed']).multinomial(len(g),np.ones(len(g))/len(g),size=1000))
    # Independently expand the first 25 resamples into sample lists.
    group_ids={group:[i for i,r in enumerate(metadata) if r['group_id']==group] for group in g}
    for w in weights[:25]:
        ids=[i for group,n in zip(g,w) for _ in range(int(n)) for i in group_ids[group]]
        yy=[y[i] for i in ids]
        for name in ['LR_03','POS','Tree_03']:
            got=scores(yy,[pred[name][i] for i in ids])['macro_f1']
            c=np.zeros((5,5),dtype=int)
            for group,n in zip(g,w):
                for i in group_ids[group]:c[LABELS.index(y[i]),LABELS.index(pred[name][i])]+=int(n)
            f=[]
            for k in range(5):
                den=c[k].sum()+c[:,k].sum();f.append(2*c[k,k]/den if den else 0)
            assert abs(got-sum(f)/5)<1e-12
    assert pred['LR_08']==pred['LR_02'] and pred['Tree_08']==pred['Tree_02']
    notes=read(OUT/'manual_error_review.json')['notes'];sel=read(OUT/'audit_selection.json')['rows']
    assert {str(r['annotation_id']) for r in notes}=={str(r['annotation_id']) for r in sel} and len(notes)==40
    assert read(OUT/'lock.json')['locked_at']<read(OUT/'test_scoring_started.json')['at']
    dump(OUT/'verification.json',{'all_20_prediction_metrics_exact':True,'bootstrap_weights_reproduced':True,'expanded_group_resample_checks':25*3,'alias_predictions_exact':True,'all_40_audit_notes_present':True,'locked_before_test':True,'lock_sha256':sha(OUT/'lock.json'),'test_prediction_sha256':sha(OUT/'test_predictions.csv'),'python_executable':sys.executable})
    print('VERIFIED: 20 metrics, 75 expanded group checks, aliases, audit, lock order')
if __name__=='__main__':run()
