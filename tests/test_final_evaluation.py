import json
import numpy as np
import pytest
from src.evaluation import nucle_final_evaluation as final
from src.training.nucle_final_selection import save_encoder,load_encoder
from src.training.nucle_ablation import feature_indices
from src.features.nucle_features import TrainOnlyEncoder
from src.evaluation.metrics import scores

def test_pair_metric_keeps_predictions_outside_pair_as_false_negatives():
    c=np.zeros((5,5),dtype=int);c[3,3]=1;c[3,0]=1;c[4,4]=1;c[4,1]=1
    assert final.matrix_metric(c,[3,4])==pytest.approx(2/3)

def test_group_resampling_counts_entire_group_and_agrees_with_expansion():
    # Two errors from one essay group must move together, never sampled independently.
    cms=np.zeros((2,5,5),dtype=int);cms[0,0,0]=1;cms[0,1,0]=1;cms[1,2,2]=1
    weights=np.array([[2,0],[0,2],[1,1]])
    boot=(weights@cms.reshape(2,25)).reshape(-1,5,5)
    assert boot.sum((1,2)).tolist()==[4,2,3]
    expected=scores(['ArtOrDet','Prep']*2,['ArtOrDet','ArtOrDet']*2)['macro_f1']
    assert final.matrix_metric(boot)[0]==pytest.approx(expected)

def test_encoder_serialization_preserves_unknown_and_numeric_features(tmp_path):
    e=TrainOnlyEncoder().fit([{'sample_id':'train','partition':'train','features':{'S|length':2.,'P|pattern':'NOUN|VERB','L|span_ngrams':'cat'}}])
    p=tmp_path/'encoder.json';save_encoder(e,p);r=load_encoder(p)
    f={'S|length':5.,'P|pattern':'unseen','L|span_ngrams':'dog'}
    assert e.transform_one(f)==r.transform_one(f)
    assert r.train_ids=={'train'}

def test_explicit_family_removal_and_full_minus_d_alias():
    names=['N|S|size','C|P|pos|NOUN','C|M|Number|Sing','C|D|head_pos|VERB','L|span_ngrams|x']
    assert feature_indices(names,'Full_minus_P',set('SLMD'))==[0,2,3,4]
    assert feature_indices(names,'Full_minus_D',set('SLPM'))==feature_indices(names,'S+L+P+M',set('SLPM'))

def test_lock_rejects_changed_frozen_input(tmp_path,monkeypatch):
    monkeypatch.setattr(final,'ROOT',tmp_path);monkeypatch.setattr(final,'OUT',tmp_path)
    p=tmp_path/'input.txt';p.write_text('original')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'hashes':{'input.txt':final.sha(p)}}))
    (tmp_path/'lock.sha256').write_text(final.sha(lock))
    final.verify_lock();p.write_text('changed')
    with pytest.raises(AssertionError):final.verify_lock()
