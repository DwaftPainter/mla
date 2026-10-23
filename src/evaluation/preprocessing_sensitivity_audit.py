"""Validation-only alignment audit and fixed-configuration v1/v2 sensitivity."""
from pathlib import Path
import csv, json, sys, hashlib
from collections import Counter
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.features.nucle_features import TrainOnlyEncoder, extract_features
from src.features.nucle_pos import load_tagger, align_span, POSRule
from src.training.nucle_benchmark import matrix_for
from src.models.nucle_logistic import fit_logistic
from src.evaluation.nucle_span_analysis import metric, LABELS

OUT=ROOT/'reports/contextual_model_validation/audit'
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def run():
    OUT.mkdir(parents=True,exist_ok=False); torch.set_num_threads(4)
    splitrows=csvread(ROOT/'data_set/split/nucle_group_split/split_manifest.csv')
    split={r['sample_id']:r['partition'] for r in splitrows}
    train=[s for s in split if split[s]=='train']; val=[s for s in split if split[s]=='validation']
    keep=set(train+val); versions={}; nlp=load_tagger(); results={}; parsed={}
    for v in ('v1','v2'):
        rows={}
        with (ROOT/f'data_set/processed/nucle_preparation_{v}/samples.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['sample_id'] in keep: rows[r['sample_id']]=r
        assert set(rows)==keep
        texts=list(dict.fromkeys(r['sentence'] for r in rows.values()))
        docs=dict(zip(texts,nlp.pipe(texts,batch_size=128)))
        features={}; aligned={}; offset_fail=[]; token_mismatch=[]
        for sid,r in rows.items():
            assert r['sentence'][r['error_start']:r['error_end']]==r['error_span']
            doc=docs[r['sentence']]; aligned[sid]=align_span(doc,r['error_start'],r['error_end'])
            features[sid]=extract_features(r,doc)
            if len(aligned[sid]['span_token_indices'])!=len(r['span_token_indices']): token_mismatch.append(sid)
        enc=TrainOnlyEncoder().fit([{'sample_id':s,'partition':'train','features':features[s]} for s in train])
        vectors={s:enc.transform_one(f) for s,f in features.items()}
        x=matrix_for(train,vectors,len(enc.feature_names)); xv=matrix_for(val,vectors,len(enc.feature_names))
        w,b,diag=fit_logistic(x,[LABELS.index(rows[s]['error_type']) for s in train],0.1,False)
        with torch.no_grad(): pred={s:LABELS[i] for s,i in zip(val,(torch.sparse.mm(xv,w)+b).argmax(1).tolist())}
        pos=POSRule().fit([dict(aligned[s],error_type=rows[s]['error_type']) for s in train])
        pp={s:pos.predict(aligned[s]) for s in val}
        result={'macro':metric(val,rows,pred,LABELS),'pos_macro':metric(val,rows,pp,LABELS),'dimension':len(enc.feature_names),'convergence':diag,'alignment_counts_validation':dict(Counter(aligned[s]['span_alignment'] for s in val)),'token_count_mismatch_ids':token_mismatch,'offset_failures':offset_fail,'test_scored':False}
        results[v]=result; versions[v]=rows
        with (OUT/f'{v}_validation_predictions.csv').open('w',encoding='utf-8',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['sample_id','error_type','lr','pos']); wr.writeheader()
            for s in val: wr.writerow(dict(sample_id=s,error_type=rows[s]['error_type'],lr=pred[s],pos=pp[s]))
        if v=='v2': parsed=docs; v2pred=pred; v2aligned=aligned
        print(v, result['macro']['macro_f1'],diag,flush=True)
    changed=[s for s in keep if any(versions['v1'][s][k]!=versions['v2'][s][k] for k in ('sentence','error_start','error_end'))]
    for s in keep: assert versions['v1'][s]['error_type']==versions['v2'][s]['error_type']
    results['changed_counts']={p:sum(split[s]==p for s in changed) for p in ('train','validation')}
    results['method']='Fixed LR C=0.1, class_weight=None; refit train-only encoder per preparation; same locked IDs/split; no retuning.'
    old={r['sample_id']:r['best_lr'] for r in csvread(ROOT/'reports/nucle_feature_benchmark_solver_v2/validation_predictions.csv')}
    results['v2_reproduction_prediction_disagreements']=sum(v2pred[s]!=old[s] for s in val)
    for v in ('v1','v2'):
        preds={r['sample_id']:r['lr'] for r in csvread(OUT/f'{v}_validation_predictions.csv')}
        for name,ids in [('changed',[s for s in val if s in changed]),('unchanged',[s for s in val if s not in changed])]:
            results[v][name]=metric(ids,versions[v],preds,LABELS)
        signatures={}
        for s,r in versions[v].items(): signatures.setdefault((r['sentence'],r['error_start'],r['error_end']),set()).add(split[s])
        results[v]['exact_input_signatures_cross_train_validation']=sum(len(p)>1 for p in signatures.values())
    # Deterministic purposeful sample, not a prevalence estimate.
    selected=[]; rows=versions['v2']
    for truth,prediction in [('Vform','SVA'),('SVA','Vform'),('Nn','ArtOrDet'),('ArtOrDet','Nn')]:
        ids=sorted(s for s in val if rows[s]['error_type']==truth and old[s]==prediction)
        selected.extend(ids[:3])
    selected+=sorted(s for s in val if s not in selected and len(rows[s]['span_token_indices'])>1 and old[s]!=rows[s]['error_type'])[:6]
    packet=[]
    for s in selected:
        r=rows[s]; doc=parsed[r['sentence']]
        packet.append({'sample_id':s,'annotation_id':r['annotation_id'],'gold':r['error_type'],'prediction':old[s],'sentence':r['sentence'],'span':r['error_span'],'start':r['error_start'],'end':r['error_end'],'alignment':v2aligned[s],'parse':[{'i':t.i,'text':t.text,'pos':t.pos_,'dep':t.dep_,'head':t.head.i,'morph':str(t.morph)} for t in doc]})
    dump(OUT/'private_audit_packet.json',packet)
    results['audit_sample_ids']=selected
    results['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data_set/split/nucle_group_split/split_manifest.csv',ROOT/'data_set/processed/nucle_preparation_v1/samples.jsonl',ROOT/'data_set/processed/nucle_preparation_v2/samples.jsonl']}
    dump(OUT/'summary.json',results)
if __name__=='__main__': run()
