"""Finish validation ablations and freeze replayed checkpoints before test."""
from pathlib import Path
import csv, json, hashlib, sys, time, platform
from datetime import datetime, timezone
from collections import Counter, defaultdict
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.training.nucle_ablation import VARIANTS, feature_indices, remap_rows, sparse_tensor
from src.training.nucle_benchmark import load_rows
from src.features.nucle_features import TrainOnlyEncoder, extract_features, lexical_candidates, _span_and_context, SPECIAL
from src.features.nucle_pos import load_tagger, POSRule, align_span
from src.models.nucle_logistic import fit_logistic
from src.models.nucle_tree import GiniTree
from src.evaluation.metrics import scores

OUT=ROOT/'reports/final_evaluation'; FEAT=ROOT/'data_set/features/nucle_features_v1'
PREP=ROOT/'data_set/processed/nucle_preparation_v2'; SPLIT=ROOT/'data_set/split/nucle_group_split'
OLD=ROOT/'reports/nucle_feature_ablation_v1'
LABELS=['ArtOrDet','Prep','Nn','Vform','SVA']
FULL='S+L+P+M+D'; MASK='Full_mask_span_lexical'; NONLEX='S+P+M+D_nonlexical'
EXTRA=[('Full_minus_P',set('SLMD'),'Remove explicit P family'),('Full_minus_M',set('SLPD'),'Remove explicit M family')]

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def save_encoder(e,p):
    dump(p,{k:getattr(e,k) for k in ['numeric_keys','categories','lexical_vocab','feature_names']} | {'train_ids':sorted(e.train_ids)})
def load_encoder(p):
    d=read(p); e=TrainOnlyEncoder()
    for k,v in d.items(): setattr(e,k,v)
    e.train_ids=set(e.train_ids); e.lexical_vocab=[tuple(x) for x in e.lexical_vocab]; e.index={n:i for i,n in enumerate(e.feature_names)}
    return e
def restore_full():
    d=read(FEAT/'encoder.json'); e=TrainOnlyEncoder(); e.feature_names=d['feature_names']
    e.index={n:i for i,n in enumerate(e.feature_names)}; e.numeric_keys=[n[2:] for n in e.feature_names if n.startswith('N|')]
    cats=defaultdict(list)
    for n in e.feature_names:
        if n.startswith('C|'):
            _,family,key,value=n.split('|',3); cats[family+'|'+key].append(value)
    e.categories=dict(cats); e.lexical_vocab=[tuple(x) for x in d['lexical_vocabulary']]
    return e
def masked_features(sample,doc):
    f=extract_features(sample,doc); _,left,right,_=_span_and_context(doc,sample['error_start'],sample['error_end'])
    terms=lexical_candidates(doc,[],left,right); f['L|span_unigrams']=SPECIAL
    f['L|span_ngrams']='\x1f'.join(' '.join(t[1:]) for t in terms) or SPECIAL
    return f
def make_encoders(raw,split,train_ids):
    samples={}
    with (PREP/'samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if split[r['sample_id']]['partition'] in {'train','validation'}: samples[r['sample_id']]=r
    texts={r['sentence_id']:r['sentence'] for r in samples.values()}; nlp=load_tagger()
    parsed=dict(zip(texts,nlp.pipe(texts.values(),batch_size=256)))
    e=restore_full(); e.train_ids=set(train_ids); masked=[]; pos_train=[]; val_pos={}
    for sid,r in samples.items():
        doc=parsed[r['sentence_id']]; inp={k:r[k] for k in ['sentence','error_start','error_end','error_span','span_kind']}
        transformed=e.transform_one(extract_features(inp,doc))
        assert {k:v for k,v in transformed.items() if v}=={k:v for k,v in raw[sid].items() if v},sid
        part=split[sid]['partition']; masked.append({'sample_id':sid,'partition':part,'features':masked_features(inp,doc)})
        pos={'error_type':r['error_type'],**align_span(doc,r['error_start'],r['error_end'])}
        if part=='train': pos_train.append(pos)
        else: val_pos[sid]=pos
    me=TrainOnlyEncoder().fit([r for r in masked if r['partition']=='train'])
    save_encoder(e,OUT/'full_encoder.json'); save_encoder(me,OUT/'masked_encoder.json')
    rule=POSRule().fit(pos_train)
    saved={r['sample_id']:r['prediction'] for r in rows(ROOT/'reports/nucle_pos_baseline_v1/validation_predictions.csv')}
    assert all(rule.predict(p)==saved[sid] for sid,p in val_pos.items())
    dump(OUT/'pos_rule.json',{'majority':rule.majority,'lookup':{s:rule._winner(c) for s,c in rule.signature_counts.items()},'fixed':rule.FIXED})
    return {r['sample_id']:me.transform_one(r['features']) for r in masked},me.feature_names

def fit_one(model,config,tr,va,yt,names):
    if model=='LR':
        x=sparse_tensor(tr,len(names)); w,b,conv=fit_logistic(x,yt,config['C'],config['class_weight']=='balanced')
        pred=(torch.sparse.mm(sparse_tensor(va,len(names)),w)+b).argmax(1).tolist()
        tp=(torch.sparse.mm(x,w)+b).argmax(1).tolist(); payload={'w':w,'b':b}
    else:
        tree=GiniTree(config['max_depth'],config['min_samples_leaf'],config['class_weight']=='balanced',{i for i,n in enumerate(names) if n.startswith('N|')}).fit(tr,yt)
        pred=tree.predict(va); tp=tree.predict(tr); payload={'root':tree.root}; conv={'converged':True,'termination':'deterministic_tree'}
    return payload,[LABELS[i] for i in pred],sum(a==b for a,b in zip(tp,yt))/len(yt),conv

def grid(model):
    if model=='LR': return [{'C':c,'class_weight':'balanced' if b else None} for c in [.01,.1,1.,10.] for b in [False,True]]
    return [{'max_depth':d,'min_samples_leaf':l,'class_weight':'balanced' if b else None} for d in [4,8,12] for l in [5,20] for b in [False,True]]

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(); torch.set_num_threads(10); start=time.perf_counter()
    dump(OUT/'selection_started.json',{'at':now(),'protocol_sha256':sha(ROOT/'docs/final_evaluation_protocol.md'),'test_scored':False})
    raw,targets,split,trids,vaids=load_rows(FEAT,SPLIT)
    for name,h in read(FEAT/'manifest.json')['outputs'].items(): assert sha(FEAT/name)==h
    old=read(OLD/'summary.json'); assert old['split_manifest_sha256']==sha(SPLIT/'split_manifest.csv')
    names=read(FEAT/'encoder.json')['feature_names']; yt=[LABELS.index(targets[s]['error_type']) for s in trids]; yv=[targets[s]['error_type'] for s in vaids]
    print('Verifying train/validation feature transform and reconstructing encoders',flush=True)
    masked,mnames=make_encoders(raw,split,trids)
    dump(OUT/'encoder_verification.json',{'full_rows_exact':len(raw),'pos_validation_exact':len(vaids),'train_only':True})
    historic=defaultdict(dict)
    for r in rows(OLD/'validation_predictions.csv'): historic[(r['variant'],r['model'])][r['sample_id']]=r['prediction']
    entries=[]; allfits=[]; valpred=[]
    for vi,(variant,groups,_) in enumerate([*VARIANTS,*EXTRA]):
        src=masked if variant==MASK else raw; base=mnames if variant==MASK else names
        keep=feature_indices(base,variant,groups); local=[base[i] for i in keep]
        assert all(n.split('|')[1] in groups if n.startswith(('N|','C|')) else 'L' in groups for n in local)
        tr=remap_rows(trids,src,keep); va=remap_rows(vaids,src,keep)
        for model,oldname in [('LR','LogisticRegression'),('Tree','DecisionTree')]:
            configs=grid(model) if variant.startswith('Full_minus_') else [old['selected_by_validation'][variant][oldname]['configuration']]
            best=None
            for config in configs:
                clock=time.perf_counter(); payload,pred,ta,conv=fit_one(model,config,tr,va,yt,local)
                metrics=scores(yv,pred); result={'variant':variant,'model':model,'configuration':config,'feature_count':len(keep),'fit_seconds':time.perf_counter()-clock,'train_accuracy':ta,'convergence':conv,**metrics}
                allfits.append(result); dump(OUT/'fit_progress.json',allfits)
                print(json.dumps({k:result[k] for k in ['variant','model','configuration','macro_f1','fit_seconds']}),flush=True)
                if conv['converged'] and (best is None or metrics['macro_f1']>best[0]['macro_f1']): best=(result,payload,pred)
            assert best is not None,(variant,model)
            result,payload,pred=best
            if not variant.startswith('Full_minus_'):
                mismatch=sum(p!=historic[(variant,oldname)][sid] for sid,p in zip(vaids,pred)); assert mismatch==0,(variant,model,mismatch)
            ident=f'{model}_{vi:02d}'; cp=OUT/'checkpoints'/f'{ident}.pt'; cp.parent.mkdir(exist_ok=True)
            torch.save(payload,cp)
            entry={'id':ident,'variant':variant,'model':model,'configuration':result['configuration'],'feature_count':len(keep),'keep':keep,'encoder':'masked_encoder.json' if variant==MASK else 'full_encoder.json','checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'validation_macro_f1':result['macro_f1'],'validation_accuracy':result['accuracy'],'convergence':result['convergence'],'historical_replay_exact':not variant.startswith('Full_minus_')}
            entries.append(entry); valpred.extend({'sample_id':sid,'error_type':y,'pipeline':ident,'prediction':p} for sid,y,p in zip(vaids,yv,pred))
            dump(OUT/'selected_progress.json',entries)
    for e in list(entries):
        if e['variant']=='S+L+P+M': entries.append({**e,'id':e['model']+'_08','variant':'Full_minus_D','alias_of':e['id']})
    with (OUT/'validation_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','error_type','pipeline','prediction']);w.writeheader();w.writerows(valpred)
    dump(OUT/'selection_summary.json',{'at':now(),'threads':10,'unique_fits':len(allfits),'new_grid_fits':40,'historical_refits':12,'full_minus_D_reused':True,'runtime_seconds':time.perf_counter()-start,'selected':entries,'test_scored':False})
    print('SELECTION COMPLETE; run freeze separately after verification',flush=True)

def freeze():
    assert not (OUT/'lock.json').exists()
    s=read(OUT/'selection_summary.json'); assert len(s['selected'])==18 and s['unique_fits']==52
    paths=[PREP/'samples.jsonl',PREP/'manifest.json',PREP/'manifest.sha256',SPLIT/'split_manifest.csv',SPLIT/'essay_groups.csv',FEAT/'features.jsonl',FEAT/'targets.jsonl',FEAT/'encoder.json',FEAT/'manifest.json',ROOT/'docs/final_evaluation_protocol.md',ROOT/'uv.lock',ROOT/'pyproject.toml']
    paths+=list((ROOT/'src').rglob('*.py'))+list((OUT/'checkpoints').glob('*.pt'))
    paths +=[OUT/n for n in ['full_encoder.json','masked_encoder.json','pos_rule.json','selection_summary.json','encoder_verification.json','validation_predictions.csv']]
    lock={'locked_at':now(),'label_order':LABELS,'threads':10,'bootstrap_seed':20261023,'bootstrap_replicates':1000,'python':platform.python_version(),'torch':str(torch.__version__),'selected':s['selected'],'hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},'test_scored_at_lock':False}
    dump(OUT/'lock.json',lock); (OUT/'lock.sha256').write_text(sha(OUT/'lock.json'),encoding='ascii'); print('LOCKED',sha(OUT/'lock.json'))
if __name__=='__main__':
    if '--freeze' in sys.argv: freeze()
    else: run()
