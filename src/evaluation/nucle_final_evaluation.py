"""Score frozen pipelines once; bootstrap saved predictions by connected group."""
from pathlib import Path
import csv,json,sys,random
from collections import defaultdict,Counter
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.training.nucle_final_selection import OUT,FEAT,PREP,SPLIT,LABELS,FULL,MASK,NONLEX,read,rows,dump,sha,now,load_encoder,masked_features
from src.training.nucle_ablation import sparse_tensor
from src.features.nucle_features import extract_features
from src.features.nucle_pos import load_tagger,align_span
from src.models.nucle_tree import GiniTree
from src.evaluation.metrics import scores

def verify_lock():
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    lock=read(OUT/'lock.json')
    for name,h in lock['hashes'].items(): assert sha(ROOT/name)==h, name
    return lock
def predict(entry,raw):
    remap={old:new for new,old in enumerate(entry['keep'])}
    x=[{remap[i]:v for i,v in r.items() if i in remap and v} for r in raw]
    cp=torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=True)
    if entry['model']=='LR': pred=(torch.sparse.mm(sparse_tensor(x,len(remap)),cp['w'])+cp['b']).argmax(1).tolist()
    else:
        tree=GiniTree(1,1,False,set());tree.root=cp['root'];pred=tree.predict(x)
    return [LABELS[i] for i in pred]
def matrix_metric(c, label_indices=None):
    c=np.asarray(c); idx=np.arange(5) if label_indices is None else np.asarray(label_indices)
    support=c.sum(-1); predicted=c.sum(-2); tp=np.diagonal(c,axis1=-2,axis2=-1)
    den=support+predicted; f1=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0)
    return f1[...,idx].mean(-1)
def subset_metrics(metadata,predictions):
    y=[LABELS.index(r['error_type']) for r in metadata]; output={}
    definitions={'all':(list(range(len(y))),list(range(5)))}
    for b in ['empty','single_token','multi_token']: definitions[b]=([i for i,r in enumerate(metadata) if r['span_bin']==b],list(range(5)))
    for pair,labels in [('Vform/SVA',['Vform','SVA']),('Nn/ArtOrDet',['Nn','ArtOrDet'])]:
        members=[i for i,r in enumerate(metadata) if r['error_type'] in labels]; li=[LABELS.index(l) for l in labels]
        definitions[pair]=(members,li)
        for pos in ['VERB','AUX','NOUN','DET']: definitions[pair+'@'+pos]=([i for i in members if metadata[i]['primary_upos']==pos],li)
    for name,(ids,li) in definitions.items():
        result={'support':len(ids),'groups':len({metadata[i]['group_id'] for i in ids}),'label_order':[LABELS[k] for k in li],'models':{}}
        for model,pred in predictions.items():
            c=np.zeros((5,5),dtype=int)
            for i in ids:c[y[i],LABELS.index(pred[i])]+=1
            score=scores([metadata[i]['error_type'] for i in ids],[pred[i] for i in ids]) if ids else None
            result['models'][model]={'macro_f1':float(matrix_metric(c,li)) if ids else None,'accuracy':float(np.trace(c)/len(ids)) if ids else None,'outside_label_predictions':sum(LABELS.index(pred[i]) not in li for i in ids),'confusion':c.tolist(),'per_label':score['per_label'] if score else {}}
        output[name]=result
    return output,definitions

def score_test():
    lock=verify_lock(); torch.set_num_threads(lock['threads'])
    if (OUT/'test_predictions.csv').exists() or (OUT/'test_scoring_started.json').exists():raise FileExistsError('Test already started; inspect saved artifacts, do not reselect')
    dump(OUT/'test_scoring_started.json',{'at':now(),'lock_sha256':sha(OUT/'lock.json')})
    split={r['sample_id']:r for r in rows(SPLIT/'split_manifest.csv')}; ids=[s for s,r in split.items() if r['partition']=='test']
    samples={}
    with (PREP/'samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if split[r['sample_id']]['partition']=='test': samples[r['sample_id']]=r
    assert len(ids)==len(samples)==2368
    texts={r['sentence_id']:r['sentence'] for r in samples.values()};nlp=load_tagger(); parsed=dict(zip(texts,nlp.pipe(texts.values(),batch_size=256)))
    full=load_encoder(OUT/'full_encoder.json');mask=load_encoder(OUT/'masked_encoder.json');pos=read(OUT/'pos_rule.json')
    assert all(split[s]['partition']=='train' for s in full.train_ids|mask.train_ids)
    stored={}
    with (FEAT/'features.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['partition']=='test':stored[r['sample_id']]={int(k):v for k,v in r['x'].items() if v}
    metadata=[]; frows=[];mrows=[];pr=[]
    for sid in ids:
        r=samples[sid];doc=parsed[r['sentence_id']]; inp={k:r[k] for k in ['sentence','error_start','error_end','error_span','span_kind']}
        f=full.transform_one(extract_features(inp,doc));assert {k:v for k,v in f.items() if v}==stored[sid],sid
        frows.append(f);mrows.append(mask.transform_one(masked_features(inp,doc)))
        a=align_span(doc,r['error_start'],r['error_end']);sig=a['pos_signature'];pr.append(pos['fixed'].get(sig,pos['lookup'].get(sig,pos['majority'])))
        count=len(r['span_token_indices']); assert count==len(a['span_token_indices'])
        metadata.append({k:r[k] for k in ['sample_id','annotation_id','essay_id','error_type']}|{'group_id':split[sid]['group_id'],'span_tokens':count,'span_bin':'empty' if r['error_start']==r['error_end'] else 'single_token' if count==1 else 'multi_token','primary_upos':a['primary_upos'],'pos_signature':sig,'alignment':r['alignment']})
    predictions={'Majority':[pos['majority']]*len(ids),'POS':pr}
    for e in lock['selected']:
        predictions[e['id']]=predictions[e['alias_of']] if 'alias_of' in e else predict(e,mrows if e['variant']==MASK else frows)
    with (OUT/'test_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metadata[0])+list(predictions));w.writeheader()
        for i,r in enumerate(metadata):w.writerow(r|{k:p[i] for k,p in predictions.items()})
    metrics,defs=subset_metrics(metadata,predictions);dump(OUT/'test_metrics.json',metrics)
    dump(OUT/'test_scoring_complete.json',{'at':now(),'sample_count':len(ids),'group_count':len({r['group_id'] for r in metadata}),'essay_count':len({r['essay_id'] for r in metadata}),'test_features_replay_exact':len(ids),'predictions_sha256':sha(OUT/'test_predictions.csv'),'lock_sha256':sha(OUT/'lock.json')})
    # Saved local features allow exact replay without fitting or reparsing.
    torch.save({'ids':ids,'full':frows,'masked':mrows},OUT/'test_encoded.pt')
    print('TEST SCORED', {k:round(v['macro_f1'],6) for k,v in metrics['all']['models'].items()},flush=True)

def load_predictions():
    rr=rows(OUT/'test_predictions.csv'); base=['sample_id','annotation_id','essay_id','error_type','group_id','span_tokens','span_bin','primary_upos','pos_signature','alignment']
    meta=[{k:r[k] for k in base} for r in rr];pred={k:[r[k] for r in rr] for k in rr[0] if k not in base}
    return meta,pred
def bootstrap():
    lock=verify_lock();metadata,pred=load_predictions();metrics,defs=subset_metrics(metadata,pred)
    groups=sorted({r['group_id'] for r in metadata});gindex={g:i for i,g in enumerate(groups)};rng=np.random.default_rng(lock['bootstrap_seed'])
    weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=lock['bootstrap_replicates'])
    np.save(OUT/'bootstrap_group_weights.npy',weights)
    results={}; distributions={}; index={(e['model'],e['variant']):e['id'] for e in lock['selected']}
    for name,(ids,li) in defs.items():
        truth_support=Counter(metadata[i]['error_type'] for i in ids);ng=len({metadata[i]['group_id'] for i in ids})
        eligible=bool(ids) and (name=='all' or (len(ids)>=20 and ng>=10 and all(truth_support[LABELS[k]] for k in li)))
        models=list(pred) if name=='all' else ['LR_03','POS']
        item={'support':len(ids),'groups':ng,'eligible':eligible,'models':{}}
        if eligible:
            for model in models:
                cms=np.zeros((len(groups),5,5),dtype=np.int64)
                for i in ids:cms[gindex[metadata[i]['group_id']],LABELS.index(metadata[i]['error_type']),LABELS.index(pred[model][i])]+=1
                boot=(weights@cms.reshape(len(groups),25)).reshape(-1,5,5);valid=boot.sum((1,2))>0
                dist=matrix_metric(boot,li);distributions[(name,model)]=(dist,valid)
                missing=(boot.sum(2)[:,li]==0).any(1)&valid
                ci=np.quantile(dist[valid],[.025,.975]).tolist() if sum(valid)>=950 else None
                item['models'][model]={'point':metrics[name]['models'][model]['macro_f1'],'ci95':ci,'valid_replicates':int(sum(valid)),'empty_replicates':int(sum(~valid)),'missing_class_replicates':int(sum(missing))}
        results[name]=item
    comparisons=[('LR Full - Majority','LR_03','Majority'),('Tree Full - Majority','Tree_03','Majority'),('LR Full - Tree Full','LR_03','Tree_03'),('LR Full - POS','LR_03','POS')]
    for model in ['LR','Tree']:
        for a,b in [('S+L+P','S+L'),('S+L+P+M','S+L+P'),(FULL,'S+L+P+M'),(FULL,MASK),(FULL,NONLEX),(FULL,'Full_minus_P'),(FULL,'Full_minus_M'),(FULL,'Full_minus_D')]:comparisons.append((model+' '+a+' minus '+b,index[(model,a)],index[(model,b)]))
    differences=[]
    for name in defs:
        comps=comparisons if name=='all' else [('LR Full - POS','LR_03','POS')]
        for desc,a,b in comps:
            if (name,a) not in distributions:continue
            da,va=distributions[(name,a)];db,vb=distributions[(name,b)];valid=va&vb
            differences.append({'slice':name,'contrast':desc,'a':a,'b':b,'difference':metrics[name]['models'][a]['macro_f1']-metrics[name]['models'][b]['macro_f1'],'ci95':np.quantile((da-db)[valid],[.025,.975]).tolist() if sum(valid)>=950 else None,'valid_replicates':int(sum(valid))})
    dump(OUT/'bootstrap.json',{'seed':lock['bootstrap_seed'],'replicates':len(weights),'groups':groups,'method':'paired connected-group percentile; fixed labels; pointwise intervals','slices':results,'differences':differences})
    print('BOOTSTRAP COMPLETE',len(differences),'contrasts',flush=True)

def replay():
    lock=verify_lock();metadata,pred=load_predictions(); encoded=torch.load(OUT/'test_encoded.pt',weights_only=True)
    assert encoded['ids']==[r['sample_id'] for r in metadata];torch.set_num_threads(lock['threads'])
    checks={}
    for e in lock['selected']:
        got=predict(e,encoded['masked'] if e['variant']==MASK else encoded['full']);checks[e['id']]=sum(a!=b for a,b in zip(got,pred[e['id']]))
    assert not any(checks.values());dump(OUT/'checkpoint_replay.json',{'at':now(),'disagreements':checks});print('CHECKPOINT REPLAY EXACT')
if __name__=='__main__':
    mode=sys.argv[1]
    {'score':score_test,'bootstrap':bootstrap,'replay':replay}[mode]()
