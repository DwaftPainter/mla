"""Train-only BERT gradient, readout, and 100-example memorization checks.

No validation/test metrics are computed. Existing checkpoints are read-only.
"""
from pathlib import Path
import csv,json,sys,time,hashlib,random,platform
from collections import Counter,defaultdict
from datetime import datetime,timezone
import torch
from torch.nn import functional as F
from transformers import BertTokenizerFast,BertForSequenceClassification
import transformers
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.training.bert_span_classifier import encode_span,new_model,seed_all,make_loader
from src.models.bert_span_readout import BertReadout,target_mask
from src.evaluation.metrics import scores
from src.evaluation.nucle_span_analysis import LABELS

OUT=ROOT/'reports/bert_learning_diagnostics'
SEED=20261023
def dump(p,o): p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def batch_for(rows,tok):
    batch=tok.pad([r[1] for r in rows],padding=True,return_tensors='pt')
    batch['labels']=torch.tensor([r[2] for r in rows],dtype=torch.long)
    return batch

def train_subset():
    source=json.loads((ROOT/'reports/contextual_model_validation/model_source.json').read_text())
    tok=BertTokenizerFast.from_pretrained(str(ROOT/source['path']),local_files_only=True)
    tok.add_special_tokens({'additional_special_tokens':['[SPAN_START]','[SPAN_END]']})
    partitions={r['sample_id']:r['partition'] for r in csvread(ROOT/'data_set/split/nucle_group_split/split_manifest.csv')}
    marker_ids=tok.convert_tokens_to_ids(['[SPAN_START]','[SPAN_END]'])
    assert len(set(marker_ids))==2 and tok.unk_token_id not in marker_ids
    rows=[]; signatures=defaultdict(list); counts=Counter(); audit=Counter(); sample_labels={}
    with (ROOT/'data_set/processed/nucle_preparation_v2/samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); sid=r['sample_id']
            if partitions[sid]!='train': continue
            features,stat=encode_span({k:r[k] for k in ['sentence','error_start','error_end','error_span']},tok)
            ids=features['input_ids']; start,end=(ids.index(i) for i in marker_ids)
            assert all(ids.count(i)==1 for i in marker_ids) and start<end
            assert ids[start+1:end]==tok.encode(r['error_span'],add_special_tokens=False)
            assert len(ids)<=128 and features['attention_mask']==[1]*len(ids)
            y=LABELS.index(r['error_type']); row=(sid,features,y); rows.append(row)
            signatures[tuple(ids)].append((sid,y)); counts[r['error_type']]+=1; sample_labels[sid]=r['error_type']
            audit['checked']+=1; audit['truncated']+=stat['truncated']; audit['empty_target_subwords']+=end==start+1
            audit['unknown_target_subwords']+=sum(i==tok.unk_token_id for i in ids[start+1:end])
    assert len(rows)==11052
    conflicts={k for k,v in signatures.items() if len({y for _,y in v})>1}
    candidates=defaultdict(list)
    for row in rows:
        if tuple(row[1]['input_ids']) not in conflicts: candidates[row[2]].append(row)
    rng=random.Random(SEED); selected=[]; used=set()
    for y in range(5):
        candidates[y].sort(key=lambda r:r[0]); rng.shuffle(candidates[y]); chosen=[]
        for r in candidates[y]:
            sig=tuple(r[1]['input_ids'])
            if sig in used: continue
            used.add(sig); chosen.append(r)
            if len(chosen)==20: break
        assert len(chosen)==20
        selected.extend(chosen)
    assert len({r[0] for r in selected})==100
    with (OUT/'selected_train_ids.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','label','partition','input_sha256']); w.writeheader()
        for sid,feat,y in selected: w.writerow({'sample_id':sid,'label':LABELS[y],'partition':'train','input_sha256':hashlib.sha256(json.dumps(feat,sort_keys=True).encode()).hexdigest()})
    dump(OUT/'input_audit.json',{'counts':dict(audit),'train_label_counts':dict(counts),'marker_ids':marker_ids,'marker_special':all(i in tok.all_special_ids for i in marker_ids),'duplicate_tokenized_groups':sum(len(v)>1 for v in signatures.values()),'conflicting_label_groups':len(conflicts),'selected_counts':dict(Counter(LABELS[r[2]] for r in selected)),'selection_seed':SEED,'validation_loaded':False,'test_loaded':False})
    return source,tok,selected

def no_dropout(base):
    for m in base.modules():
        if isinstance(m,torch.nn.Dropout): m.p=0.

def gradient_probe(base,rows,tok,mode):
    base.eval(); start,end=tok.convert_tokens_to_ids(['[SPAN_START]','[SPAN_END]'])
    model=BertReadout(base,mode,start,end); batch=batch_for(rows,tok)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.)
    params=[p for group in opt.param_groups for p in group['params']]
    assert len(params)==len({id(p) for p in params})
    assert {id(p) for p in params}=={id(p) for p in model.parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    o=model(**batch,output_hidden_states=True)
    for h in o.hidden_states: h.retain_grad()
    manual=F.cross_entropy(o.logits,batch['labels'])
    with torch.no_grad(): original=base(**batch)
    o.loss.backward(); groups={}; missing=[]; nonfinite=[]
    for name,p in model.named_parameters():
        if p.grad is None: missing.append(name); continue
        if not torch.isfinite(p.grad).all(): nonfinite.append(name)
        if '.encoder.layer.' in name: group='encoder.layer.'+name.split('.encoder.layer.')[1].split('.')[0]
        elif '.embeddings.' in name: group='embeddings'
        elif '.pooler.' in name: group='pooler'
        else: group='classifier'
        groups[group]=groups.get(group,0.)+float(p.grad.double().square().sum())
    groups={k:v**.5 for k,v in groups.items()}
    mask=target_mask(batch['input_ids'],batch['attention_mask'],start,end)
    emb=base.get_input_embeddings(); marker_norms={str(i):float(emb.weight.grad[i].norm()) for i in [start,end]}
    target_ids=batch['input_ids'][mask].unique()
    hidden=[]
    for i,h in enumerate(o.hidden_states):
        hidden.append({'layer_output':i,'cls_gradient_norm':float(h.grad[:,0].norm()),'target_gradient_norm':float(h.grad[mask].norm()),'marker_gradient_norm':float(h.grad[batch['input_ids'].eq(start)|batch['input_ids'].eq(end)].norm())})
    watched={'start_marker':(emb.weight,start),'end_marker':(emb.weight,end),'encoder_query':(base.bert.encoder.layer[0].attention.self.query.weight,None),'classifier':(base.classifier.weight,None)}
    before={k:p.detach().clone() if idx is None else p[idx].detach().clone() for k,(p,idx) in watched.items()}
    clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)); opt.step()
    changes={k:float(((p.detach() if idx is None else p[idx].detach())-before[k]).norm()) for k,(p,idx) in watched.items()}
    return {'mode':mode,'loss':float(o.loss.detach()),'manual_ce_abs_difference':float((manual-o.loss).abs().detach()),'original_cls_logits_max_difference':float((o.logits-original.logits).abs().max().detach()) if mode=='cls' else None,'optimizer_covers_all_trainable_parameters':True,'missing_gradients':missing,'nonfinite_gradients':nonfinite,'gradient_norms':groups,'marker_embedding_gradients':marker_norms,'target_embedding_gradient_norm':float(emb.weight.grad[target_ids].norm()),'hidden_gradients':hidden,'global_gradient_norm_before_clip':clip,'parameter_update_norms':changes}

def marker_probe(base,rows,tok):
    start,end=tok.convert_tokens_to_ids(['[SPAN_START]','[SPAN_END]'])
    batch=batch_for(rows,tok); moved=[]
    for _,features,_ in rows:
        ids=features['input_ids']; a=ids.index(start); b=ids.index(end)
        plain=[i for i in ids if i not in (start,end)]
        length=b-a-1; original_start=a
        locations=[j for j in range(1,len(plain)-length) if j!=original_start and (j+length<=original_start or j>=original_start+length)]
        if not locations: raise ValueError('No disjoint marker perturbation available')
        j=max(locations,key=lambda j:abs(j-original_start))
        perturbed=plain[:j]+[start]+plain[j:j+length]+[end]+plain[j+length:]
        assert [i for i in perturbed if i not in (start,end)]==plain
        moved.append({'input_ids':perturbed,'attention_mask':[1]*len(perturbed)})
    other=tok.pad(moved,padding=True,return_tensors='pt'); results={}
    base.eval()
    with torch.no_grad():
        for mode in ['cls','span_mean']:
            model=BertReadout(base,mode,start,end).eval()
            o=model(**batch,output_hidden_states=True); q=model(**other,output_hidden_states=True)
            if mode=='cls': rep=o.hidden_states[-1][:,0]; repq=q.hidden_states[-1][:,0]
            else:
                ma=target_mask(batch['input_ids'],batch['attention_mask'],start,end); mb=target_mask(other['input_ids'],other['attention_mask'],start,end)
                rep=(o.hidden_states[-1]*ma.unsqueeze(-1)).sum(1)/ma.sum(1,keepdim=True)
                repq=(q.hidden_states[-1]*mb.unsqueeze(-1)).sum(1)/mb.sum(1,keepdim=True)
            p=o.logits.softmax(-1); pq=q.logits.softmax(-1)
            results[mode]={'examples':len(rows),'mean_probability_L1':float((p-pq).abs().sum(1).mean()),'mean_logit_L2':float((o.logits-q.logits).norm(dim=1).mean()),'prediction_changes':int(o.logits.argmax(1).ne(q.logits.argmax(1)).sum()),'mean_representation_cosine':float(F.cosine_similarity(rep,repq,dim=1).mean())}
    # Padding invariance on the original trained CLS model.
    first=rows[0:1]; x=batch_for(first,tok)
    wider={k:F.pad(v,(0,13),value=tok.pad_token_id if k=='input_ids' else 0) for k,v in x.items() if k!='labels'}
    with torch.no_grad(): difference=float((base(**x).logits-base(**wider).logits).abs().max())
    results['padding_max_logit_difference']=difference
    return results

def assess(model,rows,tok):
    model.eval(); predicted=[]; total=0.
    with torch.no_grad():
        for i in range(0,len(rows),20):
            part=rows[i:i+20]; o=model(**batch_for(part,tok)); predicted+=o.logits.argmax(1).tolist(); total+=float(o.loss)*len(part)
    result=scores([LABELS[r[2]] for r in rows],[LABELS[p] for p in predicted]); result['cross_entropy']=total/len(rows)
    result['prediction_counts']=dict(Counter(LABELS[p] for p in predicted))
    return result,predicted

def memorize(source,tok,rows,mode,lr):
    name=f'{mode}_lr_{lr}'; directory=OUT/name; directory.mkdir()
    seed_all(SEED); base=new_model(source,tok); no_dropout(base)
    start,end=tok.convert_tokens_to_ids(['[SPAN_START]','[SPAN_END]']); model=BertReadout(base,mode,start,end)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    generator=torch.Generator().manual_seed(SEED); order=[]; cursor=0
    history=[]; consecutive=0; t=time.perf_counter(); updates=0
    initial_state_sha=hashlib.sha256(b''.join(v.detach().contiguous().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    config={'mode':mode,'learning_rate':lr,'seed':SEED,'sample_count':100,'batch_size':20,'max_updates':400,'evaluate_every':25,'dropout':0.,'weight_decay':0.,'scheduler':None,'clip_norm':1.,'criterion':'accuracy=1 and CE<=0.05 at two consecutive checks','initial_state_sha256':initial_state_sha,'partition':'train'}
    dump(directory/'config.json',config)
    result,pred=assess(model,rows,tok); history.append({'updates':0,**result,'seconds':0.})
    for step in range(1,401):
        if cursor>=len(order): order=torch.randperm(len(rows),generator=generator).tolist(); cursor=0
        subset=[rows[j] for j in order[cursor:cursor+20]]; cursor+=20
        model.train(); opt.zero_grad(set_to_none=True); loss=model(**batch_for(subset,tok)).loss
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); updates=step
        if step%25==0:
            result,pred=assess(model,rows,tok); history.append({'updates':step,**result,'seconds':time.perf_counter()-t}); dump(directory/'history.json',history)
            passed=result['accuracy']==1. and result['cross_entropy']<=.05; consecutive=consecutive+1 if passed else 0
            print(name,'step',step,'acc',round(result['accuracy'],3),'CE',round(result['cross_entropy'],4),flush=True)
            if consecutive>=2: break
    base.save_pretrained(directory/'checkpoint'); tok.save_pretrained(directory/'checkpoint')
    with (directory/'train_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_id','gold','prediction']); writer.writeheader()
        for r,p in zip(rows,pred): writer.writerow({'sample_id':r[0],'gold':LABELS[r[2]],'prediction':LABELS[p]})
    summary={'name':name,'config':config,'metrics':result,'updates':updates,'memorization_passed':consecutive>=2,'seconds':time.perf_counter()-t,'checkpoint_readout':mode,'validation_scored':False,'test_scored':False}
    dump(directory/'summary.json',summary)
    # Replay the custom readout explicitly (span checkpoint still uses the BERT base config).
    restored=BertForSequenceClassification.from_pretrained(directory/'checkpoint',local_files_only=True); no_dropout(restored)
    _,replay=assess(BertReadout(restored,mode,start,end),rows,tok)
    assert replay==pred
    return summary

def run():
    if OUT.exists(): raise FileExistsError('Use a new run directory; do not overwrite completed diagnostics')
    OUT.mkdir(parents=True); torch.set_num_threads(4)
    source,tok,rows=train_subset()
    source_selection=json.loads((ROOT/'reports/contextual_model_validation/bert/selection.json').read_text())
    checkpoint=ROOT/source_selection['primary_run']/'checkpoint'
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','threads':4,'seed':SEED,'source':source,'old_checkpoint':str(checkpoint.relative_to(ROOT)),'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'protocol_sha256':digest(ROOT/'docs/bert_learning_diagnostics_protocol.md'),'code_sha256':digest(Path(__file__)),'readout_code_sha256':digest(ROOT/'src/models/bert_span_readout.py'),'split_sha256':digest(ROOT/'data_set/split/nucle_group_split/split_manifest.csv'),'samples_sha256':digest(ROOT/'data_set/processed/nucle_preparation_v2/samples.jsonl'),'selected_ids_sha256':digest(OUT/'selected_train_ids.csv'),'validation_scored':False,'test_scored':False}
    dump(OUT/'manifest.json',manifest)
    balanced_batch=[rows[c*20+j] for c in range(5) for j in range(4)]
    gradients={}
    for origin in ['pretrained','saved_failed_run']:
        for mode in ['cls','span_mean']:
            seed_all(SEED)
            base=new_model(source,tok) if origin=='pretrained' else BertForSequenceClassification.from_pretrained(checkpoint,local_files_only=True)
            gradients[origin+'_'+mode]=gradient_probe(base,balanced_batch,tok,mode)
    dump(OUT/'gradient_checks.json',gradients)
    base=BertForSequenceClassification.from_pretrained(checkpoint,local_files_only=True)
    dump(OUT/'marker_sensitivity.json',marker_probe(base,rows,tok))
    print('Input, gradient and marker checks completed.',flush=True)
    results=[]
    for mode,lr in [('cls',5e-5),('cls',1e-3),('span_mean',1e-3)]:
        results.append(memorize(source,tok,rows,mode,lr))
    assert len({r['config']['initial_state_sha256'] for r in results})==1
    dump(OUT/'summary.json',{'scope':'train-only diagnostic; not validation performance','runs':results,'validation_scored':False,'test_scored':False,'old_results_preserved':True,'all_saved_predictions_replayed':True})
    print('All diagnostic runs finished.',flush=True)
if __name__=='__main__': run()
