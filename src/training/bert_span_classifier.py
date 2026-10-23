"""Local, validation-only fine-tuning of a compact pretrained BERT.

Run --pilot first; write protocol before --train. No test prediction path.
"""
from pathlib import Path
import argparse, csv, hashlib, json, random, sys, time, platform
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
import transformers
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.evaluation.nucle_span_analysis import LABELS, metric
OUT=ROOT/'reports/contextual_model_validation/bert'
SEEDS=(20261013,20261014,20261015); LRS=(3e-5,5e-5)
MAX_LENGTH=128; EPOCHS=5; BATCH=32

def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
def encode_span(row,tokenizer,max_length=MAX_LENGTH):
    text=row['sentence']; start,end=row['error_start'],row['error_end']
    if not 0<=start<=end<=len(text) or text[start:end]!=row['error_span']: raise ValueError('Invalid original span')
    # Three regions preserve the complete annotated span even at partial-token boundaries.
    left=tokenizer.encode(text[:start],add_special_tokens=False)
    span=tokenizer.encode(text[start:end],add_special_tokens=False)
    right=tokenizer.encode(text[end:],add_special_tokens=False)
    budget=max_length-4-len(span)
    if budget<0: raise ValueError('Annotated span exceeds available sequence budget')
    nl=min(len(left),budget//2); nr=min(len(right),budget-nl)
    nl=min(len(left),budget-nr)
    ids=[tokenizer.cls_token_id]+(left[-nl:] if nl else [])+[tokenizer.convert_tokens_to_ids('[SPAN_START]')]+span+[tokenizer.convert_tokens_to_ids('[SPAN_END]')]+right[:nr]+[tokenizer.sep_token_id]
    assert len(ids)<=max_length
    return {'input_ids':ids,'attention_mask':[1]*len(ids)}, {'truncated':nl<len(left) or nr<len(right),'original_subwords':len(left)+len(span)+len(right)+4,'span_subwords':len(span)}

def load_data():
    source=json.loads((OUT.parent/'model_source.json').read_text())
    tok=BertTokenizerFast.from_pretrained(str(ROOT/source['path']),local_files_only=True)
    tok.add_special_tokens({'additional_special_tokens':['[SPAN_START]','[SPAN_END]']})
    with (ROOT/'data_set/split/nucle_group_split/split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        splits={r['sample_id']:r['partition'] for r in csv.DictReader(f)}
    datasets={'train':[],'validation':[]}; samples={}; stats={p:[] for p in datasets}
    with (ROOT/'data_set/processed/nucle_preparation_v2/samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line); sid=row['sample_id']; part=splits[sid]
            if part not in datasets: continue
            features,stat=encode_span({k:row[k] for k in ['sentence','error_start','error_end','error_span']},tok)
            datasets[part].append((sid,features,LABELS.index(row['error_type'])))
            samples[sid]={'error_type':row['error_type']}; stats[part].append(stat)
    assert len(datasets['train'])==11052 and len(datasets['validation'])==2368
    assert not ({r[0] for r in datasets['train']}&{r[0] for r in datasets['validation']})
    return source,tok,datasets,samples,{p:{'count':len(s),'truncated':sum(r['truncated'] for r in s),'max_original_subwords':max(r['original_subwords'] for r in s),'max_span_subwords':max(r['span_subwords'] for r in s)} for p,s in stats.items()}

def make_loader(rows,tok,shuffle=False,seed=0):
    def collate(batch):
        ids,features,labels=zip(*batch)
        encoded=tok.pad(list(features),padding=True,return_tensors='pt')
        encoded['labels']=torch.tensor(labels,dtype=torch.long)
        return ids,encoded
    return DataLoader(rows,batch_size=BATCH,shuffle=shuffle,collate_fn=collate,generator=torch.Generator().manual_seed(seed),num_workers=0)
def new_model(source,tok):
    model=BertForSequenceClassification.from_pretrained(str(ROOT/source['path']),num_labels=5,local_files_only=True,id2label=dict(enumerate(LABELS)),label2id={s:i for i,s in enumerate(LABELS)})
    model.resize_token_embeddings(len(tok),mean_resizing=False)
    return model
def evaluate(model,loader,samples):
    model.eval(); pred={}; loss=0; n=0
    with torch.no_grad():
        for ids,batch in loader:
            o=model(**batch); loss+=float(o.loss)*len(ids); n+=len(ids)
            pred.update(zip(ids,[LABELS[i] for i in o.logits.argmax(1).tolist()]))
    return metric(list(pred),samples,pred,LABELS),pred,loss/n

def run(pilot=False):
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    source,tok,data,samples,stats=load_data()
    if pilot:
        seed_all(SEEDS[0]); model=new_model(source,tok); opt=torch.optim.AdamW(model.parameters(),lr=LRS[0]); model.train(); t=time.perf_counter()
        count=0
        for _,batch in make_loader(data['train'],tok,True,SEEDS[0]):
            opt.zero_grad(); loss=model(**batch).loss; loss.backward(); opt.step(); count+=1
            if count==20: break
        seconds=time.perf_counter()-t
        dump(OUT/'pilot.json',{'steps':count,'seconds':seconds,'seconds_per_step':seconds/count,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'device':'cpu','threads':4,'tokenization':stats,'source':source,'test_scored':False})
        print('Pilot',seconds,'seconds',flush=True); return
    if not (ROOT/'docs/bert_experiment_protocol.md').exists(): raise RuntimeError('Write protocol before training')
    if (OUT/'selection.json').exists(): raise FileExistsError('Completed run already exists')
    env={'started_utc':datetime.now(timezone.utc).isoformat(),'source':source,'tokenization':stats,'device':'cpu','threads':4,'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'test_scored':False,'protocol_sha256':sha(ROOT/'docs/bert_experiment_protocol.md'),'code_sha256':sha(Path(__file__)),'split_sha256':sha(ROOT/'data_set/split/nucle_group_split/split_manifest.csv'),'samples_sha256':sha(ROOT/'data_set/processed/nucle_preparation_v2/samples.jsonl')}
    dump(OUT/'manifest.json',env)
    val_loader=make_loader(data['validation'],tok); runs=[]
    def train_one(lr,seed):
        directory=OUT/f'lr_{lr}_seed_{seed}'; directory.mkdir(exist_ok=False)
        seed_all(seed); model=new_model(source,tok)
        loader=make_loader(data['train'],tok,True,seed)
        decay=[p for n,p in model.named_parameters() if not n.endswith('bias') and 'LayerNorm.weight' not in n]
        no_decay=[p for n,p in model.named_parameters() if n.endswith('bias') or 'LayerNorm.weight' in n]
        optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':0.01},{'params':no_decay,'weight_decay':0}],lr=lr,eps=1e-8)
        scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*len(loader)*EPOCHS),len(loader)*EPOCHS)
        best=-1.; logs=[]; started=time.perf_counter()
        config={'lr':lr,'seed':seed,'epochs':EPOCHS,'batch_size':BATCH,'max_length':MAX_LENGTH,'selection':'validation macro-F1, earliest epoch on ties','optimizer':'AdamW','warmup_ratio':.1,'weight_decay':.01,'gradient_clip':1.,'dropout':model.config.hidden_dropout_prob}
        dump(directory/'run_config.json',config)
        for epoch in range(1,EPOCHS+1):
            model.train(); loss_sum=0; n=0
            for _,batch in loader:
                optimizer.zero_grad(); loss=model(**batch).loss
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step()
                loss_sum+=float(loss.detach())*len(batch['labels']); n+=len(batch['labels'])
            scores,pred,val_loss=evaluate(model,val_loader,samples)
            logs.append({'epoch':epoch,'train_loss':loss_sum/n,'validation_loss':val_loss,'macro_f1':scores['macro_f1'],'elapsed_seconds':time.perf_counter()-started})
            dump(directory/'history.json',logs)
            print('lr',lr,'seed',seed,'epoch',epoch,'F1',round(scores['macro_f1'],6),'seconds',round(logs[-1]['elapsed_seconds']),flush=True)
            if scores['macro_f1']>best:
                best=scores['macro_f1']; best_epoch=epoch; best_scores=scores
                model.save_pretrained(directory/'checkpoint'); tok.save_pretrained(directory/'checkpoint')
                with (directory/'validation_predictions.csv').open('w',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=['sample_id','error_type','prediction']); writer.writeheader()
                    for sid in pred: writer.writerow({'sample_id':sid,'error_type':samples[sid]['error_type'],'prediction':pred[sid]})
        result={**config,'best_epoch':best_epoch,'metrics':best_scores,'seconds':time.perf_counter()-started,'directory':str(directory.relative_to(ROOT))}
        dump(directory/'summary.json',result); return result
    for lr in LRS: runs.append(train_one(lr,SEEDS[0]))
    selected=max(runs,key=lambda r:r['metrics']['macro_f1'])
    for seed in SEEDS[1:]: runs.append(train_one(selected['lr'],seed))
    selected_runs=[r for r in runs if r['lr']==selected['lr']]
    values=[r['metrics']['macro_f1'] for r in selected_runs]
    dump(OUT/'selection.json',{'selected_lr':selected['lr'],'selection_seed':SEEDS[0],'runs':runs,'selected_lr_macro_f1_mean':float(np.mean(values)),'selected_lr_macro_f1_sample_std':float(np.std(values,ddof=1)),'primary_seed':SEEDS[0],'primary_run':selected['directory'],'test_scored':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--pilot',action='store_true'); parser.add_argument('--train',action='store_true'); args=parser.parse_args()
    if args.pilot==args.train: parser.error('Choose exactly one of --pilot or --train')
    run(pilot=args.pilot)
