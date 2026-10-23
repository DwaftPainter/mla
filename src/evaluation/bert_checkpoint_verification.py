"""Verify saved BERT checkpoints reproduce validation predictions; no test access."""
import csv,json,sys
from pathlib import Path
import torch
from transformers import BertForSequenceClassification
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.training.bert_span_classifier import OUT,load_data,make_loader,evaluate,dump

def run():
    torch.set_num_threads(4)
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    _,tok,data,samples,_=load_data(); rows=[]
    for config in selection['runs']:
        if config['lr']!=selection['selected_lr']: continue
        directory=ROOT/config['directory']
        model=BertForSequenceClassification.from_pretrained(directory/'checkpoint',local_files_only=True)
        assert model.config.num_labels==5 and model.get_input_embeddings().num_embeddings==len(tok)
        vm,pred,_=evaluate(model,make_loader(data['validation'],tok),samples)
        with (directory/'validation_predictions.csv').open(encoding='utf-8',newline='') as f: saved={r['sample_id']:r['prediction'] for r in csv.DictReader(f)}
        assert set(saved)==set(pred)
        disagreements=sum(saved[s]!=pred[s] for s in pred)
        assert disagreements==0
        tm,_,_=evaluate(model,make_loader(data['train'],tok),samples)
        rows.append({'run':config['directory'],'prediction_disagreements':disagreements,'validation':vm,'train':tm,'test_scored':False})
        print(config['seed'],'replay matches; train F1',tm['macro_f1'],flush=True)
    dump(OUT/'checkpoint_replay.json',rows)
if __name__=='__main__': run()
