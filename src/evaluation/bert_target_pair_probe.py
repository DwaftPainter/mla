"""Stress-test the failed CLS checkpoint on distinct gold spans in one train sentence."""
from pathlib import Path
import csv,json,sys,random,hashlib
from collections import defaultdict
import torch
from transformers import BertTokenizerFast,BertForSequenceClassification
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.training.bert_span_classifier import encode_span
from src.models.bert_span_readout import target_mask
from src.evaluation.nucle_span_analysis import LABELS

def run():
    out=ROOT/'reports/bert_learning_diagnostics'; path=out/'same_sentence_pairs.json'
    if path.exists(): raise FileExistsError(path)
    with (ROOT/'data_set/split/nucle_group_split/split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        train={r['sample_id'] for r in csv.DictReader(f) if r['partition']=='train'}
    grouped=defaultdict(list)
    with (ROOT/'data_set/processed/nucle_preparation_v2/samples.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['sample_id'] in train: grouped[r['sentence']].append(r)
    candidates=[]
    for sentence,rows in grouped.items():
        rows.sort(key=lambda r:r['sample_id'])
        pairs=[(a,b) for i,a in enumerate(rows) for b in rows[i+1:] if a['error_type']!=b['error_type'] and (a['error_start'],a['error_end'])!=(b['error_start'],b['error_end'])]
        if pairs: candidates.append((hashlib.sha256(sentence.encode()).hexdigest(),pairs[0]))
    candidates.sort(key=lambda r:r[0]); random.Random(20261023).shuffle(candidates); chosen=candidates[:50]
    selection=json.loads((ROOT/'reports/contextual_model_validation/bert/selection.json').read_text()); checkpoint=ROOT/selection['primary_run']/'checkpoint'
    tok=BertTokenizerFast.from_pretrained(checkpoint,local_files_only=True); model=BertForSequenceClassification.from_pretrained(checkpoint,local_files_only=True).eval(); torch.set_num_threads(4)
    start,end=tok.convert_tokens_to_ids(['[SPAN_START]','[SPAN_END]']); results=[]
    with torch.no_grad():
        for sentence_hash,(a,b) in chosen:
            encoded=[encode_span(r,tok)[0] for r in [a,b]]; batch=tok.pad(encoded,padding=True,return_tensors='pt')
            pred=model(**batch,output_hidden_states=True); labels=[LABELS[i] for i in pred.logits.argmax(1).tolist()]
            mask=target_mask(batch['input_ids'],batch['attention_mask'],start,end)
            hidden=pred.hidden_states[-1]; mean=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            results.append({'sentence_sha256':sentence_hash,'sample_ids':[a['sample_id'],b['sample_id']],'gold':[a['error_type'],b['error_type']],'prediction':labels,'predictions_differ':labels[0]!=labels[1],'both_correct':labels==[a['error_type'],b['error_type']],'cls_cosine':float(torch.nn.functional.cosine_similarity(hidden[0,0][None],hidden[1,0][None])),'span_mean_cosine':float(torch.nn.functional.cosine_similarity(mean[0][None],mean[1][None]))})
    result={'scope':'train only, diagnostic; existing failed CLS checkpoint; no model selection','candidate_sentence_count':len(candidates),'selected_pairs':len(chosen),'predictions_differ':sum(r['predictions_differ'] for r in results),'both_correct':sum(r['both_correct'] for r in results),'mean_cls_cosine':sum(r['cls_cosine'] for r in results)/len(results),'mean_span_mean_cosine':sum(r['span_mean_cosine'] for r in results)/len(results),'rows':results,'validation_scored':False,'test_scored':False}
    path.write_text(json.dumps(result,indent=2),encoding='utf-8'); print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__': run()
