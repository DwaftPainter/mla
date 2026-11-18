"""Deterministic post-prediction audit packet; source text stays local."""
from pathlib import Path
from collections import defaultdict
import json,random,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.nucle_final_evaluation import load_predictions,verify_lock
from src.training.nucle_final_selection import OUT,PREP,read,dump,sha,now,LABELS
from src.features.nucle_pos import load_tagger,align_span
from src.features.nucle_features import extract_features
from src.data.nucle_sgml import load_nucle
from src.data.parse_nucle import documents_to_dataframe
from src.data.constants import SGML_FILE

def choose(meta,pred):
    rng=random.Random(20261023); buckets=defaultdict(list)
    for i,r in enumerate(meta):
        if pred['LR_03'][i]!=r['error_type']:buckets[(r['error_type'],pred['LR_03'][i])].append(i)
    for v in buckets.values():rng.shuffle(v)
    selected=[]
    while len(selected)<25 and any(buckets.values()):
        for k in sorted(buckets):
            if buckets[k] and len(selected)<25:selected.append(buckets[k].pop())
    tree_only=[i for i,r in enumerate(meta) if pred['LR_03'][i]==r['error_type'] and pred['Tree_03'][i]!=r['error_type']]
    rng.shuffle(tree_only); selected+=tree_only[:5]
    correct=[]
    for label in LABELS:
        pool=[i for i,r in enumerate(meta) if r['error_type']==label==pred['LR_03'][i]];rng.shuffle(pool);correct+=pool[:2]
    # IDs may overlap Tree-only errors and LR-correct controls; keep records unique.
    correct=[i for i in correct if i not in selected]
    while len(correct)<10:
        pool=[i for i,r in enumerate(meta) if r['error_type']==pred['LR_03'][i] and i not in selected+correct]
        if not pool:break
        correct.append(rng.choice(pool))
    return selected,correct

def run():
    verify_lock();meta,pred=load_predictions();errors,correct=choose(meta,pred);chosen=errors+correct
    selected=[{'sample_id':meta[i]['sample_id'],'annotation_id':meta[i]['annotation_id'],'kind':'error' if i in errors else 'LR_correct_control','gold':meta[i]['error_type'],'LR':pred['LR_03'][i],'Tree':pred['Tree_03'][i],'POS':pred['POS'][i],'span_bin':meta[i]['span_bin'],'pos_signature':meta[i]['pos_signature']} for i in chosen]
    dump(OUT/'audit_selection.json',{'at':now(),'seed':20261023,'before_correction_review':True,'selection':'25 LR errors round-robin gold/pred + 5 LR-correct Tree errors + 10 LR-correct controls','rows':selected})
    samples={};wanted={r['sample_id'] for r in selected}
    for line in (PREP/'samples.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['sample_id'] in wanted:samples[r['sample_id']]=r
    source=documents_to_dataframe(load_nucle(SGML_FILE)).set_index('annotation_id');nlp=load_tagger();packet=[]
    for item in selected:
        s=samples[item['sample_id']];doc=nlp(s['sentence']);a=align_span(doc,s['error_start'],s['error_end'])
        f=extract_features(s,doc)
        packet.append(item|{'sentence':s['sentence'],'span':s['error_span'],'offsets':[s['error_start'],s['error_end']],'correction':str(source.loc[int(item['annotation_id']),'correction']),'features':{k:v for k,v in f.items() if k.startswith(('D|','M|','P|'))},'tokens':[{'i':t.i,'text':t.text,'pos':t.pos_,'dep':t.dep_,'head':t.head.i,'morph':str(t.morph),'target':t.i in a['span_token_indices']} for t in doc]})
    dump(OUT/'private_error_audit_packet.json',packet);print('PRIVATE AUDIT PACKET',len(packet))
if __name__=='__main__':run()
