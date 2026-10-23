import pytest
import torch
pytest.importorskip("transformers", reason="Install requirements-bert.txt for BERT tests")
from transformers import BertConfig, BertForSequenceClassification
from src.models.bert_span_readout import BertReadout,target_mask

def test_mask_excludes_marker_padding_and_context():
    ids=torch.tensor([[1,5,10,6,7,11,8,2,0]])
    assert target_mask(ids,ids.ne(0),10,11).tolist()==[[False,False,False,True,True,False,False,False,False]]

def test_empty_falls_back_to_start_marker():
    ids=torch.tensor([[1,10,11,2,0]])
    assert target_mask(ids,ids.ne(0),10,11).tolist()==[[False,True,False,False,False]]

@pytest.mark.parametrize('ids',[[1,10,2],[1,11,10,2],[1,10,10,11]])
def test_invalid_markers_rejected(ids):
    t=torch.tensor([ids])
    with pytest.raises(ValueError): target_mask(t,torch.ones_like(t),10,11)

def test_cls_exactly_matches_original_and_span_has_direct_gradient():
    torch.manual_seed(11)
    base=BertForSequenceClassification(BertConfig(vocab_size=30,hidden_size=12,num_hidden_layers=2,num_attention_heads=2,intermediate_size=24,num_labels=5))
    base.eval()
    batch={'input_ids':torch.tensor([[1,10,6,7,11,2]]),'attention_mask':torch.ones(1,6,dtype=torch.long),'labels':torch.tensor([2])}
    original=base(**batch)
    cls=BertReadout(base,'cls',10,11).eval()
    assert torch.equal(original.logits,cls(**batch).logits)
    assert torch.equal(original.loss,cls(**batch).loss)
    span=BertReadout(base,'span_mean',10,11).eval()
    out=span(**batch,output_hidden_states=True); out.hidden_states[-1].retain_grad(); out.loss.backward()
    grad=out.hidden_states[-1].grad
    assert grad[0,2:4].abs().sum()>0
    assert grad[0,0].abs().sum()==0 and grad[0,1].abs().sum()==0

def test_padding_does_not_affect_span_logits():
    torch.manual_seed(11)
    base=BertForSequenceClassification(BertConfig(vocab_size=30,hidden_size=12,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,num_labels=5))
    model=BertReadout(base,'span_mean',10,11).eval()
    ids=torch.tensor([[1,10,6,11,2]])
    padded=torch.cat([ids,torch.zeros(1,3,dtype=torch.long)],1)
    with torch.no_grad():
        a=model(ids,ids.ne(0)).logits; b=model(padded,padded.ne(0)).logits
    assert torch.allclose(a,b,atol=1e-7)
