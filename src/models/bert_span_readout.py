"""Readout diagnostics sharing exactly the same BERT encoder/pooler/head."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

def target_mask(input_ids, attention_mask, start_id, end_id):
    """Interior span tokens; insertion falls back to the start-marker state."""
    start=input_ids.eq(start_id) & attention_mask.bool()
    end=input_ids.eq(end_id) & attention_mask.bool()
    if not torch.all(start.sum(1)==1) or not torch.all(end.sum(1)==1):
        raise ValueError('Expected exactly one valid start/end marker per sequence')
    left=start.to(torch.long).argmax(1); right=end.to(torch.long).argmax(1)
    if not torch.all(left<right): raise ValueError('Reversed span markers')
    positions=torch.arange(input_ids.shape[1],device=input_ids.device)[None,:]
    mask=(positions>left[:,None]) & (positions<right[:,None]) & attention_mask.bool()
    return torch.where(mask.any(1)[:,None],mask,start)

class BertReadout(nn.Module):
    def __init__(self,base,mode,start_id,end_id):
        super().__init__()
        if mode not in ('cls','span_mean'): raise ValueError(mode)
        self.base=base; self.mode=mode; self.start_id=start_id; self.end_id=end_id

    def forward(self,input_ids,attention_mask,token_type_ids=None,labels=None,output_hidden_states=False):
        out=self.base.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,
                           output_hidden_states=output_hidden_states,return_dict=True)
        if self.mode=='cls': pooled=out.pooler_output
        else:
            mask=target_mask(input_ids,attention_mask,self.start_id,self.end_id)
            mean=(out.last_hidden_state*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            pooled=self.base.bert.pooler.activation(self.base.bert.pooler.dense(mean))
        logits=self.base.classifier(self.base.dropout(pooled))
        return SequenceClassifierOutput(loss=F.cross_entropy(logits,labels) if labels is not None else None,
                                        logits=logits,hidden_states=out.hidden_states)
