import pytest
pytest.importorskip("transformers", reason="Install requirements-bert.txt for BERT tests")
from transformers import BertTokenizerFast
from src.training.bert_span_classifier import encode_span

@pytest.fixture(scope='module')
def tok(tmp_path_factory):
    # Synthetic WordPiece vocabulary: no downloaded model or private run artifacts.
    vocab=tmp_path_factory.mktemp('tokenizer')/'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',
                                'left','right','running','quickly','word','hello',
                                'they','goes','home','.'])+'\n',encoding='utf-8')
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=True)
    tokenizer.add_special_tokens({'additional_special_tokens':['[SPAN_START]','[SPAN_END]']})
    return tokenizer

def test_crop_keeps_entire_span(tok):
    text='left '*150+'running quickly'+' right'*150; start=text.index('running'); end=start+len('running quickly')
    encoded,stats=encode_span(dict(sentence=text,error_start=start,error_end=end,error_span=text[start:end]),tok,32)
    ids=encoded['input_ids']; a=ids.index(tok.convert_tokens_to_ids('[SPAN_START]')); b=ids.index(tok.convert_tokens_to_ids('[SPAN_END]'))
    assert ids[a+1:b]==tok.encode('running quickly',add_special_tokens=False)
    assert len(ids)==32 and stats['truncated']

@pytest.mark.parametrize('start,end',[(2,4),(4,4)])
def test_partial_and_empty_span(tok,start,end):
    text='running'
    ids,_=encode_span(dict(sentence=text,error_start=start,error_end=end,error_span=text[start:end]),tok)
    assert len(ids['input_ids'])<=128

def test_bad_offset_rejected(tok):
    with pytest.raises(ValueError): encode_span(dict(sentence='hello',error_start=0,error_end=4,error_span='hello'),tok)

def test_oversized_span_rejected(tok):
    text='word '*50
    with pytest.raises(ValueError): encode_span(dict(sentence=text,error_start=0,error_end=len(text),error_span=text),tok,16)

def test_labels_and_corrections_do_not_change_features(tok):
    row=dict(sentence='They goes home.',error_start=5,error_end=9,error_span='goes')
    baseline=encode_span(row,tok)
    assert encode_span(dict(row,error_type='SVA',correction='go',sample_id='a'),tok)==baseline
    assert encode_span(dict(row,error_type='Vform',correction='went',sample_id='b'),tok)==baseline
