import copy

import pytest
import torch

from test_model import config
from src.checkpoint import load_checkpoint
from src.data.dataset import make_loader
from src.evaluation.evaluate import evaluate
from src.evaluation.metrics import score
from src.inference.predict import predict
from src.models.loss import create_loss
from src.training.train import train


def test_training_checkpoint_roundtrip_and_inference(config):
    checkpoint_path, history = train(config)
    model, checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint['epoch'] == min(history, key=lambda row: row['validation_loss'])['epoch']
    assert checkpoint['validation_metric'] == min(row['validation_loss'] for row in history)
    assert checkpoint['optimizer_state_dict']['state']
    assert checkpoint['configuration'] == config
    assert not model.training
    before = copy.deepcopy(model.state_dict())
    metrics = score(model, make_loader(config, 'validation'), create_loss(), 'cpu', 2)
    for name, weights in model.state_dict().items():
        torch.testing.assert_close(weights, before[name])
    assert metrics['loss'] == pytest.approx(checkpoint['validation_metric'])
    result = predict(checkpoint_path, [0.7, 0.0])
    assert result['class_id'] in (0, 1)
    assert sum(result['probabilities']) == pytest.approx(1.0)
    assert result['confidence'] == max(result['probabilities'])
    test_metrics = evaluate(checkpoint_path)
    assert test_metrics['samples'] == 90
    assert 0 <= test_metrics['accuracy'] <= 1
    # Evaluation must reject prepared test data from a different partition.
    test_path = checkpoint_path.parent.parent / 'split' / 'test' / 'data.pt'
    payload = torch.load(test_path, weights_only=True)
    payload['ids'][0] = -1
    torch.save(payload, test_path)
    with pytest.raises(ValueError, match='test split'):
        evaluate(checkpoint_path)


def test_metrics_against_known_predictions():
    logits = torch.tensor([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
    labels = torch.tensor([0, 1, 1, 1])
    result = score(torch.nn.Identity(), [(logits, labels)], create_loss(), 'cpu', 2)
    assert result['accuracy'] == 0.75
    assert result['precision_macro'] == pytest.approx(0.75)
    assert result['recall_macro'] == pytest.approx((1 + 2 / 3) / 2)
    assert result['f1_macro'] == pytest.approx((2 / 3 + 0.8) / 2)
