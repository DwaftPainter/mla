import copy

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.dataset import PointDataset, make_loader
from src.data.preprocessing import prepare_data, transform
from src.models.loss import create_loss
from src.models.model import Model


@pytest.fixture
def config(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg['paths'] = {key: str(tmp_path / key) for key in cfg['paths']}
    cfg['paths']['checkpoint'] = str(tmp_path / 'checkpoints' / 'best.pt')
    cfg['training']['epochs'] = 3
    return cfg


def test_preprocessing_splits_and_training_only_statistics(config):
    metadata = prepare_data(config)
    datasets = [PointDataset(config, split) for split in ('train', 'validation', 'test')]
    assert [len(ds) for ds in datasets] == [420, 90, 90]
    ids = [set(ds.ids.tolist()) for ds in datasets]
    assert len(set.union(*ids)) == 600
    assert all(not ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    for ds in datasets:
        assert torch.bincount(ds.targets).tolist() == [len(ds) // 2] * 2
    train = datasets[0]
    torch.testing.assert_close(train.features.mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    raw = np.loadtxt(config['paths']['raw'], delimiter=',', skiprows=1)
    np.testing.assert_allclose(metadata['mean'], raw[train.ids.numpy(), :2].mean(0))
    x, y = next(iter(make_loader(config, 'train')))
    assert x.shape == (32, 2) and x.dtype == torch.float32
    assert y.shape == (32,) and y.dtype == torch.long
    first = train.features.clone()
    prepare_data(config)
    torch.testing.assert_close(first, PointDataset(config, 'train').features)


def test_model_forward_loss_and_weight_update():
    torch.manual_seed(1)
    model = Model(input_dim=2, hidden_dim=16, num_classes=2)
    x = torch.randn(8, 2)
    labels = torch.tensor([0, 1] * 4)
    logits = model(x)
    assert logits.shape == (8, 2)
    assert torch.isfinite(logits).all()
    loss = create_loss()(logits, labels)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


@pytest.mark.parametrize('values', [[1.0], [float('nan'), 0.0], [float('inf'), 0.0]])
def test_invalid_inference_features(values):
    with pytest.raises(ValueError):
        transform(values, {'mean': [0., 0.], 'std': [1., 1.]})
