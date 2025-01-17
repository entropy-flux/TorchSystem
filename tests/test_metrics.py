from pytest import approx
from pytest import raises
from torch import Tensor
from torchsystem.metrics.average import Cumulative
from torchsystem.metrics.average import Loss, Accuracy
from torchsystem.metrics.callback import Metrics
from torchsystem.metrics.callback import Callback

def test_cumulative():
    cumulative = Cumulative()
    cumulative.update(1)
    assert cumulative.value == 1
    cumulative.update(2)
    assert cumulative.value == 1.5
    cumulative.update(3)
    assert cumulative.value == 2
    cumulative.update(3, 3)
    assert cumulative.value == 1.5

def test_loss():
    loss = Loss()
    metric = loss(loss=0.1)
    assert metric.value == 0.1
    assert metric.name == 'loss'
    metric = loss(loss=0.2)
    assert metric.value == approx(0.15)
    assert metric.name == 'loss'

def test_metrics():
    metrics = Metrics(Loss())
    metrics(loss=0.1)
    assert metrics['loss'] == approx(0.1)
    metrics(loss=0.2)
    assert metrics['loss'] == approx(0.15)

def test_callback():
    callback = Callback(Loss())
    callback(loss=0.1)
    assert callback.metrics['loss'] == approx(0.1)

    @callback.handler
    def assert_other(loss: float, other: float):
        assert other == 0.1
        callback.metrics(loss=loss)

    callback(loss=0.2, other=0.1)
    
    with raises(Exception):
        callback(loss=0.2, other=0.2)