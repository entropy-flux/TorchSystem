from torch import Tensor
from torchsystem.metrics import Callbacks
from torchsystem.metrics.average import Cumulative
from torchsystem.metrics.average import Loss, Accuracy

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

def test_callbacks():
    callbacks = Callbacks(Loss(), Accuracy())
    loss, accuracy = callbacks(loss=0.5, output=Tensor([[0.1, 0.9], [0.2, 0.8]]), target=Tensor([1, 1]))
    assert loss.name == 'loss'
    assert accuracy.name == 'accuracy'