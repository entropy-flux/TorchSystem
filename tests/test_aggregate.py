from unittest.mock import Mock
from torchsystem.domain import Aggregate

mock = Mock()

class Model(Aggregate):
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def onepoch(self):
        mock()

def test_onepoch():
    model = Model()
    model.epoch = 5
    assert mock.call_count == 1
    assert model.epoch == 5