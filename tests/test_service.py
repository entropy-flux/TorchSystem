from unittest.mock import Mock
from typing import Any
from torchsystem import Depends
from torchsystem.services import Service

service = Service()

def getdevice() -> str:
    raise NotImplementedError('Override this function to return the device')

@service.handler
def train(model: Any, device: str = Depends(getdevice)):
    model()
    device()

def test_service():
    model = Mock()
    device = Mock()
    service.dependency_overrides[getdevice] = lambda: device
    service.handle('train', model)
    model.assert_called_once()
    device.assert_called_once()