from logging import getLogger

from torch import Tensor
from torcheval.metrics import Mean, MulticlassAccuracy

logger = getLogger(__name__)

class Metrics:
    def __init__(self, device: str):
        self.loss = Mean(device=device)
        self.accuracy = MulticlassAccuracy(num_classes=10, device=device)

    def update(self, batch: int, loss: Tensor, predictions: Tensor, targets: Tensor) -> None:
        self.loss.update(loss)
        self.accuracy.update(predictions, targets)
        if batch % 100 == 0:
            logger.info(f"--- Batch {batch}: loss={loss.item()}")
        
    def compute(self) -> dict[str, Tensor]:
        return {
            'loss': self.loss.compute(),
            'accuracy': self.accuracy.compute()
        }
    
    def reset(self) -> None:
        self.loss.reset()
        self.accuracy.reset()