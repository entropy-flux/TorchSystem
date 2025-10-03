from torch import Tensor
from torcheval.metrics import Mean, MulticlassAccuracy
 
class Metrics:
    def __init__(self, number_of_classes: int, device: str | None = None):
        self.loss = Mean(device=device)
        self.accuracy = MulticlassAccuracy(num_classes=number_of_classes, device=device) 
        
    def update(self, batch: int, loss: Tensor, predictions: Tensor, targets: Tensor) -> None:
        self.loss.update(loss)
        self.accuracy.update(predictions, targets) 
        
    def compute(self) -> dict[str, Tensor]:
        return {
            'loss': self.loss.compute(),
            'accuracy': self.accuracy.compute()
        }
    
    def reset(self) -> None:
        self.loss.reset()
        self.accuracy.reset()