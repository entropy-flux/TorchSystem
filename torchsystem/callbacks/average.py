from torch import Tensor
from pybondi.callbacks import Callback
from torchsystem.callbacks.metrics import Metric, accuracy, predictions

class Average:
    def __init__(self):
        self.value = 0.0

    def update(self, sample: int, value: float) -> float:
        self.value = (self.value * (sample - 1) + value) / sample
        return self.value

    def reset(self):
        self.value = 0.0

class Loss(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.phase = None
        self.average = Average()

    def __call__(self, batch: int, loss: float, *args, **kwargs):
        self.batch = batch
        self.average.update(batch, loss)        

    def flush(self):
        self.publisher.publish('result', Metric('loss', self.average.value, self.batch, self.epoch, self.phase))
        self.average.reset()
        
    def reset(self):
        self.average.reset()


class Accuracy(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.phase = None
        self.average = Average()
        
    def __call__(self, batch: int, _, output: Tensor, target: Tensor, *args, **kwargs):
        self.batch = batch
        self.average.update(batch, accuracy(predictions(output), target))

    def flush(self):
        self.publisher.publish('result', Metric('accuracy', self.average.value, self.batch, self.epoch, self.phase))
        self.average.reset()

    def reset(self):
        self.average.reset()