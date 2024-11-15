from typing import Any
from torch import Tensor
from pybondi.callbacks import Callback
from pybondi.publisher import Message
from torchsystem.callbacks.metrics import Metric, accuracy, predictions

class Average:
    '''
    Class to calculate the moving average of a series of samples    
    '''
    def __init__(self):
        self.values = {}

    def update(self, id: Any, sample: int, value: float) -> float:
        '''
        Update the average with a new value given a sample index.
        '''
        self.values[id] = (self.values.get(id, 0)  * (sample - 1) + value) / sample
        return self.values[id]

    def reset(self):
        '''
        Reset the average.
        '''
        self.values = {}

class Loss(Callback):
    def __init__(self, topic: str = 'result'):
        super().__init__()
        self.epoch = 0
        self.phase = None
        self.average = Average()
        self.topic = topic

    def __call__(self, id: Any, batch: int, loss: float, *args, **kwargs):
        self.batch = batch
        self.average.update(id, batch, loss)        

    def flush(self):
        for sender, value in self.average.values.items():
            self.publisher.publish(self.topic, Message(str(sender), Metric('loss', value, self.batch, self.epoch, self.phase)))
        self.average.reset()
        
    def reset(self):
        self.average.reset()

class Accuracy(Callback):
    def __init__(self, topic: str = 'result'):
        super().__init__()
        self.epoch = 0
        self.phase = None
        self.average = Average()
        self.topic = topic
        
    def __call__(self, id: Any, batch: int, _, output: Tensor, target: Tensor, *args, **kwargs):
        self.batch = batch
        self.average.update(id, batch, accuracy(predictions(output), target))

    def flush(self):
        for sender, value in self.average.values.items():
            self.publisher.publish(self.topic, Message(str(sender), Metric('loss', value, self.batch, self.epoch, self.phase)))
        self.average.reset()

    def reset(self):
        self.average.reset()