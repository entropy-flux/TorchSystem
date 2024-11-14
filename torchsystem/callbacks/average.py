from torch import Tensor
from pybondi.callbacks import Callback
from torchsystem.callbacks.metrics import Metric, accuracy, predictions

class Average:
    '''
    Class to calculate the moving average of a series of samples.    
    '''
    def __init__(self):
        self.value = 0.0

    def update(self, sample: int, value: float) -> float:
        '''
        Update the average with a new value given a sample index.
        '''
        self.value = (self.value * (sample - 1) + value) / sample
        return self.value

    def reset(self):
        self.value = 0.0

class Loss(Callback):
    def __init__(self, topic: str = 'result'):
        super().__init__()
        self.epoch = 0
        self.phase = None
        self.average = Average()
        self.topic = topic

    def __call__(self, batch: int, loss: float, *args, **kwargs):
        self.batch = batch
        self.average.update(batch, loss)        

    def flush(self):
        self.publisher.publish(self.topic, Metric('loss', self.average.value, self.batch, self.epoch, self.phase))
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
        
    def __call__(self, batch: int, _, output: Tensor, target: Tensor, *args, **kwargs):
        self.batch = batch
        self.average.update(batch, accuracy(predictions(output), target))

    def flush(self):
        self.publisher.publish(self.topic, Metric('accuracy', self.average.value, self.batch, self.epoch, self.phase))
        self.average.reset()

    def reset(self):
        self.average.reset()