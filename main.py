from typing import Sequence
from pybondi.publisher import Publisher

from dataclasses import dataclass

@dataclass
class Metric:
    name: str
    value: float

publisher = Publisher()

@publisher.subscriber('metrics')
def on_metrics(metrics: Sequence[Metric]):
    for metric in metrics:
        publisher.publish(metric.name, metric.value)

@publisher.subscriber('loss')
def on_loss(loss: float):
    print(f'Loss: {loss}')

@publisher.subscriber('loss')
def early_stopping_on_loss(loss: float):
    if loss < 0.1:
        raise StopIteration

@publisher.subscriber('accuracy')
def early_stopping(accuracy: float):
    if accuracy > 0.9:
        raise StopIteration

@publisher.subscriber('accuracy')
def on_accuracy(accuracy: float):
    print(f'Accuracy: {accuracy}')

publisher.publish('metrics', [Metric('loss', 0.1), Metric('accuracy', 0.9)])