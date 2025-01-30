from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Sequence
from torchsystem.services import Subscriber
from torchsystem.services import Consumer
from torchsystem.services import Publisher


class Event:...

@dataclass
class ModelTrained(Event):
    model: Callable
    metrics: Sequence

@dataclass
class ModelEvaluated(Event):
    model: Callable
    metrics: Sequence

consumer = Consumer()
publisher = Publisher()

@consumer.handler
def on_model_iterated(event: ModelTrained | ModelEvaluated):
    for metric in event.metrics:
        publisher.publish(metric, metric['name'])

def model():...


subscriber = Subscriber()

store = []

@subscriber.subscribe('loss', 'accuracy')
def store_metric(metric):
    store.append(metric)


publisher.register(subscriber)
consumer.consume(ModelTrained(model, [{'name': 'loss', 'value': 0.1}, {'name': 'accuracy', 'value': 0.9}]))
consumer.consume(ModelEvaluated(model, [{'name': 'loss', 'value': 0.1}, {'name': 'accuracy', 'value': 0.9}]))

assert store == [
    {'name': 'loss', 'value': 0.1}, 
    {'name': 'accuracy', 'value': 0.9},
    {'name': 'loss', 'value': 0.1}, 
    {'name': 'accuracy', 'value': 0.9}
]