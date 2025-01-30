from pytest import raises
from torchsystem import Depends
from torchsystem.services import Subscriber
from torchsystem.services import Publisher

subscriber = Subscriber()
metricsdb = []

def metrics():
    return metricsdb

@subscriber.subscribe('metric')
def deliver(metric: dict):
    subscriber.receive(metric['value'], metric['name'])

@subscriber.subscribe('loss', 'accuracy')
def store_metric(metric, metrics: list = Depends(metrics)):
    metrics.append(metric)

@subscriber.subscribe('accuracy')
def on_accuracy_to_high(metric):
    if metric > 0.99:
        raise StopIteration    

def test_publisher():
    publisher = Publisher()
    publisher.register(subscriber)

    publisher.publish(0.1, 'loss')
    publisher.publish(0.9, 'accuracy')
    assert metricsdb == [0.1, 0.9]

    with raises(StopIteration):
        publisher.publish(1.0, 'accuracy')

    with raises(StopIteration):
        publisher.publish({'value': 1.0, 'name': 'accuracy'}, 'metric')