from torchsystem import Depends
from torchsystem.services import Subscriber
from torchsystem.services import Publisher

subscriber = Subscriber()
metricsdb = []

def metrics():
    return metricsdb

@subscriber.subscribe('loss', 'accuracy')
def store_metric(metric, metrics: list = Depends(metrics)):
    metrics.append(metric)

@subscriber.subscribe('accuracy')
def on_accuracy_to_high(metric):
    if metric > 0.99:
        raise StopIteration
    
publisher = Publisher()
publisher.register(subscriber)

publisher.publish(0.1, 'loss')
publisher.publish(0.9, 'accuracy')
assert metricsdb == [0.1, 0.9]

try:
    publisher.publish(1.0, 'accuracy')
except StopIteration:
    print("Early stopping") 


###

subscriber = Subscriber()

@subscriber.subscribe('metrics')
def store_metric(metrics: list):
    for metric in metrics:
        subscriber.receive(metric, metric['name'])

@subscriber.subscribe('loss')
def on_loss(loss):
    print(f"Loss: {loss}")

@subscriber.subscribe('accuracy')
def on_accuracy(accuracy):
    print(f"Accuracy: {accuracy}")

subscriber.receive([{'name': 'loss', 'value': 0.1}, {'name': 'accuracy', 'value': 0.9}], 'metrics')