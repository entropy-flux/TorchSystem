from typing import Sequence
from torchsystem import Subscriber
from torchsystem.metrics import Metric

subscriber = Subscriber()

#NOTE: This is complicated on purpose to show you the flexibility of the subscriber
# You can publish messages not only from the aggregate but also from the handlers.
@subscriber.subscribe('metrics')
def handle_metrics(metric: Metric):
    # This handler will receive all metrics and invoke the handler below¿
    subscriber.receive(topic=metric.name, message=metric)

@subscriber.subscribe('accuracy', 'loss')
def handle_metric_by_name(metric: Metric):
    if metric.batch % 300 == 0:
        print(f'Batch: {metric.batch}, Metric: {metric.name}, Value: {metric.value}, Phase: {metric.phase}')

@subscriber.subscribe('results') # NOTE: You can do better than this. This is just an example
def handle_results(results: dict):
    print("Results", results)

@subscriber.subscribe('end')
def handle_end(message):
    print(message)