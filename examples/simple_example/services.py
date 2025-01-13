from torchsystem import Service
from torchsystem.metrics import Metric

service = Service()

@service.subscriber('loss', 'accuracy')
def print_metrics(metric: Metric):
    if metric.batch % 100 == 0:
        print(f'Batch: {metric.batch} {metric.name}: {metric.value}')

@service.subscriber('accuracy')
def early_stopping(metric: Metric):
    if metric.value > 0.9:
        raise StopIteration