from logging import getLogger
from pybondi import Session
from torchsystem.events import Trained, Evaluated, Iterated
from torchsystem.callbacks.metrics import Metric

#TODO: Move all loggings here

logger = getLogger(__name__)

def log_trained(event: Trained):
    logger.info(f'Trained {event.aggregate.__class__.__name__} loaders in {event.end - event.start} seconds')

def log_evaluated(event: Evaluated):
    logger.info(f'Evaluated {event.aggregate.__class__.__name__} loaders in {event.end - event.start} seconds')

def log_iterated(event: Iterated):
    logger.info(f'Iterated {event.aggregate.__class__.__name__} in {event.end - event.start} seconds')

def log_average_metric(metric: Metric):
    logger.info(f'Average {metric.name}: {metric.value} on metric {metric.batch} batches in epoch {metric.epoch} phase {metric.phase}')

Session.event_handlers.setdefault(Trained, []).append(log_trained)
Session.event_handlers.setdefault(Evaluated, []).append(log_evaluated)
Session.event_handlers.setdefault(Iterated, []).append(log_iterated)