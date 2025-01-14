from logging import getLogger
from torchsystem import Subscriber
from torchsystem import Consumer
from torchsystem.schemas import Metric
from torchsystem.schemas import Message
from torchsystem.storage import get_metadata
from examples._2_with_events.aggregate import Classifier
from examples._2_with_events.events import Stored, Restored

logger = getLogger(__name__)
subscriber = Subscriber()
consumer = Consumer()
db_epochs = {}

@subscriber.subscribe('metrics')
def handle_metrics_log(metrics: tuple[int, list]):
    logger.info(f'.:Batch: {metrics[0]} ' + ', '.join([f'{name}: {value:.3f}' for name, value in metrics[1]]))

@subscriber.subscribe('results')
def handle_result_log(result: Message[Metric]):
    logger.info(f'.:Result: {result.payload.name}: {result.payload.value:.3f} phase {result.payload.phase} at epoch {result.payload.epoch}')
    if result.payload.name == 'accuracy' and result.payload.value > 0.9:
        logger.info('Accuracy is greater than 0.9')
        logger.info('Early stopping the training')
        raise StopIteration
    
@consumer.handler
def handle_stored(event: Stored[Classifier]):
    logger.info("Storing the model")
    logger.info(get_metadata(event.aggregate.model))

@consumer.handler
def handle_restored(event: Restored[Classifier]):
    logger.info(f"Restoring the model at epoch {db_epochs[event.aggregate.id]}")
    event.aggregate.epoch = db_epochs[event.aggregate.id]