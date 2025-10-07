from logging import getLogger 
from torchsystem.services import Subscriber
from src.checkpoints import Metric, provider

subscriber = Subscriber(provider=provider)
logger = getLogger(__name__)

@subscriber.subscribe('metrics')
def log_metrics(metric: Metric):  
    logger.info(f"Average {metric.name}: {metric.value}")  
