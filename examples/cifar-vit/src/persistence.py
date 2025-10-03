from os import makedirs
from mltracker.ports import Models
from torch import save
from torchsystem import Depends
from torchsystem.services import Consumer
from src.training import provider, models, location
from src.training import Trained, Evaluated 
from logging import getLogger

consumer = Consumer(provider=provider)
logger = getLogger(__name__)

@consumer.handler
def bump_epoch(event: Trained):
    event.model.epoch += 1 

@consumer.handler
def save_epoch(event: Trained, models: Models = Depends(models)):
    model = models.read(event.model.hash) or models.create(event.model.hash) 
    model.epoch += 1

@consumer.handler
def log_metrics(event: Trained | Evaluated):
    logger.info("-----------------------------------------------------------------")
    logger.info(
        f"Epoch: {event.model.epoch}, "
        f"Average loss: {event.results['loss'].item()}, "
        f"Average accuracy: {event.results['accuracy'].item()}"
    )
    logger.info("-----------------------------------------------------------------")

@consumer.handler
def handle_results(event: Trained | Evaluated, models: Models = Depends(models)):
    model = models.read(event.model.id)
    for name, metric in event.results.items():
        model.metrics.add(name, metric.item(), event.model.epoch, event.model.phase) 
 
@consumer.handler
def persist_model(event: Trained, location: str = Depends(location)):
    makedirs(f"data/weights/{location}", exist_ok=True)
    path = f"data/weights/{location}/{event.model.name}-{event.model.hash}.pth"
    checkpoint = {
        'nn': event.model.nn.state_dict(),
        'optimizer': event.model.optimizer.state_dict()
    }
    save(checkpoint, path) 
    logger.info(f"Saved model weights at: {path}")