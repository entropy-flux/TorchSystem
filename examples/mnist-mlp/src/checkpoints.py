from os import makedirs
from torch import save
from torchsystem import Depends
from torchsystem.services import Consumer
from src.training import provider, models
from src.training import Trained, Evaluated 

consumer = Consumer(provider=provider)

@consumer.handler
def bump_epoch(event: Trained):
    event.model.epoch += 1 

@consumer.handler
def log_metrics(event: Trained | Evaluated):
    print("-----------------------------------------------------------------")
    print(
        f"Epoch: {event.model.epoch}, "
        f"Average loss: {event.results['loss'].item()}, "
    )
    print("-----------------------------------------------------------------")

@consumer.handler
def persist_model(event: Trained,):
    makedirs(f"data/weights", exist_ok=True)
    path = f"data/weights/{event.model.name}-{event.model.hash}.pth"
    checkpoint = {
        'epoch': event.model.epoch,
        'nn': event.model.nn.state_dict(),
        'optimizer': event.model.optimizer.state_dict()
    }
    save(checkpoint, path) 
    print(f"Saved model weights at: {path}")