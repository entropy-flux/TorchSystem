from torch.utils.tensorboard import SummaryWriter

from torchsystem import Depends
from torchsystem.depends import Provider
from torchsystem.services import Consumer

from tinysys.services.training import (
    Trained,
    Validated
)

provider = Provider()
consumer = Consumer(provider=provider)

def writer() -> SummaryWriter:...

@consumer.handler
def handle_metrics(event: Trained | Validated, writer: SummaryWriter = Depends(writer)):
    for key, value in event.results.items():
        writer.add_scalars(f"{event.model.id}/{key}", {event.model.phase: value}, event.model.epoch)