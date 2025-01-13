from typing import Any
from typing import Literal
from collections import deque
from pybondi.publisher import Publisher
from pybondi.events import Events, Event
from torch.nn import Module
from torchsystem.schemas import Message

class Aggregate(Module):
    events: Events
    publisher: Publisher
    
    def __init__(self, id: Any):
        super().__init__()
        self.id = id
        
    @property
    def phase(self) -> Literal['train', 'evaluation']:
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: Literal['train', 'evaluation']):
        self.train() if value == 'train' else self.eval()

    def publish(self, topic: str, message: Any):
        self.publisher.publish(topic, message)

    def deliver(self, data: Any, topic: str):
        self.publisher.publish(topic, Message(sender=self.id, payload=data, topic=topic))

    def emit(self, event: Event):
        self.events.consume(event)