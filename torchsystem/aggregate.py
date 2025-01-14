from abc import ABC, abstractmethod
from typing import Any
from typing import Literal
from torch.nn import Module
from pymsgbus import Producer
from pymsgbus import Publisher, Message
from pymsgbus import Subscriber, Consumer

class Aggregate(Module, ABC):
    """
    An AGGREGATE is a cluster of associated objects that we treat as a unit for the purpose
    of data changes. Each AGGREGATE has a root and a boundary. The boundary defines what is
    inside the AGGREGATE. The root is a single, specific ENTITY contained in the AGGREGATE.

    An AGGREGATE is responsible for maintaining the consistency of the data within its boundary
    and enforcing invariants that apply to the AGGREGATE as a whole. It can communicate data
    to the outside world using domain events or messages.
    """
    def __init__(self):
        super().__init__()
        self.publisher = Publisher()
        self.producer = Producer()
        
    @property
    def phase(self) -> Literal['train', 'evaluation']:
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: Literal['train', 'evaluation']):
        self.train() if value == 'train' else self.eval()
    
    def publish(self, message: Message, topic: str):
        """
        Deliver a message to all subscribers of a given topic.

        Args:
            message (Message): The message to deliver.  
            topic (str): The topic to deliver the message to.
        """
        self.publisher.publish(topic=topic, message=message)
    
    @property 
    @abstractmethod
    def id(self) -> Any:
        """
        The id of the AGGREGATE. An AGGREGATE always has a unique identifier.
        """
        ...

    def bind(self, observer: Subscriber | Consumer):
        """
        Bind an observer to the AGGREGATE. The observer will receive messages
        from the AGGREGATE.

        Args:
            observer (Subscriber | Consumer): The observer to bind.
        """
        if isinstance(observer, Subscriber):
            self.publisher.register(observer)
        elif isinstance(observer, Consumer):
            self.producer.register(observer)