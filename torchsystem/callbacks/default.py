from typing import Any
from logging import getLogger
from pybondi.callbacks import Callback

logger = getLogger(__name__)

class Default(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.phase = None

    def __call__(self, id: Any, batch: int, loss: float, *args, **kwargs):
        self.batch = batch
        if batch == 100:
            logger.info(f'Epoch: {self.epoch}, Phase: {self.phase}, Batch: {batch}, Loss: {loss} from aggregate with ID {id}')

    def flush(self):
        logger.info(f'End of epoch {self.epoch}, Phase: {self.phase}')
        
    def reset(self):
        pass