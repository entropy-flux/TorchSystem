from logging import getLogger
from torchsystem.services import Consumer
 
from tinysys.services.training import (
    Trained,
    Validated,
    Iterated,
)

from logging import getLogger
from torchsystem.services import Consumer 

logger = getLogger(__name__)
consumer = Consumer() 

@consumer.handler
def on_trained_log(event: Trained):
    logger.info(f'------------------End of training phase-----------------')
    logger.info(f'--- Average loss: {event.results['loss'].item():.4f}')
    logger.info(f'--- Average accuracy: {100 * event.results['accuracy'].item():.2f}%')
    logger.info(f'--------------------------------------------------------')

@consumer.handler
def on_validated_log(event: Validated):
    logger.info(f'-----------------End of validation phase----------------')
    logger.info(f'--- Average loss: {event.results['loss'].item():.4f}')
    logger.info(f'--- Average accuracy: {100 * event.results['accuracy'].item():.2f}%')
    logger.info(f'--------------------------------------------------------')

@consumer.handler
def on_iterated_log(event: Iterated):
    logger.info(f':::::::::::::::::::: End of epoch {event.model.epoch} ::::::::::::::::::::')