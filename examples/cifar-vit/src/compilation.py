from os import makedirs
from torch import load 
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Depends
from torchsystem.compiler import Compiler, compile
from src.metrics import Metrics
from src.classifier import Classifier
from src.training import models, device, provider, location
from mltracker.ports import Models
from logging import getLogger
import os

compiler = Compiler[Classifier](provider=provider)
logger = getLogger(__name__)
    
@compiler.step
def build_model(nn: Module, criterion: Module, optimizer: Module, metrics: Metrics, device: str = Depends(device)):
    if device != 'cpu':
        logger.info(f"Moving classifier to device {device}...")
        metrics.accuracy.to(device)
        metrics.loss.to(device)
        return Classifier(nn, criterion, optimizer, metrics).to(device)
    else:
        return Classifier(nn, criterion, optimizer, metrics)

@compiler.step
def bring_to_current_epoch(classifier: Classifier, models: Models = Depends(models)):
    logger.info("Retrieving model from store...")
    model = models.read(classifier.id)
    if not model:
        logger.info("Model not found, creating one...")
        model = models.create(classifier.id, 'classifier')
    else:
        logger.info(f"Model found on epoch {model.epoch}")
    classifier.epoch = model.epoch
    return classifier 

@compiler.step
def restore_weights(classifier: Classifier, location: str = Depends(location), device: str = Depends(device)): 
    if classifier.epoch != 0:    
        path = f"data/weights/{location}/{classifier.name}-{classifier.hash}.pth"
        if os.path.exists(path):
            logger.info(f"Restoring model weights from: {path}")   
            checkpoint = load(path, map_location=device)
            classifier.nn.load_state_dict(checkpoint['nn'])
            classifier.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            logger.warning(f"No weights found at {path}, skipping restore")
    return classifier

@compiler.step
def compile_model(classifier: Classifier, device: str = Depends(device)):
    if device != 'cpu':
        logger.info("Compiling model...")
        return compile(classifier) 
    else:
        return classifier

@compiler.step
def debug_model(classifier: Classifier):
    logger.info(
        f"Compiled model with:\n"
        f"Name:  {classifier.name}\n"
        f"Hash:  {classifier.hash}\n"
        f"Epochs: {classifier.epoch}"
    )
    logger.info(classifier)
    return classifier