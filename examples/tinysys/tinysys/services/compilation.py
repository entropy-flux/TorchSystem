from logging import getLogger 
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem.depends import Depends
from torchsystem.depends import Provider
from torchsystem.compiler import compile
from torchsystem.compiler import Compiler
from torchsystem.registry import getname
from tinysys.ports.models import Models
from tinysys.domain import Repository
from tinysys.classifier import Classifier

logger = getLogger(__name__)
provider = Provider()
compiler = Compiler[Classifier](provider=provider)

def device() -> str:...

def models() -> Models:...

def repository() -> Repository:...

@compiler.step
def build_classifier(model: Module, criterion: Module, optimizer: Optimizer):
    logger.info(f'Building Classifier')
    logger.info(f'- model: {model.__class__.__name__}')
    logger.info(f'- criterion: {criterion.__class__.__name__}')
    logger.info(f'- optimizer: {optimizer.__class__.__name__}')
    return Classifier(model, criterion, optimizer)

@compiler.step
def move_to_device(classifier: Classifier, device = Depends(device)):
    logger.info(f'Moving classifier to device: {device}')
    return classifier.to(device)

@compiler.step
def compile_classifier(classifier: Classifier):
    logger.info(f'Compiling classifier')
    return compile(classifier)

@compiler.step
def bring_epoch(classifier: Classifier, models: Models = Depends(models)):
    model = models.read(id=classifier.id)
    if not model:
        logger.info(f'Creating model with id {classifier.id} in database')
        models.create(id=classifier.id, name=f'{getname(classifier.nn)}-classifier')

    else:
        logger.info(f'Model found with id {classifier.id} in database')
        if model.epoch > classifier.epoch:
            logger.info(f'Bringing classifier to epoch {model.epoch}')
            classifier.epoch = model.epoch
        elif model.epoch == classifier.epoch:
            logger.info(f'Starting classifier from epoch {classifier.epoch}')
        else:
            raise ValueError(f'Epoch mismatch')
    return classifier

@compiler.step
def restore_weights(classifier: Classifier, repository: Repository = Depends(repository)):
    if classifier.epoch > 0:
        logger.info(f'Restoring weights for classifier')
        repository.restore(classifier)
    return classifier