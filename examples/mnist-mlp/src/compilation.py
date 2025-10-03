from torch import load 
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Depends
from torchsystem.compiler import Compiler, compile
from src.classifier import Classifier
from src.training import device, provider
import os

compiler = Compiler[Classifier](provider=provider)

@compiler.step
def build_model(nn: Module, criterion: Module, optimizer: Optimizer, device: str = Depends(device)):
    if device != 'cpu':
        print(f"Moving classifier to device {device}...")
        return Classifier(nn, criterion, optimizer).to(device)
    else:
        return Classifier(nn, criterion, optimizer)

@compiler.step
def restore_weights(classifier: Classifier, device: str = Depends(device)): 
    path = f"data/weights/{classifier.name}-{classifier.hash}.pth"
    if os.path.exists(path):
        (f"Restoring model weights from: {path}")   
        checkpoint = load(path, map_location=device)
        classifier.epoch = checkpoint['epoch']
        classifier.nn.load_state_dict(checkpoint['nn'])
        classifier.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print(f"No weights found at {path}, skipping restore")
    return classifier

@compiler.step
def compile_model(classifier: Classifier, device: str = Depends(device)):
    if device != 'cpu':
        print("Compiling model...") 
        return compile(classifier) 
    else:
        return classifier

@compiler.step
def debug_model(classifier: Classifier):
    print(
        f"Compiled model with:\n"
        f"Name:  {classifier.name}\n"
        f"Hash:  {classifier.hash}\n"
        f"Epochs: {classifier.epoch}"
    )
    print(classifier)
    return classifier