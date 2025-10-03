from torch import cuda
from torch.nn import CrossEntropyLoss 
from torch.optim import Adam
from torch.utils.data import DataLoader 
from torchsystem.registry import register, getmetadata
from mltracker import getExperiment 
from dataset import Images 
from src import training 
from src import persistence 
from src.metrics import Metrics
from src.compilation import compiler  
from model import ViT
from logging import basicConfig, INFO 

import torch
import random 

if __name__ == '__main__':  
    seed = 42
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    basicConfig(level=INFO)
    register(Adam, excluded_args=[0], excluded_kwargs={'params'})
    register(CrossEntropyLoss)
    register(Images) 
    register(ViT) 

    experiment = getExperiment("CIFAR-Classification")

    training.provider.override(training.device, lambda: 'cuda' if cuda.is_available() else 'cpu') 
    training.provider.override(training.models, lambda: experiment.models)
    training.provider.override(training.location, lambda: experiment.name)
    training.producer.register(persistence.consumer)

    for number_of_layers in [3, 6, 9]:
        nn = ViT(
            patch_size=4,
            input_channels=3,
            image_size=32,
            model_dimension=128,
            ffn_hidden_dimension=256,
            number_of_heads=4,
            number_of_layers=number_of_layers,
            number_of_classes=100
        ) 
        
        criterion = CrossEntropyLoss()
        optimizer = Adam(nn.parameters(), lr=0.001)
        metrics = Metrics(number_of_classes=100)
        classifier = compiler.compile(nn, criterion, optimizer, metrics)
        datasets = {
            'train': Images(train=True, normalize=True),
            'evaluation': Images(train=False, normalize=True),
        }

        batch_size = 256

        loaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, pin_memory=True, pin_memory_device='cuda', num_workers=4),
            'evaluation': DataLoader(datasets['evaluation'], batch_size=batch_size, shuffle=False, pin_memory=True, pin_memory_device='cuda', num_workers=4) 
        } if cuda.is_available() else {
            'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
            'evaluation': DataLoader(datasets['evaluation'], batch_size=batch_size, shuffle=False) 
        }

        for epoch in range(250):
            training.train(classifier, loaders['train'])
            training.evaluate(classifier, loaders['evaluation'])

        model = experiment.models.read(classifier.hash)
        iteration = model.iterations.create(model.epoch)
        iteration.modules.add("optimizer", getmetadata(optimizer))
        iteration.modules.add("data", {"dataset": "CIFAR10" , "batch_size": batch_size})          