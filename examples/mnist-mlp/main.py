from torch import cuda

def device() -> str:
    return 'cuda' if cuda.is_available() else 'cpu'

if __name__ == '__main__':
    from src import training, checkpoints, logs
    from src.compilation import compiler

    from model import MLP
    from dataset import Digits
 
    from torch.nn import CrossEntropyLoss
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from torchsystem import registry

    registry.register(MLP)
    training.provider.override(training.device, device) 
    training.producer.register(checkpoints.consumer)
    checkpoints.publisher.register(logs.subscriber)

    nn = MLP(input_size=784, hidden_size=256, output_size=10, dropout=0.5)
    criterion = CrossEntropyLoss()
    optimizer = SGD(nn.parameters(), lr=0.001)
    classifier = compiler.compile(nn, criterion, optimizer)

    datasets = {
        'train': Digits(train=True, normalize=True),
        'evaluation': Digits(train=False,  normalize=True),
    }

    loaders = {
        'train': DataLoader(datasets['train'], batch_size=256, shuffle=True, pin_memory=True, pin_memory_device='cuda', num_workers=4),
        'evaluation': DataLoader(datasets['evaluation'], batch_size=256, shuffle=False, pin_memory=True, pin_memory_device='cuda', num_workers=4) 
    } if cuda.is_available() else {
        'train': DataLoader(datasets['train'], batch_size=256, shuffle=True),
        'evaluation': DataLoader(datasets['evaluation'], batch_size=256, shuffle=False) 
    }

    for epoch in range(5):
        training.train(classifier, loaders['train'])
        training.evaluate(classifier, loaders['evaluation'])