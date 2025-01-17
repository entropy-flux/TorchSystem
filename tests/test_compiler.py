from torchsystem import Depends
from torchsystem.compiler import Compiler

compiler = Compiler()

def device():
    return 'cuda'

def epoch():
    return 100

@compiler.step
def build_aggregate(model, criterion, optimizer, device = Depends(device)):
    assert device == 'cuda'
    return (model, criterion, optimizer)

@compiler.step
def retrieve_epoch(aggregate, epoch=Depends(epoch)):
    return (aggregate, epoch)

def test_compiler():
    model = 'model'
    criterion = 'criterion'
    optimizer = 'optimizer'
    (model, criterion, optimizer), epoch = compiler(model, criterion, optimizer)
    assert model == 'model'
    assert criterion == 'criterion'
    assert optimizer == 'optimizer'
    assert epoch == 100 