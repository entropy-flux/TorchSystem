from datetime import datetime
from tinysys.ports import structure, unstructure 
from tinysys.ports.experiments import Experiments
from tinysys.ports.models import Models
from tinysys.ports.modules import Module, Modules
from tinysys.ports.metrics import Metric, Metrics
from tinysys.ports.iterations import Iteration, Iterations

def test_experiments(experiments: Experiments):
    experiment = experiments.create(name='experiment1')
    assert experiment.name == 'experiment1'
    experiment = experiments.update(id=experiment.id, name='experiment2')
    assert experiment.name == 'experiment2'
    assert len(experiments.list()) == 1
    experiments.delete(id=experiment.id)
    assert len(experiments.list()) == 0

def test_models(models: Models):
    model = models.create(id='1', name='mlp-classifier')
    assert model.id == '1'
    assert model.name == 'mlp-classifier'
    assert model.epoch == 0

    model = models.update(id='1', epoch=1)
    assert model.id == '1'
    assert model.epoch == 1

    model = models.read(id='1')
    assert model.id == '1'
    assert model.epoch == 1

    assert len(models.list()) == 1
    models.delete(id='1')
    assert len(models.list()) == 0


def test_metrics(metrics: Metrics):
    metrics.add(Metric(name='accuracy', value=0.9, epoch=1, phase='train'))
    metrics.add(Metric(name='accuracy', value=0.2, epoch=1, phase='test'))
    metrics.add(Metric(name='accuracy', value=0.2, epoch=2, phase='train'))
    metrics.add(Metric(name='loss', value=0.3, epoch=3, phase='train'))
    assert len(metrics.list()) == 4
    metrics.clear()
    assert len(metrics.list()) == 0

def test_modules(modules: Modules):
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=1, arguments={'lr': 0.1}))
    modules.put(Module(type='criterion', hash='hash1', name='CrossEntropyLoss', epoch=1, arguments={}))
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=10, arguments={'lr': 0.1}))
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=11, arguments={'lr': 0.1})) #UPDATE
    assert len(modules.list('optimizer')) == 1

    modules.put(Module(type='optimizer', hash='hash2', name='Adam', epoch=12, arguments={'lr': 0.2})) #ADD SINCE LAST HASH IS DIFFERENT
    assert len(modules.list('optimizer')) == 2

    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=13, arguments={'lr': 0.1})) #ADD SINCE LAST HASH IS DIFFERENT
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=14, arguments={'lr': 0.1})) #UPDATE
    assert len(modules.list('optimizer')) == 3
    assert len(modules.list('criterion')) == 1

    print(modules.list('optimizer'))
    print(modules.list('criterion'))

    modules.clear()
    assert len(modules.list('optimizer')) == 0
    assert len(modules.list('criterion')) == 0

def test_iterations(iterations: Iterations):
    iterations.put(structure({
        'hash': "1234",
        'epoch': 5,
        'phase': 'train',
        'arguments': {
            'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
            'batch_size': 64,
            'shuffle': True
        }
    }, Iteration))

    assert len(iterations.list()) == 1
    
    iterations.put(structure({
        'hash': "1234",
        'epoch': 5,
        'phase': 'train',
        'arguments': {
            'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
            'batch_size': 64,
            'shuffle': True
        }
    }, Iteration))

    assert len(iterations.list()) == 1

    iterations.put(structure({
        'hash': "1234",
        'epoch': 5,
        'phase': 'evaluation',
        'arguments': {
            'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
            'batch_size': 64,
            'shuffle': True
        }
    }, Iteration))
    assert len(iterations.list()) == 2

    
    iterations.put(structure({
        'hash': "1234",
        'epoch': 6,
        'phase': 'train',
        'arguments': {
            'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
            'batch_size': 64,
            'shuffle': True
        }
    }, Iteration))

    assert len(iterations.list()) == 2

    iterations.put(structure({
        'hash': "12345",
        'epoch': 7,
        'phase': 'train',
        'arguments': {
            'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': False}},
            'batch_size': 64,
            'shuffle': True
        }
    }, Iteration))

    assert isinstance(iterations.list()[0], Iteration)


def test_model_accessors(models: Models):
    model = models.create(id='1', name='mlp-classifier')
    model.metrics.add(structure({
        'name': 'accuracy',
        'value': 0.9,
        'epoch': 1,
        'phase': 'train'
    }, Metric))

    model.modules.put(structure({
        'type': 'optimizer',
        'hash': 'hash1',
        'name': 'Adam',
        'epoch': 1,
        'arguments': {'lr': 0.1}
    }, Module))

    model.iterations.put(structure({
        'hash': "1234",
        'epoch': 5,
        'phase': 'train',
        'arguments': {
            'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
            'batch_size': 64,
            'shuffle': True
        }
    }, Iteration))

    assert len(model.metrics.list()) == 1
    assert len(model.modules.list('optimizer')) == 1
    assert len(model.iterations.list()) == 1