from os import path, makedirs
from shutil import rmtree
from pytest import fixture
from logging import getLogger
from tinydb import TinyDB 
from tinysys.adapters.experiments import Experiments 
from tinysys.adapters.models import Models
from tinysys.adapters.modules import Modules
from tinysys.adapters.metrics import Metrics
from tinysys.adapters.iterations import Iterations
from tinysys.repository import Repository

logger = getLogger(__name__)

@fixture(scope='function')
def database():
    if not path.exists('data/test'):
        makedirs('data/test')
    yield TinyDB('data/test/database.json')
    try:
        rmtree('data/test')
    except PermissionError:
        logger.warning('Could not remove data directory') 
        
@fixture(scope='function')
def experiments(database):
    experiments = Experiments(database)
    yield experiments

@fixture(scope='function')
def experiment(experiments: Experiments):
    experiment = experiments.create(name='test')
    yield experiment
    experiments.delete(id=experiment.id)

@fixture(scope='function')
def models(database):
    models = Models(database, 'test')
    yield models
    models.clear()

@fixture(scope='function')
def modules(database):
    modules = Modules(database, 'test')
    yield modules
    modules.clear()

@fixture(scope='function')
def metrics(database):
    metrics = Metrics(database, 'test')
    yield metrics
    metrics.clear()

@fixture(scope='function')
def iterations(database):
    iterations = Iterations(database, 'test')
    yield iterations
    iterations.clear()

@fixture(scope='function')
def repository():
    yield Repository(root='data/test', path='weights')
