from os import path, makedirs
from shutil import rmtree
from pytest import fixture
from logging import getLogger

logger = getLogger(__name__)

@fixture(scope='session')
def directory():
    if not path.exists('data/test'):
        makedirs('data/test')
    yield
    try:
        rmtree('data/test')
    except PermissionError:
        logger.warning('Could not remove data directory')