import os
from torch import save
from torch import load
from tinysys.domain import Model

class Repository:
    def __init__(self, root: str = 'data/weights', path: str = None):
        self.root = root
        self.path = path or 'default'
        if not os.path.exists(os.path.join(self.root, self.path)):
            os.makedirs(os.path.join(self.root, self.path))

    def store(self, model: Model):
        save(model.nn, os.path.join(self.root, self.path, f'{model.id}.pth'))

    def restore(self, model: Model):
        model.nn.load_state_dict(load(os.path.join(self.root, self.path, f'{model.id}.pth'), weights_only=False).state_dict())