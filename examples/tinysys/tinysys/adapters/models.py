from typing import Optional
from tinydb import TinyDB, where
from typing import override

from tinysys.ports.models import Models as Collection
from tinysys.ports.models import Model

from tinysys.adapters.metrics import Metrics
from tinysys.adapters.modules import Modules
from tinysys.adapters.iterations import Iterations

class Models(Collection):
    def __init__(self, database: TinyDB, owner: str):
        self.database = database
        self.owner = owner
        self.table = self.database.table('models')

    @override
    def create(self, id: str, name: str) -> Model:
        if self.table.search(where('experiment') == self.owner and where('id') == id):
            raise ValueError(f'Model with id {id} already exists')
        self.table.insert({'id': id, 'name': name, 'epoch': 0} | {'experiment': self.owner})
        return Model(
            id=id, 
            name=name,
            epoch = 0, 
            metrics=Metrics(self.database, id),
            modules=Modules(self.database, id),
            iterations=Iterations(self.database, id)
        )

    @override
    def update(self, id: str, epoch: int):
        self.table.update({'epoch': epoch}, where('experiment') == self.owner and where('id') == id)
        return self.read(id)

    @override
    def read(self, id: str) -> Optional[Model]:
        model = self.table.get(where('experiment') == self.owner and where('id') == id) 
        return Model(
            id=model['id'], 
            name=model['name'], 
            epoch = model['epoch'],
            metrics=Metrics(self.database, id),
            modules=Modules(self.database, id),
            iterations=Iterations(self.database, id) 
        ) if model else None

    @override
    def delete(self, id: str): 
        model = self.read(id)
        if not model:
            raise ValueError(f'Model with id {id} does not exist')
        model.metrics.clear()
        model.modules.clear()
        model.iterations.clear()
        self.table.remove(where('experiment') == self.owner and where('id') == id)

    @override
    def list(self) -> list[Model]:
        models = self.table.search(where('experiment') == self.owner)
        return [Model(
            id=model['id'],  
            name=model['name'],
            epoch = model['epoch'], 
            metrics=Metrics(self.database, id),
            modules=Modules(self.database, id),
            iterations=Iterations(self.database, id)
        ) for model in models]
    
    @override
    def clear(self): 
        for model in self.list():
            self.delete(model.id)