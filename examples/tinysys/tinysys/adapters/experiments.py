from uuid import uuid4, UUID
from typing import override
from typing import Optional

from tinydb import TinyDB, where

from tinysys.ports.experiments import Experiment
from tinysys.ports.experiments import Experiments as Collection 
from tinysys.adapters.models import Models

class Experiments(Collection):
    def __init__(self, database: TinyDB):
        self.database = database
        self.table = self.database.table('experiments')
    
    @override
    def create(self, name: str) -> Experiment:
        if self.table.search(where('name') == name):
            raise ValueError(f"Experiment with name {name} already exists")
        id = uuid4()
        self.table.insert({'id': str(id), 'name': name})
        return Experiment(
            id=id, 
            name=name, 
            models=Models(self.database, str(id))
        )

    @override    
    def read(self, by: str, **kwargs) -> Optional[Experiment]:
        data = None
        match by:
            case 'id':
                id = kwargs['id']
                data = self.table.get(where('id') == str(id))
            case 'name':
                name = kwargs['name']
                data = self.table.get(where('name') == name)
            case _:
                raise ValueError(f"Invalid search criteria {by}")
             
        return Experiment(
            id=UUID(data['id']),
            name=data['name'],
            models=Models(self.database, data['id'])
        ) if data else None
    
    @override
    def update(self, id: UUID, name: str) -> Experiment:
        self.table.update({'name': name}, where('id') == str(id))
        return Experiment(
            id=id, 
            name=name,
            models=Models(self.database, str(id))
        )
    
    @override
    def delete(self, id: UUID):
        experiment = self.read('id', id=id)
        if not experiment:
            raise ValueError(f"Experiment with id {id} not found")
        experiment.models.clear()
        self.table.remove(where('id') == str(id))

    @override
    def list(self) -> list[Experiment]:
        return [Experiment(
            id=UUID(data['id']), 
            name=data['name'],
            models=Models(self.database, data['id'])
        ) for data in self.table.all()]

def getexperiments() -> Experiments:
    return Experiments(database=TinyDB('data/database.json'))

def getexperiment(name: str) -> Experiment:
    experiments = getexperiments()
    experiment = experiments.read('name', name=name)
    return experiment if experiment else experiments.create(name)

def getmodels(expname: str) -> Models:
    experiment = getexperiment(expname)
    return experiment.models