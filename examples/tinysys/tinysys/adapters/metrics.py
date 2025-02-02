from typing import override

from tinydb import TinyDB, where
from cattrs import unstructure

from tinysys.ports.metrics import Metric
from tinysys.ports.metrics import Metrics as Collection

class Metrics(Collection):
    def __init__(self, database: TinyDB, owner: str):
        self.owner = str(owner)
        self.database = database
        self.table = self.database.table('metrics')
    
    @override
    def add(self, metric: Metric):
        self.table.insert(unstructure(metric) | {'owner': self.owner})

    @override
    def list(self) -> list[Metric]:
        metrics = self.table.search(where('owner') == self.owner)
        return [Metric(**{key: value for key, value in metric.items() if key != 'owner'}) for metric in metrics]
    
    @override
    def clear(self):
        self.table.remove(where('owner') == self.owner)