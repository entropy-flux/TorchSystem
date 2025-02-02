from typing import override
from typing import Optional
from tinydb import TinyDB, where
from cattrs import unstructure
from typing import Optional

from tinysys.ports.modules import Module
from tinysys.ports.modules import Modules as Collection 

class Modules(Collection):
    def __init__(self, database: TinyDB, owner: str):
        self.owner = owner
        self.database = database
        self.table = self.database.table('modules')
    
    @override
    def build(self, *args, **kwargs) -> Module:
        return Module(*args, **kwargs)
    
    @override
    def list(self, type: str) -> list[Module]:
        results = self.table.search(where('owner') == self.owner and where('type') == type)
        return [Module(**{key: value for key, value in result.items() if key != 'owner'}) for result in results]
        
    @override
    def last(self, type: str) -> Optional[Module]:
        modules = self.table.search(where('owner') == self.owner and where('type') == type)
        last = modules[-1] if modules else None
        return Module(**{key: value for key, value in last.items() if key != 'owner'}) if last else None
    
    @override
    def put(self, module: Module):
        modules = self.table.search(where('owner') == self.owner and where('type') == module.type)
        if not modules:
            self.table.insert(unstructure(module) | {'owner': self.owner})
        elif module.hash == modules[-1]['hash']:
            self.table.update({'epoch': module.epoch}, doc_ids=[modules[-1].doc_id])
        else:
            self.table.insert(unstructure(module) | {'owner': self.owner})

    @override
    def clear(self):
        self.table.remove(where('owner') == self.owner)