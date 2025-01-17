from torchsystem import Repository, Session

db = {}

class Aggregate:
    def __init__(self, id, name):
        self.id = id
        self.name = name

class ExampleRepository(Repository[Aggregate]):
    def store(self, aggregate: Aggregate):
        db[aggregate.id] = aggregate.name

    def restore(self, aggregate: Aggregate):
        aggregate.name = db[aggregate.id]

def test_repository():
    repository = ExampleRepository()
    aggregate = Aggregate(1, 'example')
    with Session(repository) as session:
        repository.put(aggregate)
    assert db[1] == 'example'

    with Session(repository) as session:
        repository.put(aggregate)
        aggregate.name = 'example 2'
        session.rollback()
    assert db[1] == 'example'

    with Session(repository) as session:
        repository.put(aggregate)
        aggregate.name = 'example 2'
    assert db[1] == 'example 2'