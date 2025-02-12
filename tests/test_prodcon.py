from typing import Sequence 
from torchsystem import Depends
from torchsystem.services import event
from torchsystem.services import Consumer
from torchsystem.services import Producer

@event
class ModelTrained:
    metrics: Sequence

@event
class ModelEvaluated:
    metrics: Sequence

@event
class ModelDeployed:
    pass

consumer = Consumer() 

def getdb():
    pass

@consumer.handler
def on_model_iterated(event: ModelTrained | ModelEvaluated, db = Depends(getdb)): 
    db.append(event.metrics)

@consumer.handler
def on_model_deployed(event: ModelDeployed, db = Depends(getdb)):
    db.clear()

db = []

def test_consumer():
    db.clear()
    consumer.dependency_overrides[getdb] = lambda: db

    producer = Producer()
    producer.register(consumer)

    producer.dispatch(ModelTrained([1, 2, 3]))
    producer.dispatch(ModelEvaluated([4, 5, 6]))

    assert db == [[1, 2, 3], [4, 5, 6]]

    producer.dispatch(ModelDeployed())

    assert db == []