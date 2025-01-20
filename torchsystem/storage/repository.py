from abc import ABC
from pymsgbus.models import Event
from pymsgbus.consumers import Consumer
from torchsystem.aggregate import Aggregate

class Repository[T: Aggregate]:
    """
    In DDD, the REPOSITORY is a design pattern that provides an abstraction for data access andis used to separate
    the business logic from the underlying data access logic, and have the ability of storing and restoring entire AGGREGATES.
    
    Some storage workflows are complex and may require from EVENTS, since it's logic may not be known during the
    implementation of the REPOSITORY. This base class for repositories implements the PRODUCER protocol stated in the pymsgbus
    library to provide a mechanism for dispatching EVENTS to CONSUMERS.

    A CONSUMER can listen to the REPOSITORY and consume the EVENTS produced by it to help it to persist the AGGREGATE or restore it
    from the underlying storage, for example when some data is not directly related to pytorch objects but necessary to mantain the
    consistency of the AGGREGATE.
    """
    def __init__(self):
        self.consumers = list[Consumer]()

    def register(self, *consumers):
        """
        Register consumers to consume events produced by the repository.
        """
        self.consumers.extend(consumers)

    def dispatch(self, event: Event):
        """
        Dispatch an event to all consumers registered in the repository

        Args:
            event (Event): The event to be dispatched to the consumers.
        """
        for consumer in self.consumers:
            consumer.consume(event)

    def store(self, aggregate: T):
        """
        Implement this method to store the AGGREGATE in the underlying storage. The AGGREGATE should be
        stored entirely in a single operation.
        Args:
            aggregate (T): The AGGREGATE to be stored.
        """
        ...

    def restore(self, aggregate: T):
        """
        Restore an aggregate. Implement this method to restore the AGGREGATE from the underlying storage. The
        AGGREGATE should be restored entirely in a single operation.

        Args:
            aggregate (T): The AGGREGATE to be restored.
        """
        ...