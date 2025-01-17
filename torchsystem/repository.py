from abc import ABC, abstractmethod
from typing import Any
from torchsystem.aggregate import Aggregate

class Repository[T: Aggregate](ABC):
    """
    An abstract base class for repository classes. It is responsible for coordinating the storage of aggregates.
    In order to implement a repository class, the `store` and `restore` methods must be implemented. It implements
    the necessary methods to be used with a torchsystem `Session` object. It implements the necessary methods to
    be used with a `Session` object to handle ACID transactions.

    Methods:
        begin:
            Begin a transaction by creating empty identity maps.
        
        commit:
            Commit the uncommited aggregates to the repository. It will persist the aggregates using the `store` method
            and clear the uncommited identity map.
        
        rollback:
            Rollback the uncommited aggregates. It will restore the aggregates using the `restore` method and clear the
            uncommited identity map.
        
        close:
            Close the repository by clearing the identity maps.
        
        put:
            Put an aggregate in the repository. If the aggregate is already in the repository, it will be replaced.
        
        store:
            Overwrite this method to store an aggregate in the repository. It should persist the aggregate in a storage.
        
        restore:
            Overwrite this method to restore an aggregate from the repository. It should restore the aggregate from the
            underlying storage.

    Example:

        .. code-block:: python

        from torchsystem.storage import Weights
        from torchsystem.repository import Repository
        from torchsystem.settings import Settings

        class Classifiers(Repository[Classifier]):
            def __init__(self, settings: Settings = None):
                self.settings = settings or Settings()
                self.weights = Weights(self.settings)

            def store(self, classifier: Classifier):
                self.weights.store(classifier.model)

            def restore(self, classifier: Classifier):
                self.weights.restore(classifier.model)

        ...

        classifiers = Classifiers()

        with Session(classifiers) as session:
            classifiers.put(classifier)
            session.commit()
    """
    
    def begin(self):
        """
        Begin a transaction by creating empty identity maps.
        """
        self.commited = dict[Any, T]()
        self.uncommited = dict[Any, T]()
    
    def commit(self):
        """
        Commit the uncommited aggregates to the repository. It will persist
        the aggregates using the `store` method and clear the uncommited identity map.
        """
        for id, aggregate in self.uncommited.items():
            self.commited[id] = aggregate
        self.uncommited.clear()

        for id, aggregate in self.commited.items():
            self.store(aggregate)

    def rollback(self):
        """
        Rollback the uncommited aggregates. It will restore the aggregates using the `restore` method
        and clear the uncommited identity map.
        """
        for aggregate in self.uncommited.values():
            self.restore(aggregate)
        self.uncommited.clear()

    def close(self):
        """
        Close the repository by clearing the identity maps.
        """
        self.commited.clear()
        self.uncommited.clear()

    def put(self, aggregate: T):
        """
        Put an aggregate in the repository. If the aggregate is already in the repository, it will be replaced.

        Args:
            aggregate (T): The aggregate to put in the repository.
        """
        if aggregate.id in self.commited.keys():
            self.commited.pop(aggregate.id)
        self.uncommited[aggregate.id] = aggregate

    @abstractmethod
    def store(self, aggregate: T):
        """
        Overwrite this method to store an aggregate in the repository. It should persist the aggregate in a storage.
        Will be called by the `commit` method within a session.
        Args:
            aggregate (T): The aggregate to store in the repository.
        """ 
        ...

    @abstractmethod
    def restore(self, aggregate: T):
        """
        Overwrite this method to restore an aggregate from the repository. It should restore the aggregate from the
        underlying storage. Will be called by the `rollback` method within a session.

        Args:
            aggregate (T): The aggregate to restore from the repository.
        """
        ...