from torchsystem.storage import Models, Criterions, Optimizers, Datasets
from torchsystem.repository import Repository
from torchsystem.settings import Settings
from examples._2_with_events.aggregate import Classifier
from examples._2_with_events.events import Stored, Restored

class Classifiers(Repository):
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.models = Models(self.settings)
        self.criterions = Criterions(self.settings)
        self.optimizers = Optimizers(self.settings)
        self.datasets = Datasets()

    def store(self, classifier: Classifier):
        self.models.store(classifier.model)
        classifier.emit(Stored(classifier))

    def restore(self, classifier: Classifier):
        self.models.restore(classifier.model)
        classifier.emit(Restored(classifier))