from torchsystem import Settings
from torchsystem.storage import Models
from torchsystem.repository import Repository
from examples.simple_example.aggregate import Classifier

class Classifiers(Repository):
    def __init__(self, settings: Settings):
        self.models = Models(settings)

    def store(self, classifier: Classifier):
        self.models.store(classifier.model)

    def restore(self, classifier: Classifier):
        self.models.restore(classifier.model)