from torchsystem.storage import Weights
from torchsystem.repository import Repository
from torchsystem.settings import Settings
from examples._1_simple_mnist.aggregate import Classifier

class Classifiers(Repository):
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.weights = Weights(self.settings)

    def store(self, classifier: Classifier):
        self.weights.store(classifier.model)

    def restore(self, classifier: Classifier):
        self.weights.restore(classifier.model)