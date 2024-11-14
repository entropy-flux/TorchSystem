from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchsystem.loaders import Loaders
from torchsystem.storage import Datasets
from mlregistry import get_metadata

class Digits(Dataset):
    def __init__(self, train: bool, normalize: bool):
        transform = Compose([ToTensor()]) if not normalize else Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        self.dataset = MNIST(root='data/datasets', train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


def test_loaders():
    Datasets.register(Digits)
    loaders = Loaders()
    loaders.add('train', Digits(train=True, normalize=True), batch_size=32, shuffle=True, drop_last=True)
    loaders.add('test', Digits(train=False, normalize=True), batch_size=32, shuffle=False)

    for phase, loader in loaders:
        if phase == 'train':
            metadata = get_metadata(loader)
            assert metadata.arguments == {'batch_size': 32, 'shuffle': True, 'drop_last': True}
            metadata = get_metadata(loader.dataset)
            assert metadata.name == 'Digits'
            assert metadata.arguments == {'train': True, 'normalize': True}
        elif phase == 'test':
            metadata = get_metadata(loader)
            assert metadata.arguments == {'batch_size': 32, 'shuffle': False}
            metadata = get_metadata(loader.dataset)
            assert metadata.name == 'Digits'
            assert metadata.arguments == {'train': False, 'normalize': True}