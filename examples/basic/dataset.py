from torch.utils.data import Dataset
from torchvision.datasets.mnist import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

class Fashion(Dataset):
    def __init__(self, train: bool, normalize: bool = True):
        self.transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) if normalize else ToTensor()
        self.data = FashionMNIST(root='data/datasets', train=train, download=True, transform=self.transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]