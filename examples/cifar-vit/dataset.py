from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

class Images(Dataset):
    def __init__(self, train=True, normalize=True):
        super().__init__()
        self.cifar = CIFAR10(root="./data", train=train, download=True)
        self.transform = Compose(
            [ToTensor(), Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))] 
            if normalize else [ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, index: int):
        image, label = self.cifar[index]
        image = self.transform(image)
        return image, label