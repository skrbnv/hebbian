from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from typing import Tuple, Any
from torch import Tensor


class CIFAR10_GPU(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        augment: bool = False,
        download: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(root=root, train=train, download=download)
        tfn = transforms.Normalize(
            mean=(0.49139968, 0.48215827, 0.44653124),
            std=(0.24703233, 0.24348505, 0.26158768),
        )
        self.data = tfn(
            Tensor(self.data / 255.0).float().to(device).permute(0, 3, 1, 2)
        )
        self.targets = Tensor(self.targets).long().to(device)
        self.augment = augment
        self.augment_fn = transforms.Compose(
            [
                # transforms.Pad(8, fill=0, padding_mode="constant"),
                transforms.RandomCrop(28),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = self.data[index], self.targets[index]
        if self.augment is True:
            img = self.augment_fn(img)
        return (img, label)


"""
class MNIST_GPU(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        augment: bool = False,
        download: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(root=root, train=train, download=download)
        tfn = transforms.Normalize((0.1307,), (0.3081,))
        self.data = tfn((self.data / 255.0).float().unsqueeze(1).to(device))
        self.targets = self.targets.long().to(device)
        self.augment = augment
        self.augment_fn = transforms.Compose(
            [
                transforms.Pad(2, fill=0, padding_mode="constant"),
                transforms.RandomCrop(28),
            ]
        )
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = self.data[index], self.targets[index]
        if self.augment is True:
            img = self.augment_fn(img)
        return (img, label)
"""


def generate(batch_size: list = [16, 16], num_workers: list = [0, 0], device="cpu"):
    trainset = CIFAR10_GPU(
        root="./CIFAR10/train", train=True, augment=False, download=True, device=device
    )
    testset = CIFAR10_GPU(
        root="./CIFAR10/test", train=False, augment=False, download=True, device=device
    )
    traintestset = Subset(
        CIFAR10_GPU(
            root="./CIFAR10/train",
            train=True,
            augment=False,
            download=True,
            device=device,
        ),
        range(10000),
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
    )
    train_test_loader = DataLoader(
        traintestset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
    )
    return train_loader, test_loader, train_test_loader
