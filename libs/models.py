import torch
import torch.nn as nn
import libs.hbb as hbb
import libs.ff as ff


class Models(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def has_hebbian(self):
        for m in self.modules():
            if isinstance(m, hbb.Linear) or isinstance(m, hbb.Conv2d):
                return True
        return False

    def has_backprop(self):
        for m in self.modules():
            if (isinstance(m, nn.Linear) and not isinstance(m, hbb.Linear)) or (
                isinstance(m, nn.Conv2d) and not isinstance(m, hbb.Conv2d)
            ):
                return True
        return False


# Pure backprop 98.6-99.0% acc
model_bp = nn.Sequential(
    *[
        nn.Conv2d(1, 64, (3, 3), 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        nn.Linear(12544, 2000),
        nn.ReLU(),
        nn.LayerNorm(2000),
        nn.Linear(2000, 10),
    ]
)


# Hebbian
class HEBB1(Models):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            *[
                hbb.Conv2d(3, 100, (3, 3), 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(100),
                nn.Flatten(),
                hbb.Linear(25600, 10),
            ]
        )

    def forward(self, x):
        return self.seq(x)


class HEBB2(Models):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            *[
                hbb.Conv2d(3, 100, (3, 3), 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(100),
                nn.Flatten(),
                hbb.Linear(25600, 10),
            ]
        )

    def forward(self, x):
        return self.seq(x)


class HEBB3(Models):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            *[
                hbb.Conv2d(3, 100, 3, 2, 1, supervised=True),
                nn.ReLU(),
                nn.BatchNorm2d(100),
                hbb.Conv2d(100, 250, 3, 2, 1, supervised=True),
                nn.ReLU(),
                nn.BatchNorm2d(250),
                hbb.Conv2d(250, 500, 3, 2, 1, supervised=True),
                nn.ReLU(),
                nn.BatchNorm2d(500),
                hbb.Conv2d(500, 1000, 3, 2, 1, supervised=True),
                nn.ReLU(),
                nn.BatchNorm2d(1000),
                nn.Flatten(),
                hbb.Linear(4000, 10, supervised=True),
            ]
        )
        # self.final = nn.Linear(4000, 10)

    def forward(self, x):
        x = self.seq(x)
        # x = self.final(x)
        return x


class BP3(Models):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            *[
                nn.Conv2d(3, 100, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(100),
                nn.Conv2d(100, 250, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(250),
                nn.Conv2d(250, 500, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(500),
                nn.Conv2d(500, 1000, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(1000),
                nn.Flatten(),
                # hbb.Linear(4000, 10),
            ]
        )
        self.final = nn.Linear(4000, 10)

    def forward(self, x):
        x = self.seq(x)
        x = self.final(x)
        return x


class HEBB3SUM(Models):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.s1 = nn.Sequential(
            *[
                hbb.Conv2d(3, 100, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(100),
            ]
        )
        self.s2 = nn.Sequential(
            *[
                hbb.Conv2d(3, 100, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(100),
            ]
        )
        self.seq = nn.Sequential(
            *[
                hbb.Conv2d(200, 250, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(250),
                hbb.Conv2d(250, 500, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(500),
                hbb.Conv2d(500, 1000, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(1000),
                nn.Flatten(),
                hbb.Linear(4000, 10),
            ]
        )

    def forward(self, x):
        x1 = self.s1(x)
        x2 = self.s2(x)
        x = torch.stack((x1, x2), dim=1)
        return self.seq(x)


# Hebbian + Forward-Forward
class HEBB3FF(Models):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            *[
                hbb.Conv2d(3, 64, (3, 3), 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                hbb.Conv2d(64, 256, (3, 3), 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Flatten(),
            ]
        )
        self.hebb = ff.Linear(16384, 10)

    def forward(self, x, labels):
        x = self.seq(x)
        x = self.hebb(x, labels)
        return x


model_hebb2 = nn.Sequential(
    *[
        hbb.Conv2d(1, 64, (3, 3), 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        hbb.Linear(12544, 10),
    ]
)

model_hebb3 = nn.Sequential(
    *[
        hbb.Conv2d(3, 64, (3, 3), 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        hbb.Conv2d(64, 256, (3, 3), 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Flatten(),
        hbb.Linear(16384, 10),
    ]
)

model_hebb4 = nn.Sequential(
    *[
        hbb.Conv2d(1, 64, (3, 3), 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.AvgPool2d(2),
        hbb.Conv2d(64, 256, (3, 3), 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        hbb.Conv2d(256, 512, (3, 3), 2, 2),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Flatten(),
        hbb.Linear(4608, 10),
    ]
)


model2 = nn.Sequential(
    *[
        nn.Flatten(),
        hbb.Linear(784, 2000),
        nn.ReLU(),
        nn.LayerNorm(2000),
        nn.Linear(2000, 10),
    ]
)
