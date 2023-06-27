import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(x)


class Self(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class Ones(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.ones_like(x)


class WTA(nn.Module):
    def __init__(self, topk=5) -> None:
        super().__init__()
        self.topk = topk

    def forward(self, x):
        output = torch.zeros_like(x).flatten(1)
        for i in range(x.size(0)):
            topk = torch.topk(x[i].flatten(), self.topk)
            output[i][topk.indices] = 1.0
        return output.reshape(x.shape)


class WTA_with_lateral_inhibition(nn.Module):
    def __init__(self, topk=5, inhibition_radius=5) -> None:
        super().__init__()
        self.topk = topk
        self.radius = inhibition_radius

    def forward(self, x):
        output = torch.zeros_like(x).flatten(1)
        y = x.flatten(1)
        # for each element of the batch
        for i in range(x.size(0)):
            # for each _ in topk
            for _ in range(self.topk):
                # get top1
                top = torch.topk(y[i].flatten(), 1)
                output[i][top.indices] = 1.0
                # mask y[i]
                y[i][top.indices - self.radius : top.indices + self.radius] = y[i].min()
        return output.reshape(x.shape)
