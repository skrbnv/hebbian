from torch.linalg import norm
from torch.nn import Module


class SimpleNorm(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            return x / (norm(x, ord=2, dim=1, keepdim=True) + 1e-4)
        elif len(x.shape) == 4:
            return x / (norm(x, ord=2, dim=(-1, -2), keepdim=True) + 1e-4)
        else:
            raise Exception("Unsupported shape")
