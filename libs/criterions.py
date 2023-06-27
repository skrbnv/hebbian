import torch
from typing import Any


class WTA:
    def __init__(self, topk, delta) -> None:
        self.topk = topk
        self.delta = delta

    def __call__(self, y, weights) -> Any:
        bs = y.size(0)
        _, indices = torch.topk(y, k=self.topk, dim=1)
        amp = torch.zeros((bs, weights.size(1)), device=y.device)
        for i in range(bs):
            amp[i, indices[i, 0]] = 1.0
            amp[i, indices[i, 1:]] = -self.delta

        return torch.sum(amp * y, 0), amp
