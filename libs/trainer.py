import torch

# import torch.nn as nn
import libs.hbb as hbb

# from libs.criterions import WTA

# from libs.models import Models


class Trainer:
    def __init__(self, model, strategy=None, lr=1e-4, num_classes=10) -> None:
        self.model = model
        for m in self.model.modules():
            if isinstance(m, (hbb.Linear, hbb.Conv2d)):
                m.strategy = strategy
                m.lr = lr
                m.num_classes = num_classes

    def __call__(self, inputs, labels):
        for m in self.model.modules():
            if isinstance(m, (hbb.Linear, hbb.Conv2d)):
                m.enable_hebbian(labels)
        with torch.no_grad():
            output = self.model(inputs)
        for m in self.model.modules():
            if isinstance(m, (hbb.Linear, hbb.Conv2d)):
                m.disable_hebbian()
        return output


"""
class SimpleHebbian:
    def __init__(self, lr=1e-4) -> None:
        self.lr = lr

    def __call__(self, model, inputs):
        x = inputs
        for m in model.modules():
            if isinstance(m, nn.Sequential):
                continue
            elif isinstance(m, hbb.Linear):
                y = m(x)
                # x: inputs
                # y: outputs
                # m.weight: weights (256, 784)
                delta = self.lr * torch.bmm(x.unsqueeze(2), y.unsqueeze(1))
                m.weights += delta.mean(dim=0)
                x = m(x)
            else:
                x = m(x)


class WeightDecayHebbian:
    def __init__(self, lr=1e-4) -> None:
        self.lr = lr

    def __call__(self, model, inputs):
        x = inputs
        for m in model.modules():
            if isinstance(m, nn.Sequential):
                continue
            elif isinstance(m, hbb.Linear):
                y = m(x)
                # x: inputs
                # y: outputs
                # m.weight: weights (256, 784)
                dot = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))
                sub = x.unsqueeze(-1).repeat(
                    1, 1, m.weights.size(-1)
                ) - m.weights.unsqueeze(0).repeat(x.size(0), 1, 1)
                diffs = self.lr * dot * sub
                m.weights += diffs.mean(dim=0)
                x = m(x)
            else:
                x = m(x)


class KrotovHebbian:
    def __init__(
        self, lr=1e-6, precision=1e-30, delta=0.4, p=2, topk=5, normalize=False
    ) -> None:
        self.precision = precision
        self.lr = lr
        self.p = p
        self.normalize = normalize
        self.activation = WTA(topk, delta)

    def __call__(self, model, inputs):
        x = inputs
        for m in model.modules():
            if isinstance(m, nn.Sequential):
                continue
            elif isinstance(m, hbb.Linear):
                with torch.no_grad():
                    # [normalization ignored
                    if self.normalize:
                        norm = torch.norm(x, dim=1)
                        norm[norm == 0] = 1
                        x = torch.div(x, norm.view(-1, 1))
                    # ]

                    y = torch.matmul(
                        x, torch.sign(m.weights) * torch.abs(m.weights) ** (self.p - 1)
                    )

                    xx, amp = self.activation(y, m.weights)

                    # Apply the actual learning rule, from here on the tensor has the same dimension as the weights
                    norm_factor = (
                        xx.unsqueeze(0).broadcast_to(m.weights.shape) * m.weights
                    )
                    yy = x.T @ amp - norm_factor

                    # Normalize the weight updates so that the largest update is 1
                    # (which is then multiplied by the learning rate)
                    nc = torch.max(torch.abs(yy))
                    if nc < self.precision:
                        nc = self.precision
                    diffs = torch.div(yy, nc)
                    m.weights += self.lr * diffs
                    x = m(x)
            else:
                x = m(x)
                """
