import torch
import torch.nn as nn

# from libs.conv2d import calc_conv_return_shape


class Linear(nn.Linear):
    def __init__(self, *args, supervised=False, **kwargs) -> None:
        super().__init__(*args, **kwargs, bias=False)
        self.weight.requires_grad = False
        self.labels = None
        self.hebbian_learning = False
        self.supervised = supervised

    def forward(self, x, labels=None):
        if self.hebbian_learning is True:
            u = x @ self.weights
            # add teached signal to neuron output
            teacher = torch.ones_like(u)
            if self.supervised is True:
                assert self.labels is not None, "No labels provided for teacher signal"
                repeats = self.out_features // self.num_classes
                # if not one-hot - one-hot
                if len(self.labels.shape) == 1:
                    labels_onehot = torch.nn.functional.one_hot(
                        self.labels, num_classes=self.num_classes
                    ).float()
                labels_onehot = labels_onehot.repeat_interleave(repeats, dim=-1)
                teacher[:, : labels_onehot.size(-1)] *= labels_onehot
            u *= teacher
            uw = u.unsqueeze(1).repeat(
                1, self.weights.size(0), 1
            ) * self.weights.unsqueeze(0)
            y = self.strategy(u)
            diffs = self.lr * y.unsqueeze(1) * (x.unsqueeze(-1) - uw)
            self.weights += diffs.mean(dim=0)
        # train or not, return super()
        return super().forward(x)

    @property
    def weights(self):
        return self.weight.T

    @weights.setter
    def weights(self, value):
        self.weight.data = value.T

    def enable_hebbian(self, labels=None):
        self.hebbian_learning = True
        self.labels = labels

    def disable_hebbian(self):
        self.hebbian_learning = False
        self.labels = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"""Hebbian Linear(in_features={self.in_features}, out_features={self.out_features},
            bias={self.bias}, supervised={self.supervised})"""


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        supervised=False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.labels = None
        self.hebbian_learning = False
        self.weight.requires_grad = False
        self.supervised = supervised

    def forward(self, x):
        if self.hebbian_learning is True:
            x_unf = torch.nn.functional.unfold(
                input=x,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
            wf = self.weights
            u = torch.einsum("bij,bik->bjk", x_unf, wf.mT.unsqueeze(0)).unsqueeze(-1)
            # add teached signal to neuron output
            teacher = torch.ones_like(u)
            if self.supervised is True:
                assert self.labels is not None, "No labels provided for teacher signal"
                repeats = self.out_channels // self.num_classes
                # if not one-hot - one-hot
                if len(self.labels.shape) == 1:
                    labels_onehot = torch.nn.functional.one_hot(
                        self.labels, num_classes=self.num_classes
                    ).float()
                labels_onehot = (
                    labels_onehot.unsqueeze(-1)
                    .unsqueeze(1)
                    .repeat_interleave(repeats, dim=2)
                    .broadcast_to(teacher.shape)
                )
                teacher *= labels_onehot

            y = self.strategy(u)
            diffs_all = self.lr * y * (x_unf.mT.unsqueeze(2) - u * wf)
            diffs = diffs_all.mean(1)
            self.weights += diffs.mean(dim=0)
        # train or not, return super()
        return super().forward(x)

    @property
    def weights(self):
        return self.weight.flatten(1)

    @weights.setter
    def weights(self, value):
        self.weight.data = value.reshape(self.weight.shape)

    def enable_hebbian(self, labels=None):
        self.hebbian_learning = True
        self.labels = labels

    def disable_hebbian(self):
        self.hebbian_learning = False
        self.labels = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"""Hebbian Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size},
            padding={self.padding}, bias={self.bias}, supervised={self.supervised})"""
