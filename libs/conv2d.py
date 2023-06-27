from typing import Union
import math


def calc_conv_return_shape(
    dim: tuple[int, int],
    kernel_size: Union[int, tuple],
    stride: Union[int, tuple] = (1, 1),
    padding: Union[int, tuple] = (0, 0),
    dilation: Union[int, tuple] = (1, 1),
) -> tuple[int, int]:
    """
    Calculates the return shape of the Conv2D layer.
    Works on the MaxPool2D layer as well

    See Also: https://github.com/pytorch/pytorch/issues/79512

    See Also: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    See Also: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    Args:
        dim: The dimensions of the input. For instance, an image with (h * w)
        kernel_size: kernel size
        padding: padding size
        dilation: dilation size
        stride: stride size

    Returns:
        Dimensions of the output of the layer
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(stride, int):
        stride = (stride, stride)
    h_out = math.floor(
        (dim[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]
        + 1
    )
    w_out = math.floor(
        (dim[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]
        + 1
    )
    return h_out, w_out
