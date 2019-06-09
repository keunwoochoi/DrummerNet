from globals import *
import torch.nn.functional as F


def dcnp(torch_array):
    """helper function for detach - cpu - numpy conversion"""
    return torch_array.detach().cpu().numpy()


def pad_multiple_of(x, div):
    """
    Compute and pad so that `x.dim[1] % div` becomes `0`.
    Ii assumes x.shape to be (b, time, ch)

    Args:
        x (torch.Tensor): (batch, time, *) tensor to be padded

        div (int): number to divide

    """
    time_length = x.shape[1]
    pad_length = (div - (time_length % div)) % div
    if pad_length == 0:
        return x
    else:
        return F.pad(x, (0, pad_length))
