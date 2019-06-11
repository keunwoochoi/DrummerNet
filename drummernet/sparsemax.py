"""Sparsemax activation function.
Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)

https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import *


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.contiguous().view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, device=input.device).view(1, -1)
        range = range.expand_as(zs).type(TCDTYPE)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


def stable_gumbel_softmax(x, dim, tau=1, hard=False):
    """do it with stabilizing."""

    # reshape to make it 2D
    original_size = x.size()
    x = x.contiguous().view(-1, x.size(dim))

    # get numerical stable logits
    x = x - torch.max(x, dim=dim, keepdim=True)[0].expand_as(x)
    x = torch.exp(x)

    # apply gumbel softmax!
    return F.gumbel_softmax(x, tau=tau, hard=hard).view(original_size)


def relaxed_hardmax(x, dim=-1):
    """Also called straight-through backprop, which uses softmax's gradient.

    Dunno why but it doesn't work (2018-11-28-hardmax_over_time) - strange training..
    """

    def _onehot(x_in, dim):
        shape = x_in.size()
        _, ind = x_in.max(dim=dim)
        y_hard = torch.zeros_like(x_in).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        return y_hard.view(*shape)

    x = x - torch.max(x, dim=dim, keepdim=True)[0].expand_as(x)
    y = F.softmax(x, dim=dim)
    return y + (_onehot(y, dim=dim) - y).detach()
