import torch


def complex_mul(t1, t2):
    """complex number multiplication

    Args:
        t1, t2 (torch.Tensor): complex representations of torch tensor.
            t1 and t2 sizes should be the same and one of {2, 3, 4}-dimensional.
    """

    if t1.dim() != t2.dim():
        raise ValueError('dim mismatch in complex_mul, {} and {}'.format(t1.dim(), t2.dim()))

    if t1.dim() == 2:
        r1, i1 = t1[:, 0], t1[:, 1]
        r2, i2 = t2[:, 0], t2[:, 1]
    elif t1.dim() == 3:
        r1, i1 = t1[:, :, 0], t1[:, :, 1]
        r2, i2 = t2[:, :, 0], t2[:, :, 1]
    elif t1.dim() == 4:
        r1, i1 = t1[:, :, :, 0], t1[:, :, :, 1]
        r2, i2 = t2[:, :, :, 0], t2[:, :, :, 1]
    else:
        raise NotImplementedError

    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def fast_conv1d(signal, kernel):
    """fast 1d convolution, assuming filter is shorter than signal
    This function is not so much general (shapes of input etc) but
    let's just use it for now...

    The operation is exactly convolution - the kernel doesn't need to be flipped.

    Args:
        signal (torch.Tensor): (batch, ch=1, time)

        kernel (torch.Tensor): (time, ), or some dim expansion of it
    """
    batch, ch, L_sig = signal.shape
    assert ch == 1
    kernel = kernel.reshape(1, -1)
    L_I = kernel.shape[1]
    L_F = 2 << (L_I - 1).bit_length()
    L_S = L_F - L_I + 1

    device_ = signal.device
    pad_kernel = L_F - L_I
    FDir = torch.rfft(torch.cat((kernel, torch.zeros(1, pad_kernel, device=device_)),
                                dim=1), signal_ndim=1)

    signal_sizes = [L_F]
    len_pad = (L_S - L_sig % L_S) % L_S
    offsets = range(0, L_sig, L_S)

    signal = torch.cat((signal, torch.zeros(batch, ch, len_pad, device=device_)), dim=2)

    result = torch.zeros(batch, 1, offsets[-1] + L_F).to(device_)
    pad_slice = L_F - L_S

    for idx_fr in offsets:
        idx_to_in = idx_fr + L_S
        idx_to_out = idx_fr + L_F
        to_rfft = torch.cat((signal[:, 0, idx_fr:idx_to_in],
                             torch.zeros(batch, pad_slice, device=device_)), dim=1)

        to_mul = torch.rfft(to_rfft, signal_ndim=1,
                            normalized=True)
        to_irfft = complex_mul(to_mul, FDir)

        conved_slice = torch.irfft(to_irfft, signal_ndim=1,
                                   signal_sizes=signal_sizes,
                                   normalized=True)
        result[:, 0, idx_fr: idx_to_out] += conved_slice

    return result[:, :, :L_sig]
