import torch
import time_freq


def norm_losses(impulses, p, weight=1.0, eps=0.0):
    """L-1 regulariser for a list input

    Args:
        impulses (Iterable): a list of impulses, each is torch tensor (of course)

        p (float): which norm, e.g., 1, 2, ..

    """
    loss_sum = 0
    for impulse in impulses:
        loss_sum += torch.norm(impulse + eps, p)

    return weight * loss_sum / len(impulses)


def loss_stft(src_true, src_pred, n_fft, metrics, metric_names, losses, hop_length=None, weight=1.0):
    """compute stft magnitude based loss function

    Args:
        src_true (torch.Tensor): true waveforms

        src_pred (torch.Tensor): predicted waveforms

        n_fft (int): number of fft for the stft computation

        metrics (list of functions): metrics (e.g., mae, mse, ..)

        metric_names (list of str): metric names corresponding to `metrics`

        losses (dict): {loss_name: loss_value} dictionary to save the result

        hop_length (None or int): hop length for stft

        weight (float): final weighting value 

    """

    def _log_stft(src, n_fft, hop_length):
        win = torch.hann_window(n_fft).to(src.device)
        mag_specgrams = torch.stft(src, n_fft, hop_length, window=win).pow(2).sum(-1)  # (*, freq, time)

        return time_freq.to_decibel(mag_specgrams)

    mag_specgrams_true = _log_stft(src_true, n_fft, hop_length)
    mag_specgrams_pred = _log_stft(src_pred, n_fft, hop_length)

    for name, metric in zip(metric_names, metrics):
        losses['stft_' + name] = weight * metric(mag_specgrams_true, mag_specgrams_pred)


def loss_melgram(src_true, src_pred, mel_fb, n_fft, metrics, metric_names, losses, hop_length=None, weight=1.0):
    """Compute melgram-based loss
    by {compute STFT ** 2 -> transpose -> matmul with `mel_fb` -> log10()}
    This can be for either batch or an item

    Args:
        src_true (torch.Tensor): true waveforms

        src_pred (torch.Tensor): predicted waveforms

        mel_fb (torch.Tensor): Mel filterbank to use

        n_fft (int): number of fft for the stft computation

        metrics (list of functions): metrics (e.g., mae, mse, ..)

        metric_names (list of str): metric names corresponding to `metrics`

        losses (dict): {loss_name: loss_value} dictionary to save the result

        hop_length (None or int): hop length for stft

        weight (float): final weighting value
    """
    if mel_fb.shape[1] != n_fft // 2 + 1:
        raise ValueError('mel_fb.shape[0](=%d) != n_fft(=%d).. u sure?' % (mel_fb.shape[0], n_fft))

    def _log_melgram(src):
        win = torch.hann_window(n_fft).to(src.device)
        mag_specgrams = torch.stft(src, n_fft, hop_length, window=win).pow(2).sum(-1)  # (*, freq, time)

        melgrams = torch.matmul(mel_fb, mag_specgrams)  # (*, mel_freq, time)
        return time_freq.to_decibel(melgrams)

    melgram_true = _log_melgram(src_true)
    melgram_pred = _log_melgram(src_pred)

    for name, metric in zip(metric_names, metrics):
        losses['mel_' + name] = weight * metric(melgram_true, melgram_pred)
