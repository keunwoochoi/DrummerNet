"""
All the torch toys for time-frequency
"""
from globals import *
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

cqt_filter_fft = librosa.constantq.__cqt_filter_fft


def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True, length=None):
    """stft_matrix = (batch, freq, time, complex)

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    assert normalized == False
    assert onesided == True
    assert window == 'hann'
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

        ytmp = istft_window * iffted
        y[:, sample:(sample + n_fft)] += ytmp

    y = y[:, n_fft // 2:]

    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = torch.cat(y[:, :length], torch.zeros(y.shape[0], length - y.shape[1], device=y.device))

    coeff = n_fft / float(
        hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff


# def onset(x):
#     """
#     A simple magnitude-based onset detector
#
#     Args:
#         x (Tensor): (batch, freq, time)"""
#
#     def batch_diff(x):
#         return x[:, :, 1:] - x[:, :, :-1]
#
#     return F.relu(batch_diff(x))


class Melgramer():
    """A class for melspectrogram computation

    Args:
        n_fft (int): number of fft points for the STFT, based on which melspectrogram is computed

        hop_length (int): STFT hop length [samples]

        sr (int): sampling rate

        n_mels (int): number of mel bins

        fmin (float): minimum frequency of mel bins

        fmax (float): maximum frequency of mel bins

        power_melgram (float): the power of the STFT

        window (torch.Tensor): window function for `torch.STFT`

        log (bool): whether the result will be as log(melgram) or not

        dtype: the torch datatype for this melspectrogram
    """

    def __init__(self, n_fft=1024, hop_length=None, sr=22050, n_mels=128, fmin=0.0, fmax=None,
                 power_melgram=1.0, window=None, log=False, dtype=None):
        # thor:  is this correct python idiom for assert / what is the diff between this and raise?
        assert sr > 0
        assert fmin >= 0.0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin
        assert isinstance(log, bool)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = int(sr)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.log = log
        self.power_melgram = power_melgram
        self.window = window
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype

        self.mel_fb = nn.Parameter(self._get_mel_fb())

    def _get_mel_fb(self):
        """returns (n_mels, n_fft//2+1)"""
        return torch.from_numpy(librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                                    fmin=self.fmin, fmax=self.fmax)).type(self.dtype)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, waveforms):
        """x is perhaps (batch, freq, time).
        returns (batch, n_mel, time)"""
        mag_stfts = torch.stft(waveforms, self.n_fft,
                             hop_length=self.hop_length,
                             window=self.window).pow(2).sum(-1)  # (batch, n_freq, time)
        mag_stfts = torch.sqrt(mag_stfts + EPS)  # without EPS, backpropagating can yield NaN.
        # Project onto the pseudo-cqt basis
        mag_melgrams = torch.matmul(self.mel_fb, mag_stfts)
        if self.log:
            mag_melgrams = to_log(mag_melgrams)
        return mag_melgrams


# def _coo_to_float_tensor(coo_matrix):
#     """"""
#     values = np.abs(coo_matrix.data, dtype=NPDTYPE)
#     indices = np.vstack([coo_matrix.row, coo_matrix.col])
#
#     i = torch.LongTensor(indices).to(DEVICE)
#     v = torch.FloatTensor(values).to(DEVICE)
#     shape = coo_matrix.shape
#
#     return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class PseudoCqt():
    """A class to compute pseudo-CQT with Pytorch.
    API (+implementations) follows librosa
    (
    https://librosa.github.io/librosa/generated/librosa.core.pseudo_cqt.html
    )

    Usage:
        src, _ = librosa.load(filename)
        src_tensor = torch.tensor(src)
        cqt_calculator = PseudoCqt()
        cqt_calculator(src_tensor)

    """

    def __init__(self, sr=22050, hop_length=512, fmin=None, n_bins=84,
                 bins_per_octave=12, tuning=0.0, filter_scale=1,
                 norm=1, sparsity=0.01, window='hann', scale=True,
                 pad_mode='reflect'):

        if scale is not True:
            raise NotImplementedError('scale=False is not implemented')
        if window != 'hann':
            raise NotImplementedError('Window function other than hann is not implemented')

        if fmin is None:
            fmin = 2 * 32.703195  # note_to_hz('C2') because C1 is too low

        if tuning is None:
            tuning = 0.0  # let's make it simple

        fft_basis, n_fft, _ = cqt_filter_fft(sr, fmin, n_bins, bins_per_octave,
                                             tuning, filter_scale, norm, sparsity,
                                             hop_length=hop_length, window=window)

        self.fft_basis = torch.tensor(np.array(np.abs(fft_basis.todense())), dtype=TCDTYPE,
                                      device=DEVICE)  # because it was sparse. (n_bins, n_fft)

        self.fmin = fmin
        self.fmax = fmin * 2 ** (float(n_bins) / bins_per_octave)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.scale = scale
        win = torch.zeros((self.n_fft,), device=DEVICE)
        win[self.n_fft // 2 - self.n_fft // 8:self.n_fft // 2 + self.n_fft // 8] = torch.hann_window(self.n_fft // 4)
        self.window = win
        msg = 'PseudoCQT init with fmin:{}, {}, bins, {} bins/oct, win_len: {}, n_fft:{}, hop_length:{}'
        print(msg.format(int(fmin), n_bins, bins_per_octave, len(self.window), n_fft, hop_length))

    def __call__(self, y):
        return self.forward(y)

    def forward(self, y):
        # thor:  lowercase variable names
        mag_stfts = torch.stft(y, self.n_fft,
                               hop_length=self.hop_length,
                               window=self.window).pow(2).sum(-1)  # (batch, n_freq, time)
        mag_stfts = torch.sqrt(mag_stfts + EPS)  # without EPS, backpropagating through CQT can yield NaN.
        # Project onto the pseudo-cqt basis
        # C_torch = torch.stack([torch.sparse.mm(self.fft_basis, D_torch_row) for D_torch_row in D_torch])
        mag_melgrams = torch.matmul(self.fft_basis, mag_stfts)

        mag_melgrams /= torch.tensor(np.sqrt(self.n_fft), device=y.device)  # because `scale` is always True
        return to_log(mag_melgrams)


def to_log(mag_specgrams):
    """

    Args:
        mag_specgrams (torch.Tensor), non-power spectrum, and non-negative.

    """
    return (torch.log10(mag_specgrams + EPS) - torch.log10(torch.tensor(EPS, device=mag_specgrams.device)))


def to_decibel(mag_specgrams):
    """
    Args:
        mag_specgrams (torch.Tensor), non-power spectrum, and non-negative.

    """
    return 20 * to_log(mag_specgrams)


def log_stft(waveforms, n_fft, hop, center=True, mode='normal'):
    """
    if mode == 'high', window is hann(n_fft//4) with zero-padded.

    Args:
        waveforms (torch.Tensor): audio signal to perform stft

        n_fft (int): number of fft points

        hop (int): hop length [samples]

        center (bool): if stft is center-windowed or not

        mode (str): 'normal' or 'high'

    """
    assert mode in ('normal', 'high')
    if mode == 'normal':
        win = torch.hann_window(n_fft).to(waveforms.device)
    else:
        win = torch.zeros((n_fft,), device=waveforms.device)
        win[n_fft // 2 - n_fft // 8:n_fft // 2 + n_fft // 8] = torch.hann_window(n_fft // 4)
        assert hop <= (n_fft // 4), 'hop:{}, n_fft:{}'.format(hop, n_fft)
    #

    complex_stfts = torch.stft(torch.tensor(waveforms), n_fft, hop, window=win, center=center)
    mag_stfts = complex_stfts.pow(2).sum(-1)  # (*, freq, time)

    return to_log(mag_stfts)
