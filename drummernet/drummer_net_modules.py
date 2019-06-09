import torch
import torch.nn as nn
import torch.nn.functional as F
import sparsemax
from globals import *
from fast_conv import fast_conv1d


class ValidAutoUnet(nn.Module):
    """
    A U-net class with valid convolution

    Args:
        args: arg from arg parser

        nnconv: nn.Conv1d, nn.Conv2d, nn.Conv3d.

        nnmp: nn.MaxPool1d, nn.MaxPool2d
    """

    def __init__(self, args, nnconv, nnmp):

        super(ValidAutoUnet, self).__init__()
        assert args.kernel_size % 2 == 1, ('kernel_size should be odd number but i have this -> %d' % args.kernel_size)
        self.num_channel = args.num_channel
        self.kernel_size = args.kernel_size
        self.padding = self.kernel_size // 2
        self.act = get_act_functional(args.activation)
        self.scale_r = args.scale_r
        self.n_layer_enc = args.n_layer_enc
        self.n_layer_dec = args.n_layer_dec
        self.sr_ratio = self.scale_r ** (self.n_layer_enc - self.n_layer_dec)  # sampling rate diff rate btwn x and z
        self.compress_ratio = self.scale_r ** self.n_layer_enc
        print('| With a sampling rate of %d Hz,' % SR_WAV)
        print('| the deepest encoded signal: 1 sample == %d ms.' % (1000.0 / SR_WAV * self.compress_ratio))
        print('| At predicting impulses, which is done at u_conv3, 1 sample == %d ms.' % (
                1000.0 / SR_WAV * self.sr_ratio))

        n_ch, k_size, pd = self.num_channel, self.kernel_size, self.padding
        pd = 0
        first_ch = min(128, n_ch)
        st = 1
        bias = args.conv_bias
        self.ch_out = 2 * n_ch
        # down conv
        self.d_conv0 = nnconv(1, first_ch, k_size, st, pd, bias=bias)

        self.d_convs = nn.ModuleList([nnconv(first_ch, n_ch, k_size, st, pd, bias=bias)] +
                                     [nnconv(n_ch, n_ch, k_size, st, pd, bias=bias) for _ in
                                      range(self.n_layer_enc - 1)])
        self.pools = nn.ModuleList([nnmp(self.scale_r) for _ in range(self.n_layer_enc)])
        # deepest conv
        self.encode_conv = nnconv(n_ch, n_ch, k_size, st, pd, bias=bias)
        # up conv
        self.u_convs = nn.ModuleList([nnconv(n_ch, n_ch, k_size, st, pd, bias=bias)] +
                                     [nnconv(2 * n_ch, n_ch, k_size, st, pd, bias=bias) for _ in
                                      range(self.n_layer_dec - 1)])
        # last conv
        self.last_conv = nnconv(2 * n_ch, self.ch_out, self.kernel_size, st, pd)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # Downconv 0
        x = self.act(self.d_conv0(x))
        # Downconv list
        xs = []
        for pool, conv in zip(self.pools, self.d_convs):
            x = conv(x)
            x = pool(self.act(x))
            xs.append(x)

        ys = []

        y_end = self.encode_conv(xs.pop())

        y = self.act(y_end)
        ys.append(y)
        for conv in self.u_convs:
            y = conv(y)
            # now, this is the smallest.
            y = self.act(y)
            y = F.interpolate(y, scale_factor=self.scale_r,
                              mode=int(y.dim() == 4) * 'bi' + 'linear', align_corners=False)
            x = xs.pop()
            crop = (x.shape[2] - y.shape[2]) // 2
            x = x[:, :, crop:-crop]
            y = torch.cat((y, x), dim=1)
            ys.append(y)

        r = self.last_conv(y)
        return self.act(r), xs, ys


class Convolver(nn.Module):
    """
    A convolution-based replacement for Recurrenter

    Args:
        input_size (int): number of input channel in the sequential conv layers

        hidden_size (int): number of conv layer channels
    """

    def __init__(self, input_size, hidden_size, args):
        super(Convolver, self).__init__()
        self.input_size = input_size
        n_component = hidden_size  # i.e., 11
        self.n_component = n_component
        self.act = get_act_functional(args.activation)
        ks = args.kernel_size
        n_ch = args.num_channel
        self.convs = nn.Sequential(nn.Conv1d(input_size, n_ch, ks, padding=ks // 2),
                                   nn.Conv1d(n_ch, n_ch, ks, padding=ks // 2),
                                   nn.Conv1d(n_ch, n_component, ks, padding=ks // 2, bias=False))

    def forward(self, x):
        return self.convs(x)


class Recurrenter(nn.Module):
    """
    Sequential recurrent layers for impulse predictions

    Args:
        input_size (int): (in the default case,) n_notes

        hidden_size (int): hidden vector size (which will be n_notes)
    """

    def __init__(self, input_size, hidden_size, args):
        super(Recurrenter, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act = get_act_functional(args.activation)
        self.midi_x2h = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                               batch_first=True, bidirectional=True,
                               bias=True)
        self.midi_h2hh = nn.GRU(input_size=2 * hidden_size, hidden_size=hidden_size,
                                batch_first=True, bidirectional=False,
                                bias=True)
        self.midi_hh2y = nn.GRU(input_size=1, hidden_size=1,
                                batch_first=True, bidirectional=False,
                                bias=False)

    def forward(self, r):
        """input: r (representation) """
        midis_in = r.transpose(1, 2)  # (b, t, ch), ready to rnn-ed
        midis_hi = self.act(self.midi_x2h(midis_in)[0])  # 1st RNN. (b, t, 2*n_insts)
        midis_hi = self.act(self.midi_h2hh(midis_hi)[0])  # 2nd RNN. (b, t, n_insts)

        b, time, n_insts = midis_hi.shape
        midis_hi = midis_hi.reshape((b * time), 1, n_insts)  # (b * t, 1, n_insts)
        midis_hi = midis_hi.transpose(1, 2)  # (b * t, n_insts, 1)

        midis_ou, _ = self.midi_hh2y(midis_hi)  # 3rd RNN

        midis_ou = midis_ou.transpose(1, 2).reshape(b, time, n_insts)  #
        return self.act(midis_ou.transpose(1, 2))  # (batch, n_insts, time)


class SoftSoftSeq(nn.Module):
    def __init__(self, lst=64):
        """softmax along inst, softmax along time
        they are """
        super(SoftSoftSeq, self).__init__()
        self.lst = lst
        self.softmax_inst = nn.Softmax(dim=1)  # along insts
        self.softmax_time = nn.Softmax(dim=-1)  # along time
        print('| and lst=%d samples at the same, at=`r` level' % self.lst)

    def forward(self, midis_ou):
        """midis_ou: (batch, n_insts, time)"""
        batch, n_insts, time = midis_ou.shape
        lst = self.lst
        len_pad = (lst - time % lst) % lst
        #
        midis_ou = F.pad(midis_ou, [0, len_pad])

        # midis_ou = self.softmax_inst(midis_ou.transpose(1, 2)).transpose(1, 2)  # inst-axis Sparsemax
        midis_ou = self.softmax_inst(midis_ou)

        midis_ou = midis_ou.reshape(batch, n_insts, (time + len_pad) // lst, lst)
        midis_ou = self.softmax_time(midis_ou)
        midis_ou = midis_ou.reshape(batch, n_insts, (time + len_pad))

        midis = midis_ou[:, :, :time]
        return midis


class SoftSoftMul(nn.Module):
    def __init__(self, lst=64):
        """softmax along inst, softmax along time
        they are """
        super(SoftSoftMul, self).__init__()
        self.lst = lst
        self.softmax_inst = nn.Softmax(dim=1)  # along insts
        self.softmax_time = nn.Softmax(dim=-1)  # along time
        print('| and lst=%d samples at the same, at=`r` level' % self.lst)

    def forward(self, midis_ou):
        """midis_ou: (batch, n_insts, time)"""
        batch, n_insts, time = midis_ou.shape
        lst = self.lst
        len_pad = (lst - time % lst) % lst
        #
        midis_ou = F.pad(midis_ou, [0, len_pad])

        midis_ou_i = self.softmax_inst(midis_ou.transpose(1, 2)).transpose(1, 2)  # inst-axis Sparsemax

        midis_ou_t = midis_ou.reshape(batch, n_insts, (time + len_pad) // lst, lst)
        midis_ou_t = self.softmax_time(midis_ou_t)
        midis_ou_t = midis_ou_t.reshape(batch, n_insts, (time + len_pad))

        midis = midis_ou_i[:, :, :time] * midis_ou_t[:, :, :time]
        return midis


class MultiplySparsemax(nn.Module):
    """Multiplication-based sparsemax layers.
    It compute sparsemax over time and channel, separately, then multiply their outputs

    Args:
        sparsemax_lst (int): the 'frame' length of sparsemax over time.
    """

    def __init__(self, sparsemax_lst=64):
        super(MultiplySparsemax, self).__init__()
        self.lst = sparsemax_lst
        self.sparsemax_inst = sparsemax.Sparsemax(dim=-1)  # along insts
        self.sparsemax_time = sparsemax.Sparsemax(dim=-1)  # along time
        print('| and sparsemax_lst=%d samples at the same, at=`r` level' % self.lst)

    def forward(self, midis_out):
        """midis_ou: (batch, n_insts, time)"""
        batch, n_insts, time = midis_out.shape
        lst = self.lst
        len_pad = (lst - time % lst) % lst
        #
        midis_out = F.pad(midis_out, [0, len_pad])

        midis_out_inst = self.sparsemax_inst(midis_out.transpose(1, 2)).transpose(1, 2)  # inst-axis Sparsemax

        midis_out_time = midis_out.reshape(batch, n_insts, (time + len_pad) // lst, lst)
        midis_out_time = self.sparsemax_time(midis_out_time)
        midis_out_time = midis_out_time.reshape(batch, n_insts, (time + len_pad))

        midis_final = midis_out_inst[:, :, :time] * midis_out_time[:, :, :time]
        return midis_final


class SequentialSparsemax(nn.Module):
    """Sequential sparsemax layers.
        It compute sparsemax over time and channel, separately, then multiply their outputs

        Args:
            sparsemax_lst (int): the 'frame' length of sparsemax over time.
    """

    def __init__(self, sparsemax_lst=64):
        super(SequentialSparsemax, self).__init__()
        self.lst = sparsemax_lst
        self.sparsemax_inst = sparsemax.Sparsemax(dim=-1)  # along insts
        self.sparsemax_time = sparsemax.Sparsemax(dim=-1)  # along time
        print('| and sparsemax_lst=%d samples at the same, at=`r` level' % self.lst)

    def forward(self, midis_out):
        """midis_out: (batch, n_insts, time)"""
        batch, n_insts, time = midis_out.shape
        lst = self.lst
        len_pad = (lst - time % lst) % lst

        midis_out = F.pad(midis_out, [0, len_pad])

        midis_out = self.sparsemax_inst(midis_out.transpose(1, 2)).transpose(1, 2)  # inst-axis Sparsemax

        midis_out = midis_out.reshape(batch, n_insts, (time + len_pad) // lst, lst)
        midis_out = self.sparsemax_time(midis_out)
        midis_out = midis_out.reshape(batch, n_insts, (time + len_pad))

        midis = midis_out[:, :, :time]
        return midis


class ZeroInserter(nn.Module):
    """Insert zeros for upsampling the transcription"""

    def __init__(self, insertion_rate):
        super(ZeroInserter, self).__init__()
        self.insertion_rate = insertion_rate

    def forward(self, downsampled_y):
        batch, ch, time = downsampled_y.shape
        upsampled_y = []
        for ch_idx in range(ch):
            ds_y = downsampled_y[:, ch_idx:ch_idx + 1, :]  # (batch, 1, time)
            us_y = torch.cat((ds_y,
                              torch.zeros((batch, self.insertion_rate - 1, time),
                                          device=downsampled_y.device)),
                             dim=1)  # (batch, insert_rate, time)
            us_y = us_y.transpose(2, 1)  # (b, t, insert_rate)
            us_y = torch.reshape(us_y, (batch, 1, self.insertion_rate * time))  # (b, 1, t*insert_rate)
            upsampled_y.append(us_y)

        upsampled_y = torch.cat(upsampled_y, dim=1)
        return upsampled_y


class FastDrumSynthesizer(nn.Module):
    """Freq-domain convolution-based drum synthesizer"""

    def __init__(self, n_notes, drum_srcset):
        super(FastDrumSynthesizer, self).__init__()
        self.drum_srcset = drum_srcset
        self.n_notes = n_notes

    def forward(self, midis):
        """
        midis: (batch, inst, time)
        teturned tracks: (batch, inst, time)
        """
        device_ = midis[0].device

        rv_insts = [self.drum_srcset.random_pick(note_name).to(device_) for note_name in DRUM_NAMES]

        igw = [1. / rv_i.abs().sum() for rv_i in rv_insts]
        igw = torch.tensor(igw) / max(igw)

        tracks = []
        for i in range(self.n_notes):
            md = midis[:, i:i + 1, :]
            track = fast_conv1d(md, torch.flip(rv_insts[i].expand(1, 1, -1), dims=(2,)))
            tracks.append(track)

        return torch.cat(tracks, dim=1)


class Mixer(nn.Module):
    """Sum mixer"""

    def forward(self, tracks, group_by=None):
        """tracks: (batch, inst, time)
        return: (batch, time)"""
        if group_by:
            return tracks[:, group_by, :].sum(dim=1)
        else:
            return tracks.sum(dim=1)


def get_act_module(act_name):
    act_name = act_name.lower()
    if act_name == 'relu':
        return nn.ReLU
    elif act_name == 'elu':
        return nn.ELU
    elif act_name in ('lrelu', 'leakyrelu'):
        return nn.LeakyReLU
    else:
        raise RuntimeError('Wrong activation:', act_name)


def get_act_functional(act_name):
    act_name = act_name.lower()
    if act_name == 'relu':
        return F.relu
    elif act_name == 'elu':
        return F.elu
    elif act_name in ('lrelu', 'leakyrelu'):
        return F.leaky_relu
    else:
        raise RuntimeError('Wrong activation:', act_name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
