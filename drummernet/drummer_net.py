import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import *
from inst_src_set import InstrumentSourceSet
import drummer_net_modules
from drummer_net_modules import ValidAutoUnet, FastDrumSynthesizer, Recurrenter, SequentialSparsemax
from drummer_net_modules import MultiplySparsemax, ZeroInserter, Mixer
from drummer_net_modules import SoftSoftMul, SoftSoftSeq, Convolver


class DrummerNet(nn.Module):
    def __init__(self, inst_srcs: torch.tensor,
                 inst_names: list,
                 drum_srcset: InstrumentSourceSet,
                 args):
        """
        inst_srcs: 2d torch tensor, instrument waveforms to use in the resynthesis
        """
        super(DrummerNet, self).__init__()
        self.test_inst_srcs = inst_srcs
        self.test_inst_names = inst_names  #

        self.drum_srcset = drum_srcset
        self.n_notes = self.drum_srcset.n_notes  # 11

        self.sparsemax_lst = args.sparsemax_lst
        n_ch = args.num_channel
        n_ch_repre = 2 * n_ch  # number of channel of $r$

        if args.sparsemax_type == 'double':
            SpMax = SequentialSparsemax
        elif args.sparsemax_type == 'multiply':
            SpMax = MultiplySparsemax
        elif args.sparsemax_type == 'softsoftseq':
            SpMax = SoftSoftSeq
        elif args.sparsemax_type == 'softsoftmul':
            SpMax = SoftSoftMul
        else:
            raise NotImplementedError

        if args.recurrenter == 'three':
            Rec = Recurrenter
        elif args.recurrenter == 'conv':
            Rec = Convolver
        else:
            raise NotImplementedError

        self.unet = ValidAutoUnet(args, nn.Conv1d, nn.MaxPool1d)
        self.recurrenter = Rec(n_ch_repre, hidden_size=self.n_notes, args=args)
        self.double_sparsemax = SpMax(self.sparsemax_lst)
        self.zero_inserter = ZeroInserter(insertion_rate=self.unet.sr_ratio)
        self.synthesizer = FastDrumSynthesizer(self.n_notes, self.drum_srcset)
        self.mixer = Mixer()
        print(drum_srcset)
        print(self)
        print('')
        print('NUM_PARAM overall:', drummer_net_modules.count_parameters(self))
        print('             unet:', drummer_net_modules.count_parameters(self.unet))
        print('      recurrenter:', drummer_net_modules.count_parameters(self.recurrenter))
        print('       sparsemaxs:', drummer_net_modules.count_parameters(self.double_sparsemax))
        print('      synthesizer:', drummer_net_modules.count_parameters(self.synthesizer))

    def forward(self, x):
        """x: (batch, time)
        (unet unsqueezes the input x)

        """
        nsp_src = x.shape[1]
        div = self.unet.compress_ratio
        nsp_pad = (div - (nsp_src % div)) % div
        if nsp_pad != 0:
            x = F.pad(x, (0, nsp_pad))

        r, _, _ = self.unet(x)
        dense_y = self.recurrenter(r)
        sparse_y = self.double_sparsemax(dense_y)
        y_hat = upsampled_y = self.zero_inserter(sparse_y)
        tracks = self.synthesizer(y_hat)
        x_hat = est_mix = self.mixer(tracks)

        trimmed = (x.shape[1] - x_hat.shape[1]) // 2
        x_trimmed = x[:, trimmed: -trimmed]
        return x_trimmed, x_hat, y_hat
