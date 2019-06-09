import os
import torch
from globals import *
from torch.utils.data import Dataset, DataLoader
import librosa
import madmom
import sys
from util_etc import dcnp


# thor:  use more letters in names! : )
def load_drum_srcs(idx=1):
    """Returns a list of torch.tensor"""
    dura = 1.0  # s

    srcs = []
    for drum_note_name in DRUM_NAMES:
        filename = '%d)%s.wav' % (idx, drum_note_name)
        src, _ = librosa.load(os.path.join(DATA_PATH, filename), sr=SR_WAV, duration=dura)
        srcs.append(src)
    #
    inst_srcs = torch.from_numpy(np.array(srcs))
    return inst_srcs


class TxtSrcDataset(Dataset):
    """textfile-based source dataset

    Args:
        txt_path (str): text file path

        src_path (str): audio file path

        duration (float): length of audio signal for each items to load

        ext (str): extension of the audio file in interest
    """
    def __init__(self, txt_path, src_path, duration, sr_wav, ext='mp3'):
        super(TxtSrcDataset, self).__init__()
        self.txt_path = txt_path
        self.src_path = src_path
        self.duration = duration
        self.sr_wav = sr_wav
        self.ext = ext
        self.lines = []
        self._read_txt()

    def _read_error(self, size):
        raise NotImplementedError()

    def _read_txt(self):
        raise NotImplementedError()

    def _read_audio(self, path, duration, file_dura):
        raise NotImplementedError()

    def _line_to_readpath(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        path, file_dura = self._line_to_readpath(idx)
        mix = self._read_audio(path, duration=self.duration, file_dura=file_dura)
        return torch.from_numpy(mix), torch.zeros(mix.shape), torch.zeros(mix.shape)


class TxtDrumstemDataset(TxtSrcDataset):
    """textfile-based datast but for drum stems
    """
    def __init__(self, *args, **kwargs):
        super(TxtDrumstemDataset, self).__init__(*args, **kwargs)

    def _read_error(self, size):
        return np.random.uniform(-0.01, 0.01, size=size).astype(NPDTYPE)

    def _read_txt(self):

        with open(self.txt_path) as f_read:
            for idx, line in enumerate(f_read):
                if int(line.rstrip('\n').split('\t')[1]) > self.duration + 1:
                    self.lines.append(line.rstrip('\n'))

    def _read_audio(self, path, duration, file_dura):
        start = np.random.choice(int(file_dura - duration))  # [second]
        try:
            src, _ = madmom.io.audio.load_audio_file(path, sample_rate=SR_WAV, dtype=NPDTYPE, num_channels=1,
                                                     start=start, stop=start + duration)

            if len(src) < int(SR_WAV * duration):
                return np.concatenate(
                    (src, np.random.uniform(-0.01, 0.01, (NSP_SRC - len(src))).astype(NPDTYPE)), axis=0)
            return librosa.util.normalize(src, axis=0)

        except Exception as e:
            sys.stderr.write('AUDIO READ ERROR (%s): %s\n' % (path, e))
            return self._read_error(size=(int(SR_WAV * duration),))

    def _line_to_readpath(self, idx):
        filename, file_duration = self.lines[idx].split('\t')
        return '%s/%s' % (self.src_path, filename), float(file_duration)


class InstDataset(Dataset):
    """A class that generates an audio file with notes at random gains/timings.
    This is used for sanity check of the system, not training (it could be though and somehow works.)

    Args:
        seed (int): random seed for this dataset

        sr_wav (int): sampling rate being used in this exp.

        srcs (list of torch.Tensors): len(srcs) == number of instrument set. (e.g. 12 if there are 12 diff drum sets)
            srcs[0].shape[0] : number of insts (e.g. 3 if we do

        n_item (int): the num of items this dataset is gonna output

        reset_at_end (bool): if true, after every epoch, all idxs are reset.

        active_rate (float in [0.0, 1.0]): if 1.0, 100% all notes/components exist.
            Otherwise, it's like probabilistically (1-active_rate) are ignored.

    """

    def __init__(self, srcs, sr_wav, seed=None, n_item=128, reset_at_end=False, active_rate=1.0):

        self.reset_at_end = reset_at_end
        src_shape = srcs[0].shape
        for src in srcs:
            if src_shape != src.shape:
                raise TypeError('Shape mismatch, {} vs {}'.format(src_shape, src.shape))

        self.srcs = [dcnp(src) for src in srcs]
        self.n_insts = self.srcs[0].shape[0]  # 8 for full drums
        self.nsp_inst = self.srcs[0].shape[1]  # thor: num_samples?
        self.n_inst_set = len(srcs)

        self.seed = seed
        self.n_item = n_item
        self.active_rate = active_rate

        self.g_steps = 10  # gain steps
        self.t_steps = 10  # timing steps

        self.t_gap = sr_wav // 6  # thor:  timing_gap, imo!

        self.timings = [self.t_gap * i for i in range(self.t_steps)]
        self.gains = [np.sqrt(2, dtype=NPDTYPE) ** (0.5 * (1.0 + i - self.g_steps)) for i in range(self.g_steps)]

        self.pre_zero = self.post_zero = self.t_gap * 2
        self.nsp_src = self.pre_zero + (self.t_steps - 1) * self.t_gap + self.nsp_inst + self.post_zero

        if seed is not None:
            np.random.seed(self.seed)
            torch.random.manual_seed(seed)
        # np.random.seed(self.seed)
        self.g_idxs, self.t_idxs = None, None
        self.s_idxs = None  # instrument set index
        self.reset_idxs()

    def reset_idxs(self):
        self.g_idxs = np.random.choice(self.g_steps, size=(self.n_item, self.n_insts))
        self.t_idxs = np.random.choice(self.t_steps, size=(self.n_item, self.n_insts))
        self.s_idxs = np.random.choice(self.n_inst_set, size=(self.n_item, self.n_insts))

    def __len__(self):
        return self.n_item

    def __getitem__(self, idx):
        """outputs:
        all outputs are torch tensor but on CPU. will be moved to GPU when they're used.

        """
        if idx == self.n_item:
            idx = idx % self.n_item
            if self.reset_at_end:
                self.reset_idxs()

        srcs = np.zeros((self.n_insts, self.nsp_src), dtype=NPDTYPE)
        t_idxs, g_idxs = self.t_idxs[idx], self.g_idxs[idx]
        s_idxs = self.s_idxs[idx]

        # make sources and irs # thor:  what is an ir / better name?
        impulses = torch.from_numpy(np.zeros((self.n_insts, self.nsp_src)))

        activeness = np.random.ranf(size=self.n_insts) < self.active_rate
        for inst_i, (t_idx, g_idx, s_idx) in enumerate(zip(t_idxs, g_idxs, s_idxs)):
            gain_here = activeness[inst_i] * self.gains[g_idx]
            timing = self.pre_zero + self.timings[t_idx]
            inst_src = self.srcs[s_idx]
            srcs[inst_i, timing: timing + self.nsp_inst] += inst_src[inst_i] * gain_here

            impulses[inst_i, timing] = gain_here

        srcs = torch.from_numpy(srcs)
        mix = torch.sum(srcs, dim=0, keepdim=False)  # (self.sp_src)

        return mix, srcs, impulses  # later it will be (batch, time) and (batch, n_insts, time)


class DrumDataset(InstDataset):
    """
    instset_idxs: tuple of the indices (integer) that this dataset will use.
            For training set, use something like (1, 2, 3, 4, 5)
            For test set, maybe something like (6)
    """

    def __init__(self, instset_idxs, **kwargs):
        srcs = [load_drum_srcs(idx=i) for i in instset_idxs]
        super(DrumDataset, self).__init__(srcs, **kwargs)
