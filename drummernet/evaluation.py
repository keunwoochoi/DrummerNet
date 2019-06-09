# copied from
# https://github.com/cristiprg/drum_gans/blob/a5c95b7278afcd83c7843065328672023020980b/data/import_smt.py
# thanks! much love!!!!

from globals import *
import numpy as np
import mir_eval
import librosa
from librosa import display

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings(action='once')


def process_annotation(txtpath_r, txtpath_w, label_map, delimiter='\t'):
    """
    Args:
        txtpath_r (str): path to load the annotation

        txtpath_w (str): path to write

        label_map (dict): a dict, like {0:'KD', 1:'SD'} to process, for example

        delimiter (str): delimiter between events
    """
    with open(txtpath_r, 'r') as f_r:
        with open(txtpath_w, 'w') as f_w:
            for line in f_r:

                t, old_label = line.rstrip('\n').split(delimiter)
                t, old_label = t.strip(), old_label.strip()
                if label_map is not None:
                    new_label = label_map[old_label]
                else:
                    new_label = old_label
                f_w.write(delimiter.join([t, new_label]))
                f_w.write('\n')


def read_annotations_multilabel(onsets_tuple):
    """
    Args:
        onsets_tuple (tuple): (list_of_time, list_of_label),
            which is a result from mir_eval.io.load_labeled_events(txtpath_r)

    Returns:
        onsets_dict (dict): {label: np.array_of_time} for all the labels
    """
    labels = set(onsets_tuple[1])
    onsets_dict = {k: [] for k in labels}
    for t, label in [list(i) for i in zip(*onsets_tuple)]:
        onsets_dict[label].append(t)

    for key in onsets_dict:
        onsets_dict[key] = np.array(onsets_dict[key])
    return onsets_dict


def rename_key(old_dict, key_map=None):
    """rename the key in old_dict according to the mapping in `key_map`"""
    if key_map is None:
        return old_dict
    for old_key in old_dict:
        if old_key in key_map and old_key != key_map[old_key]:
            new_key = key_map[old_key]
            old_dict[new_key] = old_dict[old_key]
            del old_dict[old_key]

    return old_dict


def pickpeak_fix(impulse):
    """
    peak-picking using librosa

    Args:
        impulse (np.array): 1-d numpy array

    """
    div_max, div_avg, div_wait, div_thre = 20, 10, 16, 4

    impulse /= impulse.max()
    peak_idxs = librosa.util.peak_pick(impulse,
                                       SR // div_max, SR // div_max,
                                       SR // div_avg, SR // div_avg,
                                       1.0 / div_thre,
                                       SR // div_wait)

    return librosa.samples_to_time(peak_idxs, sr=SR)  # [0] to make it (1, N) to (N)


class DrumDatasetFolder():
    """Scan a folder that has a structure of
        ```
        root\annotations
            \audio
        ```

    Example:
        ```
        ddf = DrumDatasetFolder(args)
        ddf_iter = iter(ddf)
        for src, onsets_dict in ddf_iter:
            pred = some_network(src)
            some_evaluate(pred, onsets_dict)
        ```

    """

    def __init__(self, path_read, name, label_map=None, ann_folder='annotations', audio_folder='audio'):
        """

        Args:
            path_read (str): path to read the data from

            name (str): dataset name, e.g., 'smt'

            label_map (dict): {'0': 'KD'} for example. I used it to have consistent names using
                ('KD', 'SD', 'HH') instead of ('0', '1', '2')

            ann_folder (str): the sub-folder name that has annotation text files

            audio_folder (str): the sub-folder name that has audio files
        """
        self.name = name
        self.path_read = path_read
        self.label_map = label_map
        self.anno_fns, self.audio_fns = [], []
        self.ann_folder = ann_folder
        self.audio_folder = audio_folder
        self._scan_files()

    def _scan_files(self):
        """load annotations """

        anno_fns = os.listdir(os.path.join(self.path_read, self.ann_folder))
        anno_fns = [f for f in anno_fns if f.endswith('.txt')]
        self.anno_fns = sorted(anno_fns)

        audio_fns = os.listdir(os.path.join(self.path_read, self.audio_folder))
        audio_fns = [f for f in audio_fns if f.endswith('.wav')]
        self.audio_fns = sorted(audio_fns)
        assert len(self.anno_fns) == len(self.audio_fns), \
            'The number of files should be equal but %d and %d' % (len(self.anno_fns), len(self.audio_fns))
        self.n_files = len(self.audio_fns)

    def __iter__(self):
        self.n = 0
        return self

    def __len__(self):
        return self.n_files

    def __next__(self):
        """

        returns a tuple (src, onsets_dict)

        Returns:
            src (np.array): numpy 1d array, the audio data.
            onsets_dict (dict):
        """
        if self.n < self.n_files:
            anno_fn = self.anno_fns[self.n]
            audio_fn = self.audio_fns[self.n]
            # print(anno_fn, audio_fn, '...')
            src, _ = librosa.load(os.path.join(
                self.path_read, self.audio_folder, audio_fn
            ), sr=SR)
            onsets_tuple = mir_eval.io.load_labeled_events(
                os.path.join(self.path_read, self.ann_folder, anno_fn)
            )
            onsets_dict = read_annotations_multilabel(onsets_tuple)
            onsets_dict = rename_key(onsets_dict, self.label_map)
            self.n += 1
            return src, onsets_dict
        else:
            raise StopIteration


class DrumEvaluator(object):
    """
    Evaluate a drummernet
    """

    def __init__(self, drummer_net, ddf, device='cpu'):
        self.device = device
        self.drummer_net = drummer_net.to(self.device)
        self.ddf = ddf
        self.component_names = ['KD', 'SD', 'HH']
        # atts - constants
        self.lst = 1024
        self.max_nsp = self.lst * 300
        self.reset_data()
        self.midis = [None] * len(self.ddf)
        self.est_onsets = [None] * len(self.ddf)
        self.ref_onsets = [None] * len(self.ddf)
        # self.img_folder = img_folder
        # os.makedirs(self.img_folder, exist_ok=True)
        self.reset_data()

    def reset_data(self):
        self.ndc = self.n_drum_components = len(self.component_names)
        self.f_scores = {k: [] for k in self.component_names}

    def print_result(self, brief=False):
        if brief:
            print('F1 scores: ', end='')
            f1s = []
            for key in self.f_scores:  # key: kd, sd, hh
                songs_score = np.array(self.f_scores[key])  # (n_songs, 3=len([f1, p, r]))
                f1 = np.mean(songs_score, axis=0)[0]
                f1s.append(f1)
                print('{}'.format(np.round(f1, decimals=3)), end='  ')
            print(', mean:{}'.format(np.round(np.mean(f1s), decimals=3)), end='  ')
            print('')
        else:
            if self.f_scores != {}:
                print('Means of F/P/R, Stds of F/P/R')
                for key in self.f_scores:
                    songs_score = np.array(self.f_scores[key])
                    print(key, np.mean(songs_score, axis=0), np.std(songs_score, axis=0))
            else:
                print('self.f_scores is blank, so nothing to print.')

    def predict(self, verbose=False):
        """Run the drummernet to get the prediction (transcription estimation)"""

        def send_pred_reduce(src):
            pad = 4080  # the right value for the valid-conv model
            src = np.concatenate([np.zeros(pad, ), src, np.zeros(pad, )], axis=0)
            src = torch.tensor(src[np.newaxis, :], dtype=TCDTYPE).to(self.device)
            ret = self.drummer_net.forward(src)  # [(batch_size1, ch=1, time)] * n_drum_component
            est_irs = ret[2]  # y_hat
            est_irs = est_irs[0].detach().cpu().numpy()  # (n_inst, time)
            return np.stack([est_irs[0], est_irs[1], est_irs[2:4].sum(axis=0)], axis=0).astype(np.float32)

        ddf_iter = iter(self.ddf)
        if verbose:
            bar = tqdm(enumerate(ddf_iter), total=len(self.ddf), desc='predicting..')
        else:
            bar = enumerate(ddf_iter)

        for song_idx, (src, onsets_dict) in bar:
            # prepare - make it multiple of lst
            len_pad = (self.lst - len(src) % self.lst) % self.lst
            if len_pad != 0:
                src = np.concatenate((src, np.zeros(len_pad, )), axis=0)
            src = src.astype(NPDTYPE)
            src = src / np.abs(src).max()  # yes.. gotta normalize it by ourselves

            # Do the prediction
            if len(src) >= self.max_nsp:
                has_residual = (len(src) % self.max_nsp) != 0
                midis = np.zeros((self.ndc, 0), dtype=np.float32)
                for i in range(len(src) // self.max_nsp + int(has_residual)):
                    sub_midis = send_pred_reduce(src[i * self.max_nsp: (i + 1) * self.max_nsp])
                    midis = np.concatenate((midis, sub_midis), axis=1)
            else:
                midis = send_pred_reduce(src)
            self.midis[song_idx] = midis.astype(np.float32)

    def pickpeaks(self, pp_func, verbose=False, **kwargs):
        """Do the peak-picking

        Args:
            pp_func (function): peak-picking function

            verbose (bool): whether update the tqdm bar with progress or not

        """
        ddf_iter = iter(self.ddf)
        if verbose:
            bar = tqdm(enumerate(ddf_iter), total=len(ddf_iter), desc='picking peaks...')
        else:
            bar = enumerate(ddf_iter)

        for song_idx, (src, onsets_dict) in bar:
            est_onset_song = []
            ref_onset_song = []
            for i, key in zip(range(self.ndc), self.component_names):
                est_onset = pp_func(self.midis[song_idx][i], key, **kwargs)  # onset positions
                est_onset_song.append(est_onset)

                if key in onsets_dict:
                    ref_onset = onsets_dict[key]
                else:
                    ref_onset = np.array([])
                ref_onset_song.append(ref_onset)
            #
            self.est_onsets[song_idx] = np.array(est_onset_song)
            self.ref_onsets[song_idx] = np.array(ref_onset_song)

    def mir_eval(self):
        """run mir_eval for the final performance measure"""
        self.reset_data()
        for ref_onset, est_onset in zip(self.ref_onsets, self.est_onsets):
            for i, key in enumerate(self.component_names):
                f_score = mir_eval.onset.f_measure(ref_onset[i], est_onset[i])  # F, P, R
                self.f_scores[key].append(f_score)

    # thor:  maybe move the plots and prints to their own file?
    def illustrate_one(self, song_idx, img_folder, verbose=False):
        midis = self.midis[song_idx]
        est_onset = self.est_onsets[song_idx]
        ref_onset = self.ref_onsets[song_idx]
        if midis is None:
            if verbose:
                print('none...')
            return None

        # FIGURE 1
        plt.figure(figsize=(15, 3))
        for i in range(3):
            plt.subplot(3, 3, i + 1)
            display.waveplot(midis[i])
            plt.title(self.component_names[i] + ' est_irs')
            if i == 0:
                plt.title(self.component_names[i] + ' est_irs ' + str(song_idx) + ' ' + self.ddf.audio_fns[song_idx])

        for i, key in zip(range(self.ndc), self.component_names):  # KD, SD, HH
            # FIGURE 2
            plt.subplot(3, 3, i + 4)
            tmp = np.zeros_like(midis[i])
            np.put(tmp, librosa.time_to_samples(est_onset[i], sr=SR), np.ones(len(est_onset[i])))
            display.waveplot(tmp)
            plt.title('after peak picking')
            # FIGURE 3
            plt.subplot(3, 3, i + 7)
            tmp = np.zeros_like(midis[i])
            np.put(tmp, librosa.time_to_samples(ref_onset[i], sr=SR), np.ones(len(ref_onset[i])))
            display.waveplot(tmp)
            plt.title('reference')
            plt.savefig(os.path.join(img_folder + '/' + self.ddf.anno_fns[song_idx] + '.png'))
            if verbose:
                print('-%s: %3.0d %3.0d' % (key, len(ref_onset[i]), len(est_onset[i])), end='   ')
        if verbose:
            print('')

    def illustrate(self, img_folder):
        bar = tqdm(range(len(self.ddf)), total=len(self.ddf), desc='drawing..')
        for song_idx in bar:
            self.illustrate_one(song_idx, img_folder)


def get_ddf_smt():
    """smt dataset loading function"""
    ddf_smt = DrumDatasetFolder(
        os.path.join(EVALDATA_PATH, 'SMT_DRUMS'), 'smt',
        label_map=None,
        ann_folder='annotations',
        audio_folder='audio'
    )
    return ddf_smt


def get_ddf_enst():
    """enst dataset loading function"""
    ddf_enst = DrumDatasetFolder(
        os.path.join(EVALDATA_PATH, 'ENST_DTP(wet_mix-minus_one)'), 'enst',
        label_map={'0': 'KD', '1': 'SD', '2': 'HH'},
        ann_folder='annotations',
        audio_folder='audio'
    )
    return ddf_enst


def get_ddf_mdb():
    ddf_mdb = DrumDatasetFolder(
        os.path.join(EVALDATA_PATH, 'MDB_Drums'), 'mdb',
        label_map=None,
        ann_folder='annotations/class',
        audio_folder='audio/drum_only'
    )
    return ddf_mdb
