from globals import *
import torch
import inst_dataset


def get_instset_drum(norm: str):
    """helper function to instantiate InstSrcSet for drum

    # thor:  for sure do `drum_sources` over `drum_srcss`

    drum_srcss :torch tensor size of (N_DRUM_VSTS, N_DRUM_NOTES)
    """
    if norm not in ('no', 'abssum', 'sqrsum'):
        raise ValueError('Invalid normalization parameter!  Should be one of ("no", "abssum", "sqrsum")')

    if norm == 'no':
        normalizing_function = lambda x: 1
    elif norm == 'abssum':
        normalizing_function = lambda x: 400 * 1.0 / (x.abs().sum())
    elif norm == 'sqrsum':
        normalizing_function = lambda x: 10 * 1.0 / x.pow(2).sum().sqrt()

    notes = {key: [] for key in DRUM_NAMES}

    # thor:  maybe add why it is ignored?
    for i in range(1, N_DRUM_VSTS):  # the last drum vst is ignored because it's used in the test
        srcs = inst_dataset.load_drum_srcs(i)

        for src, key in zip(srcs, DRUM_NAMES):  # because the order is the same
            scale = normalizing_function(src)
            notes[key].append(scale * src)

    # thor:  `sources` is fine, imo
    for note_name in notes:
        srcs = notes[note_name]
        for i, src in enumerate(srcs):
            srcs[i] = torch.flip(src, dims=(0,))
        #
        notes[note_name] = tuple(notes[note_name])  # make them immutable
    #
    return InstrumentSourceSet(notes=notes)


class InstrumentSourceSet(object):
    """class to store inst source info"""

    def __init__(self, notes: dict, reverse: bool = True):
        """
        n_notes: number of notes
        """
        self.n_notes = len(notes)
        self.notes = notes.copy()
        self.note_names = list(self.notes.keys())
        self.reverse = reverse
        self.n_vsts = {k: len(self.notes[k]) for k in self.note_names}

    def __getitem__(self, key: str):
        return self.notes[key]

    def random_pick(self, key: str):
        return self.notes[key][np.random.choice(self.n_vsts[key])]

    def __str__(self):
        return ('n_notes: %d, n_vsts:' % self.n_notes) + str(self.n_vsts)

