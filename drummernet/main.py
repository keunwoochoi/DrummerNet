from globals import *
import os
import torch
import inst_dataset
from drummernet_trainer import DrummerNetTrainer
from torch.utils.data import DataLoader
from drummer_net import DrummerNet
from inst_src_set import get_instset_drum
import argparser

import warnings

warnings.filterwarnings('ignore', module='matplotlib')

torch.multiprocessing.set_start_method('spawn', force=True)


def main(args):
    """Main body of training procedure

    Args:
        args (ArgParse): arguments for the model and the training details

    """
    # *** set instrument sounds to feed to the drummer_net
    inst_srcs = inst_dataset.load_drum_srcs(idx=N_DRUM_VSTS)
    inst_names = DRUM_NAMES

    drummer_net = DrummerNet(inst_srcs, inst_names,
                             get_instset_drum(norm=args.source_norm), args)

    drummer_net = drummer_net.to(DEVICE)

    tr_batch_size = args.batch_size
    trainer = DrummerNetTrainer(drummer_net, args=args)
    trainer.prepare(args)

    # *** set which instruments to get
    n_epochs = [1] * 18
    n_train_items = [tr_batch_size * 2 ** i for i in range(18)]
    items_checks = [i * j for (i, j) in zip(n_train_items, n_epochs)]
    print('item check-points after this..:', items_checks)
    print('total %d n_items to train!' % (sum(items_checks)))

    tr_params = {'batch_size': tr_batch_size, 'shuffle': True, 'num_workers': tr_batch_size}

    for (n_epoch, n_train_item) in zip(n_epochs, n_train_items):
        drumstem_dataset = inst_dataset.TxtDrumstemDataset(
            txt_path=os.path.join(DRUMSTEM_PATH, 'files.txt'),
            src_path=DRUMSTEM_PATH, duration=DURATION, sr_wav=SR_WAV, ext=None)

        train_loader = DataLoader(drumstem_dataset, **tr_params)
        # training and eval
        trainer.train_many_epochs(n_epoch, train_loader, n_train_item)

    # after training
    print('ALL DONE!')


if __name__ == '__main__':
    my_arg_parser = argparser.ArgParser()
    args = my_arg_parser.parse()

    print(args)
    main(args)
