import argparse


def str2bool(v):
    """robust boolean function for command line arg"""
    return v.lower() not in ('no', 'false', 'f', '0', False)


class ArgParser:
    """A custom class for parsing the arguments"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parser for input arguments')
        self.parser.register('type', 'bool', str2bool)
        self.add_args()

    def add(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def parse(self):
        args = self.parser.parse_args()
        for loss_domain in args.loss_domains:
            assert loss_domain in ('spectrum',
                                   'melgram',
                                   "l1_reg",
                                   'stft'), 'Wrong loss domain input: %s' % loss_domain

        assert args.kernel_size % 2 == 1, "kernel size should be an odd number but got %d" % args.kernel_size
        return args

    def add_args(self):
        print('Add arguments..')
        # experiment metadata
        self.add('-en', '--exp_name',
                 help='exp_name (that will be used for the result folder name)',
                 type=str, required=True)
        # model
        self.add('-ks', '--kernel_size',
                 help='any positive, odd integer, kernel size (or length) for the analysis U-net',
                 type=int, default=3)
        self.add('-nch', '--num_channel',
                 help='any positive integer, base number of channel in u-net/other layers',
                 type=int, default=50)
        self.add('-scale', '--scale_r',
                 help='any positive integer, scaling ratio (i.e. pooling size) of U-net',
                 type=int, default=2)
        self.add('-act', '--activation',
                 help='string, {relu, lrelu, elu}',
                 type=str, default='elu')
        self.add('-nle', '--n_layer_enc',
                 help="any positive integer, number of layer in unet encoder part",
                 type=int, default=10)
        self.add('-nld', '--n_layer_dec',
                 help='any positive integer, number of layer in unet decoder part',
                 type=int, default=6)
        self.add('-st', '--sparsemax_type',
                 help='string, to specify sparsemax type (or softmax type), '
                      'multiply, (in-parallel then multiplying)'
                      'double, (two sparse max in sequence)'
                      'softsparse (soft for inst, sparse for time)'
                      'softsoftseq (softmax for both, sequentially)'
                      'softsoftmul (softmax for both, multiplied, but it isn\'t trained well)',
                 type=str, default='multiply')
        self.add('-cb', '--conv_bias',
                 help='whether to have bias term in conv layers',
                 type=str2bool, default="false")
        self.add('-splst', '--sparsemax_lst',
                 help='any positive integer, length for sparsemax_time [sample]',
                 type=int, default=64)
        self.add('-rec', '--recurrenter',
                 help='string, to specify recurrent layer numbers (or to use conv instead)'
                      'three (for 3-layer rnn), '
                      'conv (for 3-layer conv)',
                 type=str, default="three")

        # training
        self.add('-bs', '--batch_size',
                 help='batch size',
                 type=int, default=32)
        self.add('-ld', '--loss_domains',
                 help='string, loss domains, e.g. spectrum(=cqt),'
                      'melgram, stft, l1_reg',
                 type=str, default=[],
                 required=True,
                 action='append')
        self.add('-n_mels', '--n_mels',
                 help='number of mel bands if melgram is in args.loss_domains',
                 type=int, required=False)
        self.add('-lr', '--learning_rate',
                 help='lr for adam. ',
                 type=float, default=0.0004,
                 required=False)
        # thor:  do 'single quote' everywhere, docstrings are """!
        # flake8 will help with this!
        self.add('-metrics', '--metrics',
                 type=str, default=[],
                 help='mae, mse',
                 action='append', required=True,
                 )
        self.add('-source_norm', '--source_norm',
                 help='no, abssum, sqrsum',
                 type=str, default='sqrsum')
        self.add('-lrl', '--l1_reg_lambda',
                 help='float, l1 regularizer coeff',
                 type=float, default=0.003)
        self.add('-n_cqt_bins', '--n_cqt_bins',
                 help='n_bins per octave in the cqters',
                 type=int, default=12)
        self.add('-hpss', '--compare_after_hpss',
                 help='whether to use hpss before comparing x and x_hat',
                 type=str2bool, default=False)
        self.add('-eval', '--eval', default='false',
                 help='if we do evaluation on some datasets (SMT, ENST, MDB..)',
                 type=str2bool)

        # resume
        self.add('-res', '--resume',
                 help='model full name (=folder name) if resuming, otherwise false. ALL the' +
                      ' arguments are ignored and overriden from what they originally were.',
                 type=str2bool, default='false')
        self.add('-resn', '--resume_num',
                 help='resume number',
                 type=str, default='',
                 required=False)
