import os
import torch
import numpy as np

if 'mac' in os.uname()[1]:  # if you're not using mac, you might need to change it.
    LAPTOP, SERVER = True, False
    N_WORKERS = 3
else:
    LAPTOP, SERVER = False, True
    import matplotlib as mpl

    mpl.use('Agg')
    N_WORKERS = 128

LINEWIDTH = 60
EPS = 10e-6

SR = 16000
SR_WAV = SR
DURATION = 2
NSP_SRC = SR * DURATION
N_FFT = 1024
HOP = 1024 // 2

HOME = os.getenv('HOME')
EVALDATA_PATH = os.path.join('../data_evals')
DATA_PATH = '../data_drum_sources'
DRUMSTEM_PATH = os.path.join('../data_drumstems')

USE_CUDA = torch.cuda.is_available()

torch.set_default_tensor_type('torch.FloatTensor')

if USE_CUDA:
    device_ids = [i for i in range(torch.cuda.device_count())]
    DEVICE = torch.device('cuda:0')
    N_DEVICE = len(device_ids)
    if torch.cuda.device_count() > 1:
        MULTI_GPU = True
    else:
        MULTI_GPU = False
else:
    DEVICE = 'cpu'
    N_DEVICE = 1
    MULTI_GPU = False


NPDTYPE = np.float32
TCDTYPE = torch.get_default_dtype()

DRUM_NAMES = ['KD_KD', 'SD_SD', 'HH_CHH', 'HH_OHH', 'HH_PHH', 'TT_HIT', 'TT_MHT',
              'TT_HFT', 'CY_RDC', 'CY_CRC', 'OT_TMB']
N_DRUM_VSTS = 12