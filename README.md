# DrummerNet

This is supplementary material of "Deep Unsupervised Drum Transcription" by Keunwoo Choi and Kyunghyun Cho, ISMIR 2019 (Delft, Netherland). 

[Paper on arXiv](https://arxiv.org/abs/1906.03697) | [Blog post](https://keunwoochoi.wordpress.com/2019/06/11/drummernet-deep-unsupervised-drum-transcription/)   

* What we provide: Pytorch implementation for the paper 
* What we do **not** provide:
  - pre-trained model
  - drum stems that we used for the training

## Installation

If you're using conda and wanna run it DrummerNet CPU, make sure it installs mkl because we'll need its fft module.
```bash
conda install -c anaconda mkl
```
Then,
```bash
pip install -r requirements.txt
```

Using conda, it would be something like this, but customize it yourself!
```bash
conda install -c pytorch pytorch torchvision 
```

`Python3` required.

## Preparation
#### Wav files for Drum Synthesizer
 * `data_drum_sources`: folder for isolated drum sources. 12 kits x 11 drum components are included.
 If you want to add more drum sources,
 
   - Add files and update `globals.py` accordingly. 
    ```python
    # These names are matched with file names in data_drum_sources
    DRUM_NAMES = ["KD_KD", "SD_SD", "HH_CHH", "HH_OHH", "HH_PHH", "TT_HIT", "TT_MHT",
                  "TT_HFT", "CY_RDC", "CY_CRC", "OT_TMB"]
    N_DRUM_VSTS = 12
    ```
   - Note that as shown in `inst_src_sec.get_instset_drum()`, the last drum kit will be used in the test time only. 

#### Training files
We unfortunately **cannot** provide the drum-stems that we used for the trained network in the paper.
 * `/data_drumstems`: nearly blank folder, placeholder for training data. I put one wav file and `files.txt` as an minimum working example.
 * [Mark Cartwright](http://dafx2018.web.ua.pt/papers/DAFx2018_paper_60.pdf)'s and [Richard Vogl](https://arxiv.org/abs/1806.06676)'s papers/codes provide a way to synthesize large-scale drum stems   

#### Evaluation files, e.g., SMT
 * It is not part of the code, you have to download/process it by yourself.
 * First, [download SMT dataset](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/drums.html) (320.7MB)
 * Unzip it. Let's call the unzipped folder PATH_UNZIP
 * Then run `$ python3 drummernet/eval_import_smt.py PATH_UNZIP`. E.g.,
    ```bash
   $ cd drummernet
   $ python3 eval_import_smt.py ~/Downloads/SMT_DRUMS/
   Processing annotations...
   Processing audio file - copying it...
    all done! check out if everything's fine at data_evals/SMT_DRUMS
   ```  
 * `data_evals`: blank, placeholder for evaluation datasets

## Training

 * If you prepared evaluation files
```
python3 main.py --eval false -ld spectrum --exp_name temp_exp --metrics mae
```
 * Otherwise,
```
python3 main.py --eval true -ld spectrum --exp_name temp_exp --metrics mae
```

If everything's fine, you'll see..
```bash
$ cd drummernet
$ python3 main.py --eval True -ld spectrum --exp_name temp_exp --metrics mae
Add arguments..
Namespace(activation='elu', batch_size=32, compare_after_hpss=False, conv_bias=False, eval=False, exp_name='temp_exp', kernel_size=3, l1_reg_lambda=0.003, learning_rate=0.0004, loss_domains=['spectrum'], metrics=['mae'], n_cqt_bins=12, n_layer_dec=6, n_layer_enc=10, n_mels=None, num_channel=50, recurrenter='three', resume=False, resume_num='', scale_r=2, source_norm='sqrsum', sparsemax_lst=64, sparsemax_type='multiply')
| With a sampling rate of 16000 Hz,
| the deepest encoded signal: 1 sample == 64 ms.
| At predicting impulses, which is done at u_conv3, 1 sample == 1 ms.
| and sparsemax_lst=64 samples at the same, at=`r` level
n_notes: 11, n_vsts:{'KD_KD': 11, 'SD_SD': 11, 'HH_CHH': 11, 'HH_OHH': 11, 'HH_PHH': 11, 'TT_HIT': 11, 'TT_MHT': 11, 'TT_HFT': 11, 'CY_RDC': 11, 'CY_CRC': 11, 'OT_TMB': 11}
```
then you'll see the model details.
```bash
DrummerHalfUNet(
  (unet): ValidAutoUnet(
    (d_conv0): Conv1d(1, 50, kernel_size=(3,), stride=(1,), bias=False)
    (d_convs): ModuleList(
      (0): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (1): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (2): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (3): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (4): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (5): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (6): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (7): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (8): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (9): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
    )
    (pools): ModuleList(
      (0): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (4): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (7): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (8): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (encode_conv): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
    (u_convs): ModuleList(
      (0): Conv1d(50, 50, kernel_size=(3,), stride=(1,), bias=False)
      (1): Conv1d(100, 50, kernel_size=(3,), stride=(1,), bias=False)
      (2): Conv1d(100, 50, kernel_size=(3,), stride=(1,), bias=False)
      (3): Conv1d(100, 50, kernel_size=(3,), stride=(1,), bias=False)
      (4): Conv1d(100, 50, kernel_size=(3,), stride=(1,), bias=False)
      (5): Conv1d(100, 50, kernel_size=(3,), stride=(1,), bias=False)
    )
    (last_conv): Conv1d(100, 100, kernel_size=(3,), stride=(1,))
  )
  (recurrenter): Recurrenter(
    (midi_x2h): GRU(100, 11, batch_first=True, bidirectional=True)
    (midi_h2hh): GRU(22, 11, batch_first=True)
    (midi_hh2y): GRU(1, 1, bias=False, batch_first=True)
  )
  (double_sparsemax): MultiplySparsemax(
    (sparsemax_inst): Sparsemax()
    (sparsemax_time): Sparsemax()
  )
  (zero_inserter): ZeroInserter()
  (synthesizer): FastDrumSynthesizer()
  (mixer): Mixer()
)
NUM_PARAM overall: 203869
             unet: 195250
      recurrenter: 8619
       sparsemaxs: 0
      synthesizer: 0
UM_PARAM overall: 203869
             unet: 195250
      recurrenter: 8619
       sparsemaxs: 0
      synthesizer: 0
```
..as well as training details..
```bash
PseudoCQT init with fmin:32, 12, bins, 12 bins/oct, win_len: 16384, n_fft:16384, hop_length:64
PseudoCQT init with fmin:65, 12, bins, 12 bins/oct, win_len: 8192, n_fft:8192, hop_length:64
PseudoCQT init with fmin:130, 12, bins, 12 bins/oct, win_len: 4096, n_fft:4096, hop_length:64
PseudoCQT init with fmin:261, 12, bins, 12 bins/oct, win_len: 2048, n_fft:2048, hop_length:64
PseudoCQT init with fmin:523, 12, bins, 12 bins/oct, win_len: 1024, n_fft:1024, hop_length:64
PseudoCQT init with fmin:1046, 12, bins, 12 bins/oct, win_len: 512, n_fft:512, hop_length:64
PseudoCQT init with fmin:2093, 12, bins, 12 bins/oct, win_len: 256, n_fft:256, hop_length:64
PseudoCQT init with fmin:4000, 12, bins, 12 bins/oct, win_len: 128, n_fft:128, hop_length:64
item check-points after this..: [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
total 8388480 n_items to train!

```
..then the training will start..
```bash
c1mae:5.53 c2mae:4.39 c3mae:2.95 c4mae:3.19 c5mae:2.22 c6mae:1.90 c7mae:2.14 c8mae:2.26: 100%|███████████████████████████████████| 1/1 [00:25<00:00, 25.03s/it]
```

## Troubleshooting
### Install MKL for pytorch FFT
In case you face this error, 
```bash
RuntimeError: fft: ATen not compiled with MKL support
```
[As stated here](https://discuss.pytorch.org/t/error-using-fft-runtimeerror-fft-aten-not-compiled-with-mkl-support/21671/2), this is an issue of MKL library installation. 
A quick solution is to use Conda. Otherwise you should install [Interl MKL](https://software.intel.com/en-us/get-started-with-mkl-for-macos) manually.

In some cases, if Pytorch was once built without MKL, it might not able to find later-installed MKL. 
You should try to remove the cache of pip/conda. Or just make a new environment.    

## Requirement detail

These are the exact versions I used for the dependency.
```
Python==3.7.3
Cython==0.29.6
cython==0.29.6
numpy==1.16.2
librosa==0.6.2
torch==1.0.0
torchvision==0.2.1
madmom==0.16.1
matplotlib==2.2.0
tqdm==4.31.1
mir_eval==0.5
```

## Citation

```
@inproceedings{choi2019deep,
  title={Deep Unsupervised Drum Transcription},
  author={Choi, Keunwoo and Cho, Kyunghyun},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), Delft, Netherland},
  year={2019}
}
```
