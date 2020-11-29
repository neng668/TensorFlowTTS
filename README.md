<h2 align="center">
<p> TensorFlowTTS - Speech Synthesis for New Zealand English
<p align="center">
    <a href="https://colab.research.google.com/drive/1aWjxYkvh6W6W1hXXEKDPc9XjCJ262r6c?usp=sharing">
        <img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>
</h2>
<h2 align="center">
<p>Real-Time State-of-the-art Speech Synthesis with Tensorflow 2 for New Zealand English

## Requirements
This repository is tested on Ubuntu 18.04 with:

- Python 3.7+
- Cuda 10.1
- CuDNN 7.6.5
- Tensorflow 2.2/2.3
- [Tensorflow Addons](https://github.com/tensorflow/addons) >= 0.10.0

## Installation
### With pip
```bash
$ pip install TensorFlowTTS
```
### From source
Examples are included in the repository but are not shipped with the framework. Therefore, to run the latest version of examples, you need to install the source below.
```bash
$ git clone https://github.com/sabby993/TensorFlowTTS.git
$ cd TensorFlowTTS
$ pip install .
```
If you want to upgrade the repository and its dependencies:
```bash
$ git pull
$ pip install --upgrade .
```

# Model architectures

1. **MelGAN** released with the paper [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) by Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville.
2. **Tacotron-2** released with the paper [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) by Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu.


# Audio Samples
Here in an audio samples on valid set in New Zealand English. You can find the audio sample at the end of the colab notebook and also the text will be displayed above the allignemnt and spectrogram plots. [tacotron-2](https://colab.research.google.com/drive/1aWjxYkvh6W6W1hXXEKDPc9XjCJ262r6c?usp=sharing)

# Tutorial End-to-End

## Prepare Dataset

Prepare a dataset in the following format:
```
|- [NAME_DATASET]/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
```

Where `metadata.csv` has the following format: `id|transcription`. This is a ljspeech-like format; for nz_cw dataset, we modeled it according to the LJ Speech format. By doing this, we can use the LJ Speech processor to pre-process and train the nz_cw dataset.

Note that `NAME_DATASET` should be `[ljspeech/kss/baker/libritts/nz_cw]` for example.

## Preprocessing

The preprocessing has two steps:

1. Preprocess audio features
    - Convert characters to IDs
    - Compute mel spectrograms
    - Normalize mel spectrograms to [-1, 1] range
    - Split the dataset into train and validation
    - Compute the mean and standard deviation of multiple features from the **training** split
2. Standardize mel spectrogram based on computed statistics

To reproduce the steps above:
```
tensorflow-tts-preprocess --rootdir ./[ljspeech/nz_cw] --outdir ./dump_[ljspeech/nz_cw] --config preprocess/[ljspeech]_preprocess.yaml --dataset [ljspeech]
tensorflow-tts-normalize --rootdir ./dump_[ljspeech/nz_cw] --outdir ./dump_[ljspeech/nz_cw] --config preprocess/[ljspeech]_preprocess.yaml --dataset [ljspeech]
```

After preprocessing, the structure of the project folder should be:
```
|- [NAME_DATASET]/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
|- dump_[ljspeech/nz_cw]/
|   |- train/
|       |- ids/
|           |- LJ001-0001-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0001-raw-feats.npy
|           |- ...
|       |- raw-f0/
|           |- LJ001-0001-raw-f0.npy
|           |- ...
|       |- raw-energies/
|           |- LJ001-0001-raw-energy.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0001-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0001-wave.npy
|           |- ...
|   |- valid/
|       |- ids/
|           |- LJ001-0009-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0009-raw-feats.npy
|           |- ...
|       |- raw-f0/
|           |- LJ001-0001-raw-f0.npy
|           |- ...
|       |- raw-energies/
|           |- LJ001-0001-raw-energy.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0009-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0009-wave.npy
|           |- ...
|   |- stats.npy
|   |- stats_f0.npy
|   |- stats_energy.npy
|   |- train_utt_ids.npy
|   |- valid_utt_ids.npy
|- examples/
|   |- melgan/
|   |- fastspeech/
|   |- tacotron2/
|   ...
```

- `stats.npy` contains the mean and std from the training split mel spectrograms
- `stats_energy.npy` contains the mean and std of energy values from the training split
- `stats_f0.npy` contains the mean and std of F0 values in the training split
- `train_utt_ids.npy` / `valid_utt_ids.npy` contains training and validation utterances IDs respectively

We use suffix (`ids`, `raw-feats`, `raw-energy`, `raw-f0`, `norm-feats`, and `wave`) for each input type.

## Training the model:

After the above steps, a folder named ‘dump_ljspeech’ or 'dump_nz_cw' will be created inside the repository which will contain numpy files of the training and validation datasets.

Now we can start the training process which will need the actual use of the GPU. We can start training by running the following code:
```
!CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
--train-dir ./dump_ljspeech/train/ \
--dev-dir ./dump_ljspeech/valid/ \
--outdir ./examples/tacotron2/exp/train.tacotron2.v1/ \
--config ./examples/tacotron2/conf/tacotron2.v1.yaml \
--use-norm 1 \
--mixed_precision 0 \
--pretrained ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-120000.h5
```
The pretrained argument takes in the path of the pretrained model which is the base model on top of which the current dataset is fine-tuned upon.
This code will run for quite some time and checkpoints will be saved every 2000 steps of training and a checkpoint file will be created inside the examples folder (examples/tacotron2/exp/train.tacotron2.v1/checkpoints/ckpt-2000).
We can resume the training from the checkpoint by running the training command and adding the checkpoint directory after the --resume part, e.g
```
!CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
 --train-dir ./dump_ljspeech/train/ \
--dev-dir ./dump_ljspeech/valid/ \
--outdir ./examples/tacotron2/exp/train.tacotron2.v1/ \
--config ./examples/tacotron2/conf/tacotron2.v1.yaml \
--use-norm 1 \
--mixed_precision 0 \
--resume ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/ckpt-2000
```

The colab notebook used for training nz_cw dataset on top of pretrained LJ Speech model can be found here [Training and Tensorboard Graphs](https://colab.research.google.com/drive/14K72jGtCnIkQHcI9nxF-QL4Oyr8qvAND?usp=sharing)

## Inference

The whole inference steps are there in the colab notebook here - [Inference](https://colab.research.google.com/drive/1aWjxYkvh6W6W1hXXEKDPc9XjCJ262r6c?usp=sharing)

## Results

To show the difference in accent properties between the General American English and New Zealand English, we employed RStudio to get vowel space plots of both the synthesized speech models speaking hVd words and the plots are below:

For New Zealand English -->

![alt text](https://drive.google.com/file/d/1jIFWNm8j5T3kVU20beUhq_rt1PUCyno_/view?usp=sharing)

For General American English -->

![alt text](https://drive.google.com/file/d/1YR5W6qgMkVGIH0yfA-Mx18J_UBdwasWB/view?usp=sharing)

For New Zealand Real Voice -->

![alt text](https://drive.google.com/file/d/1kgOjH3GQvCDSZiML2DUJnToMrvRxURVl/view?usp=sharing)

As we can see from the vowel plots, the Tacotron-2 synthesized voice of New Zealand English is quite close to the vowel space plot of the real voice plot in terms of position of the vowels in the F1/F2 space and the plots are quite different for the General American English model vowel plot.
