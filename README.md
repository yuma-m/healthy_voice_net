# Healthy Voice Net

サバフェス 2016の参加作品です。
ニューラルネットを用いて、声色から体調を推定します。

http://2016.serverfesta.info/

## Usage

### Train Neural Network

`data/`以下のフォルダに音声のwavファイル(モノラル, 16bit, 44,1kHz)を入れておきます。

```sh
$ python train_nn.py
```

### Test Neural Network

学習済みのニューラルネットのモデルを`nn.model`として保存しておきます。

```sh
$ python test_nn.py your_voice.wav
```

### Launch System on Raspberry Pi

```sh
$ python record.py
```

## Installation

### Mac OS X

```sh
$ brew install portaudio
```

```sh
$ pip install scipy chainer pyaudio requests
```

### Raspbian / Debian / Ubuntu

```sh
$ sudo apt-get install python-pyaudio python3-pyaudio python-scipy
$ sudo apt-get install alsa-utils sox libsox-fmt-all
```

```sh
$ sudo pip install chainer
```

### Breadboard wiring example

![Breadboard wiring](/image/circuit.png)
