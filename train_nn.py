#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import glob

import numpy as np
import six
from scipy.fftpack import fft
from scipy.io import wavfile

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import computational_graph
from chainer import optimizers
from chainer import serializers


DATA_DIR = [
    './data/normal/',  # 体調が普通の時の声
    './data/cold/',    # 風邪気味の時の声
    './data/sleepy/',  # 寝不足の時の声
]

# ニューラルネットの定義
N_IN = 20000
N_1 = 100
N_2 = 20
N_OUT = len(DATA_DIR)


class AudioNet(chainer.Chain):
    def __init__(self):
        super(AudioNet, self).__init__(
            l1=L.Linear(N_IN, N_1),
            l2=L.Linear(N_1, N_2),
            l3=L.Linear(N_2, N_OUT),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class AudioNetTrainer(object):
    def __init__(self):
        pass

    def train(self):
        self.collect_data()
        self.train_and_evaluate()

    def collect_data(self):
        u'''データディレクトリから各体調の音声を読みだす'''
        self.in_data, self.out_data = [], []
        self.in_test, self.out_test = [], []
        for idx, data_dir in enumerate(DATA_DIR):
            sid, sod, sit, sot = self.read_data(idx, data_dir)
            self.in_data += sid
            self.out_data += sod
            self.in_test += sit
            self.out_test += sot

    def read_data(self, out, data_dir):
        u'''各フォルダ内のwavファイルを読みだす

        out: 0(良好), 1(風邪気味), 2(寝不足)
        '''
        wav_list = glob.glob(data_dir + '*.wav')
        sub_in_data, sub_out_data = [], []
        sub_in_test, sub_out_test = [], []
        out_data = out
        # 元のデータの9割を学習用に、1割をテスト用に使う
        for wav in wav_list[:len(wav_list) * 9 / 10]:
            in_data = self.read_wav(wav)
            sub_in_data.append(in_data)
            sub_out_data.append(out_data)

        for wav in wav_list[len(wav_list) * 9 / 10:]:
            in_data = self.read_wav(wav)
            sub_in_test.append(in_data)
            sub_out_test.append(out_data)
        return sub_in_data, sub_out_data, sub_in_test, sub_out_test

    def read_wav(self, wav):
        u'''wavファイルを読んでFFTをかける'''
        fs, data = wavfile.read(wav)
        track = data.T
        samples = [(ele/2**8.)*2-1 for ele in track]
        # FFT変換する
        spectrum = fft(samples)
        d = len(spectrum)/2
        average = sum(abs(spectrum[:(d-1)])) / d
        # 音量を正規化する
        data = abs(spectrum[:(d-1)]) / average
        return data[:N_IN]

    def train_and_evaluate(self):
        u'''学習のメインループ

        ChainerのMNISTのサンプルをもとに作成
        '''
        batchsize = 100
        n_epoch = 60

        print('load audio dataset')

        x_train = np.array(self.in_data, np.float32)
        x_test = np.array(self.in_test, np.float32)
        y_train = np.array(self.out_data, np.int32)
        y_test = np.array(self.out_test, np.int32)
        N_test = y_test.size

        model = L.Classifier(AudioNet())
        xp = np

        optimizer = optimizers.Adam()
        optimizer.setup(model)

        N = len(x_train)
        for epoch in six.moves.range(1, n_epoch + 1):
            print('epoch', epoch)

            perm = np.random.permutation(N)
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, N, batchsize):
                x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
                t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

                optimizer.update(model, x, t)

                if epoch == 1 and i == 0:
                    with open('graph.dot', 'w') as o:
                        g = computational_graph.build_computational_graph(
                            (model.loss, ))
                        o.write(g.dump())
                    print('graph generated')

                sum_loss += float(model.loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            print('train mean loss={}, accuracy={}'.format(
                sum_loss / N, sum_accuracy / N))

            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, N_test, batchsize):
                x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                                     volatile='on')
                t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                                     volatile='on')
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            print('test  mean loss={}, accuracy={}'.format(
                sum_loss / N_test, sum_accuracy / N_test))

        print('save the model')
        serializers.save_npz('nn.model', model)
        print('save the optimizer')
        serializers.save_npz('nn.state', optimizer)


def main():
    trainer = AudioNetTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
