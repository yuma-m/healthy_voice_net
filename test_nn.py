#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import sys
import math

import numpy as np

import chainer
import chainer.links as L
from chainer import optimizers
from chainer import serializers

from audio_nn import AudioNetTrainer, AudioNet


DATA_DIR = [
    './data/normal/',  # 体調が普通の時の声
    './data/cold/',    # 風邪気味の時の声
    './data/sleepy/',  # 寝不足の時の声
]


class AudioNetTester(AudioNetTrainer):
    def __init__(self):
        super(AudioNetTester, self).__init__()

    def test(self, wav):
        result = self.evaluate(wav)
        return result

    def evaluate(self, wav):
        data = np.array([self.read_wav(wav)], np.float32)

        model = L.Classifier(AudioNet())
        xp = np

        optimizer = optimizers.Adam()
        optimizer.setup(model)

        print('Load model')
        serializers.load_npz('nn.model', model)

        x = chainer.Variable(xp.asarray(data),
                             volatile='on')
        y = model.predictor(x)

        normal = 1.0 / (1.0 + math.exp(-y.data[0][0]))
        cold = 1.0 / (1.0 + math.exp(-y.data[0][1]))
        sleepy = 1.0 / (1.0 + math.exp(-y.data[0][2]))

        sum_val = normal + cold + sleepy

        nval = normal / sum_val * 100.0
        cval = cold / sum_val * 100.0
        sval = sleepy / sum_val * 100.0

        print("normal: %.2f" % (nval))
        print("cold  : %.2f" % (cval))
        print("sleepy: %.2f" % (sval))
        return (nval, cval, sval)


def test_wav(wav_file):
    tester = AudioNetTester()
    tester.test(wav_file)


if __name__ == '__main__':
    test_wav(sys.argv[1])
