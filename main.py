# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(LeeFlow)s
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon, autograd
import matplotlib.pyplot as plt
import argparse
from model import LeNetPlus
from L_GM import L_GM_Loss
import seaborn as sns
from utils import evaluate_accuracy, plot_features
plt.style.use('ggplot')
%matplotlib inline

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)).asnumpy()  / 255, label.astype(np.int32)

def train():
    mnist_set = gluon.data.vision.MNIST(train=True, transform=transform)
    test_mnist_set = gluon.data.vision.MNIST(train=False, transform=transform)
    ctx = mx.gpu(0)
    model = LeNetPlus()
    model.hybridize()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    L_Gm = L_GM_Loss(10,2,arg.margin, arg.lamda, arg.mult)
    L_Gm.initialize(mx.init.Xavier(), ctx=ctx)
    train_iter = mx.gluon.data.DataLoader(mnist_set, 250, shuffle=True)
    test_iter = mx.gluon.data.DataLoader(test_mnist_set, 500, shuffle=False)
    params = model.collect_params()
    params.update(L_Gm.collect_params())
    trainer = gluon.Trainer(params, optimizer='adam', optimizer_params={'learning_rate': 1e-3, 'wd': 5e-4})
    for e in range(50):
        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                features = model(data)
                loss, _ = L_Gm(features, label)
            loss.backward()
            trainer.step(data.shape[0])
            curr_loss = nd.mean(loss).asscalar()
        if ((e+1)%10 == 0):
            test_accuracy, test_ft, _, test_lb = evaluate_accuracy(test_iter, model, L_Gm, ctx)
            print(test_accuracy)
            plot_features(test_ft, test_lb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convolutional Neural Networks')
    # File related
    parser.add_argument('--magrin', default=0.1, type=float, help='margin in l-gm loss')
    parser.add_argument('--lamda', default=0.2, type=float, help='weight of likelihood loss')
    parser.add_argument('--mult', default=0.06, type=float, help='lr mult in variance update')
    arg = parser.parse_args()
    train()
