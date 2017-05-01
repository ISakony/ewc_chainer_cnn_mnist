# -*- coding: utf-8 -*-
import numpy as np
import math
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import \
     cuda, gradient_check, optimizers, serializers, utils, \
     Chain, ChainList, Function, Link, Variable

import os
from StringIO import StringIO
from chainer import function
import numpy
from chainer.utils import type_check
from chainer.functions.loss import mean_squared_error 

from chainer import variable
import cupy as cp
import pickle
import ewc_cnn_mnist_gpu



import gzip
import os
import struct

import six

from chainer.dataset import download
from chainer.datasets import tuple_dataset



out_model_dir = './out_models'

try:
    os.mkdir(out_model_dir)
except:
    pass
      

batchsize=320
n_epoch=10
n_epoch2=10
n_train=60000

## MNISTデータをロード
print "load MNIST dataset"
train_data, test_data = chainer.datasets.get_mnist(ndim=3)

model = ewc_cnn_mnist_gpu.NIN()
model.to_gpu()
serializers.load_npz("out_models/train_mm_1_9.npz",model)

o_model = optimizers.Adam()
o_model.setup(model)


def train(epoch,batchsize,train_data,test_data,mod,o_mod):

    x = np.ndarray((batchsize, 1, 28, 28), dtype=np.float32)
    y = np.ndarray((batchsize,), dtype=np.int32)
    for j in range(batchsize):
        rnd = np.random.randint(len(train_data))
        path = train_data[rnd][0] 
        label = train_data[rnd][1]
        x[j] = np.array(path)[:, :, ::-1]
        y[j] = np.array(label)

    x = chainer.Variable(cuda.to_gpu(x))
    y = chainer.Variable(cuda.to_gpu(y))
    #x = chainer.Variable(x)
    #y = chainer.Variable(y)

    loss = mod(x,y)
    acc_tr = mod.accuracy.data

    o_mod.zero_grads()
    loss.backward()
    o_mod.update()

    #test task
    x = np.ndarray((batchsize, 1, 28, 28), dtype=np.float32)
    y = np.ndarray((batchsize,), dtype=np.int32)
    for j in range(batchsize):
        rnd = np.random.randint(len(test_data))
        path = test_data[rnd][0] 
        label = test_data[rnd][1]
        x[j] = np.array(path)[:, :, ::-1]
        y[j] = np.array(label)

    x = chainer.Variable(cuda.to_gpu(x))
    y = chainer.Variable(cuda.to_gpu(y))
    #x = chainer.Variable(x)
    #y = chainer.Variable(y)

    acc_te = mod.predict(x,y)
    acc_te = mod.accuracy.data


    #anather testtask
    x = np.ndarray((batchsize, 1, 28, 28), dtype=np.float32)
    y = np.ndarray((batchsize,), dtype=np.int32)
    for j in range(batchsize):
        rnd = np.random.randint(len(test_data))
        path = test_data[rnd][0] 
        label = test_data[rnd][1]
        x[j] = np.array(path)
        y[j] = np.array(label)

    x = chainer.Variable(cuda.to_gpu(x))
    y = chainer.Variable(cuda.to_gpu(y))
    #x = chainer.Variable(x)
    #y = chainer.Variable(y)

    acc_an_t = mod.predict(x,y)
    acc_an_t = mod.accuracy.data


    f = open("log2.txt","a") #
    f.write('epoch' +" "+ str(epoch)+" " + str(i)+" loss " + str(loss.data)+" " +"acc_tr"+" " + str(acc_tr)+"acc_te"+" " + str(acc_te)+"acc_an_t"+" " + str(acc_an_t)
                + "\n")
    f.close
    print 'epoch',epoch,"loss",loss.data,"acc_tr",acc_tr,"acc_te",acc_te,"acc_an_t",acc_an_t
    
for epoch in xrange(0,n_epoch):
    for i in xrange(0, n_train, batchsize):    
        train(epoch,batchsize,train_data,test_data,model,o_model)
    serializers.save_npz("%s/train_mm_2_%d.npz"%(out_model_dir, epoch),model)
