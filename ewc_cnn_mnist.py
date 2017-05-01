# -*- coding: utf-8 -*-
import numpy as np
import math
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import \
     cuda, gradient_check, optimizers, serializers, utils, \
     Chain, ChainList, Function, Link, Variable
from PIL import Image
import os
from StringIO import StringIO
from chainer import function
import numpy
from chainer.utils import type_check
from chainer.functions.loss import mean_squared_error 
import pylab
from chainer import variable
#import cupy as cp


from chainer.functions.activation import log_softmax


class NIN(chainer.Chain):

    """Network-in-Network example model."""

    insize = 28

    def __init__(self):
        layers = {}
        layers["conv1"] = L.Convolution2D(1,   96, 3, pad=1)
        layers["conv2"] = L.Convolution2D(96,  256,  3, pad=1)
        layers["conv3"] = L.Convolution2D(256,  384,  3, pad=1)
        layers["conv4"] = L.Convolution2D(384, 11,  3, pad=1)

        
        super(NIN, self).__init__(**layers)
        self.train = True

        self.var_list = []
        self.var_list.append(self.conv1.W)
        self.var_list.append(self.conv1.b)
        self.var_list.append(self.conv2.W)
        self.var_list.append(self.conv2.b)
        self.var_list.append(self.conv3.W)
        self.var_list.append(self.conv3.b)
        self.var_list.append(self.conv4.W)
        self.var_list.append(self.conv4.b)


    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = F.leaky_relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.leaky_relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.reshape(F.average_pooling_2d(h, h.data.shape[2]), (x.data.shape[0], 11))
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        self.h = h
        return self.loss

    def predict(self, x, t,train=False):
        self.clear()
        h = F.leaky_relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.leaky_relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.reshape(F.average_pooling_2d(h, h.data.shape[2]), (x.data.shape[0], 11))
        self.accuracy = F.accuracy(h, t)
        return h

    def Fissher(self,imageset,shape,gpu,num_samples):

        if gpu >= 0:
            xp = cp
        else:
            xp = np

        num_samples = num_samples
        
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(xp.zeros(self.var_list[v].data.shape))
        
        for i in range(num_samples):
            c,w,h = shape
            x = np.ndarray((1, c, w, h), dtype=np.float32)
            y = np.ndarray((1,), dtype=np.int32)
            rnd = np.random.randint(len(imageset))
            path = imageset[rnd][0] 
            label = imageset[rnd][1]
            x[0] = np.array(path)
            y[0] = np.array(label)
            if gpu >= 0:
                x = cuda.to_gpu(x)
                y = cuda.to_gpu(y)

            x = chainer.Variable(x)
            y = chainer.Variable(y)

            probs = F.log_softmax(self.predict(x,y))
            class_ind = np.argmax(cuda.to_cpu(probs.data))
            loss = probs[0,class_ind]
            self.cleargrads()
            loss.backward()
            for v in range(len(self.F_accum)):
                self.F_accum[v] += xp.square(self.var_list[v].grad)
            
        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples
        print "Fii",self.F_accum[0]

    def star(self,gpu):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            if gpu >= 0:
                self.star_vars.append(Variable(cuda.to_gpu(self.var_list[v].data)))
            else:
                self.star_vars.append(Variable(cuda.to_cpu(self.var_list[v].data)))

    def restore(self):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                self.var_list[v].data = self.star_vars[v]

    def update_ewc_loss(self, lam, gpu, model2):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if gpu >= 0:
            xp = cp     
            self.ewc_loss = self.loss
            for v in range(len(self.var_list)):
                self.ewc_loss += (lam/2) * F.sum(Variable(cuda.to_gpu(self.F_accum[v].astype(xp.float32))) * F.square(self.var_list[v] - model2.star_vars[v]))
        else:
            xp = np
            self.ewc_loss = self.loss
            for v in range(len(self.var_list)):
                self.ewc_loss += (lam/2) * F.sum(Variable(cuda.to_cpu(self.F_accum[v].astype(xp.float32))) * F.square(self.var_list[v] - model2.star_vars[v]))

        return self.ewc_loss


















