# This source file is part of DiffAI
# Copyright (c) 2018 Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich
# This software is distributed under the MIT License: https://opensource.org/licenses/MIT
# SPDX-License-Identifier: MIT
# For more information see https://github.com/eth-sri/diffai

# THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
# EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO ANY WARRANTY
# THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS OR BE ERROR-FREE AND ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
# TITLE, OR NON-INFRINGEMENT.  IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY     
#  DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT,
# SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN
# ANY WAY CONNECTED WITH THIS SOFTWARE (WHETHER OR NOT BASED UPON WARRANTY,
# CONTRACT, TORT OR OTHERWISE).

import torch
import torch.nn as nn
import helpers as h
import math
import abc

from torch.nn.modules.conv import _ConvNd

class InferModule(nn.Module):
    def __init__(self, *args, normal = False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.infered = False
        self.normal = normal

    def infer(self, prev, global_args = None):
        """ this is really actually stateful. """

        if self.infered:
            return self
        self.infered = True

        super(InferModule, self).__init__()
        self.inShape = prev
        self.outShape = self.init(prev, *self.args, global_args = global_args, **self.kwargs)
        if self.outShape is None:
            raise "init should set the out_shape"
        
        self.reset_parameters()
        return self
    
    def reset_parameters(self):
        if not hasattr(self,'weight') or self.weight is None:
            return
        n = h.product(self.weight.size()) / self.outShape[0]
        stdv = 1. / math.sqrt(n)
        if self.normal:
            #stdv *= math.sqrt(2)
            self.weight.data.normal_(0, stdv)
            self.weight.data.clamp_(-1, 1)
        else:
            self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            if self.normal:
                self.bias.data.normal_(0, stdv)
                self.bias.data.clamp_(-1, 1)
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def clip_norm(self):
        if not hasattr(self, "weight"):
            return
        if not hasattr(self,"weight_g"):
            nn.utils.weight_norm(self, dim=None)
        self.weight_g.data.clamp_(-float(h.max_c_for_norm), float(h.max_c_for_norm))



    def printNet(self):
        print(self.__class__.__name__)
        
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def neuronCount(self):
        pass

def getShapeConv(in_shape, conv_shape, stride = 1, padding = 0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)


class Linear(InferModule):
    def init(self, prev, out_shape, **kargs):
        self.in_neurons = h.product(prev)
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        self.out_neurons = h.product(out_shape) 
        
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_neurons, self.out_neurons))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_neurons))

        return out_shape

    def forward(self, x):
        s = x.size()
        x = x.view(s[0], h.product(s[1:]))
        return (x.matmul(self.weight) + self.bias).view(s[0], *self.outShape)

    def neuronCount(self):
        return 0

    def printNet(self, f):
        print(str([ list(l) for l in self.weight.transpose(1,0).data]), file= f)
        print(str(list(self.bias.data)), file= f)

class ReLU(InferModule):
    def init(self, prev, global_args = None, **kargs):
        self.use_softplus = h.default(global_args, 'use_softplus', False)
        return prev

    def forward(self, x):
        return x.softplus() if self.use_softplus else x.relu()

    def neuronCount(self):
        return h.product(self.outShape)

class Conv2D(InferModule):

    def init(self, prev, out_channels, kernel_size, stride = 1, global_args = None, bias=True, padding = 0, **kargs):
        self.prev = prev
        self.in_channels = prev[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_softplus = h.default(global_args, 'use_softplus', False)
        
        weights_shape = (self.out_channels, self.in_channels, kernel_size, kernel_size)        
        self.weight = torch.nn.Parameter(torch.Tensor(*weights_shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(weights_shape[0]))
        else:
            self.bias = h.zeros(weights_shape[0])
            
        outshape = getShapeConv(prev, (out_channels, kernel_size, kernel_size), stride, padding)
        return outshape

        
    def forward(self, input):
        return input.conv2d(self.weight, self.bias, self.stride, padding = self.padding )
    
    def printNet(self, f): # only complete if we've got stride=1
        print("Conv2D", file = f)
        print("ReLU, filters={}, kernel_size={}, input_shape={}".format(self.out_channels, list(self.kernel_size), list(self.prev) ), file = f)
        print(str([[[list(r) for r in p] for p in l ] for l in self.weight.permute(2,3,1,0).data]) , file= f)
        print(str(list(self.bias.data)), file= f)

    def neuronCount(self):
        return 0

class MaxPool2D(InferModule):
    def init(self, prev, kernel_size, stride = None, **kargs):
        self.prev = prev
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        return getShapeConv(prev, (prev[0], kernel_size, kernel_size), stride)

    def forward(self, x):
        return x.max_pool2d(self.kernel_size, self.stride)
    
    def printNet(self, f):
        print("MaxPool2D stride={}, kernel_size={}, input_shape={}".format(list(self.stride), list(self.shape[2:]), list(self.prev[1:]+self.prev[:1]) ), file = f)
        
    def neuronCount(self):
        return h.product(self.outShape)

class Flatten(InferModule):
    def init(self, prev, **kargs):
        return h.product(prev)
        
    def forward(self, x):
        s = x.size()
        return x.view(s[0], h.product(s[1:]))

    def neuronCount(self):
        return 0
    
class Seq(InferModule):
    def init(self, prev, *layers, **kargs):
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.prev = prev
        for s in layers:
            prev = s.infer(prev, **kargs).outShape
        return prev
    
    def forward(self, x):
        return self.net(x)

    def clip_norm(self):
        for l in self.layers:
            l.clip_norm()
            
    def printNet(self, f):
        for l in self.layers:
            l.printNet(f)

    def neuronCount(self):
        return sum([l.neuronCount() for l in self.layers ]) 
    
def FFNN(layers, last_lin = False, **kargs):
    starts = layers
    ends = []
    if last_lin:
        starts = layers[:-1]
        ends = [Linear(layers[-1],**kargs)]
    return Seq(*([ Seq(Linear(s, **kargs), ReLU(**kargs)) for s in starts] + ends))

def Conv(*args, **kargs):
    return Seq(Conv2D(*args, **kargs), ReLU(**kargs))

MP = MaxPool2D 

def LeNet(conv_layers, ly, bias = True, normal=False, **kargs):
    def transfer(tp):
        if isinstance(tp, InferModule):
            return tp
        if isinstance(tp[0], str):
            return MaxPool2D(*tp[1:])
        return Conv(out_channels = tp[0], kernel_size = tp[1], stride = tp[-1] if len(tp) == 4 else 1, bias=bias, normal=normal)
                      
    return Seq(*([transfer(s) for s in conv_layers] + [FFNN(ly, **kargs, bias=bias)]))

class FromByteImg(InferModule):
    def init(self, prev, **kargs):
        return prev
    
    def forward(self, x):
        return x.float()/ 256.

    def neuronCount(self):
        return 0
        
class Skip(InferModule):
    def init(self, prev, net1, net2, **kargs):
        self.net1 = net1.infer(prev, **kargs)
        self.net2 = net2.infer(prev, **kargs)
        assert(net1.outShape[1:] == net2.outShape[1:])
        return [ net1.outShape[0] + net2.outShape[0] ] + net1.outShape[1:]
    
    def forward(self, x):
        r1 = self.net1(x)
        r2 = self.net2(x)
        return r1.cat(r2, dim=1)

    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()

class ParSum(InferModule):
    def init(self, prev, net1, net2, **kargs):
        self.net1 = net1.infer(prev, **kargs)
        self.net2 = net2.infer(prev, **kargs)
        assert(net1.outShape == net2.outShape)
        return net1.outShape
    
    def forward(self, x):
        r1 = self.net1(x)
        r2 = self.net2(x)
        return h.cadd(r1,r2)

    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()

def SkipNet(net1, net2, ffnn, **kargs):
    return Seq(Skip(net1,net2), FFNN(ffnn, **kargs))

def BasicBlock(in_planes, planes, stride=1, **kargs):
    block = Seq( Conv(planes, kernel_size = 3, stride = stride, padding = 1, bias=False, normal=True, **kargs)
               , Conv2D(planes, kernel_size = 3, stride = 1, padding = 1, bias=False, normal=True, **kargs))

    if stride != 1 or in_planes != planes:
        block = ParSum(block, Conv2D(planes, kernel_size=1, stride=stride, bias=False, normal=True, **kargs))
    return Seq(block, ReLU(**kargs))


def ResNet(blocksList, **kargs):

    layers = []
    in_planes = 64
    planes = 64
    stride = 0
    for num_blocks in blocksList:
        if stride < 2:
            stride += 1

        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride, **kargs))
            in_planes = planes
        planes *= 2

    return Seq(Conv(64, kernel_size=3, stride=1, padding = 1, bias=False, normal=True, printShape=True), 
               *layers)
