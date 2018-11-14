# This source file is part of DiffAI
# Copyright (c) 2018 Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich
# This software is distributed under the MIT License: https://opensource.org/licenses/MIT
# SPDX-License-Identifier: MIT
# Author: Matthew Mirman (matt@mirman.com)
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

try:
    from . import helpers as h
except:
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

    def infer(self, in_shape, global_args = None):
        """ this is really actually stateful. """

        if self.infered:
            return self
        self.infered = True

        super(InferModule, self).__init__()
        self.inShape = in_shape
        self.outShape = self.init(in_shape, *self.args, global_args = global_args, **self.kwargs)
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

    def remove_norm(self):
        if hasattr(self,"weight_g"):
            torch.nn.utils.remove_weight_norm(self)

    def printNet(self, f):
        print(self.__class__.__name__, file=f)
        
    @abc.abstractmethod
    def forward(self, x, **kargs):
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

def getShapeConvTranspose(in_shape, conv_shape, stride = 1, padding = 0, out_padding=0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = (inH - 1 ) * stride - 2 * padding + kH + out_padding
    outW = (inW - 1 ) * stride - 2 * padding + kW + out_padding
    return (outChan, outH, outW)



class Linear(InferModule):
    def init(self, in_shape, out_shape, **kargs):
        self.in_neurons = h.product(in_shape)
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        self.out_neurons = h.product(out_shape) 
        
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_neurons, self.out_neurons))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_neurons))

        return out_shape

    def forward(self, x, **kargs):
        s = x.size()
        x = x.view(s[0], h.product(s[1:]))
        return (x.matmul(self.weight) + self.bias).view(s[0], *self.outShape)

    def neuronCount(self):
        return 0

    def printNet(self, f):
        print(h.printListsNumpy(list(self.weight.transpose(1,0).data)), file= f)
        print(h.printNumpy(self.bias), file= f)

class Activation(InferModule):
    def init(self, in_shape, global_args = None, activation = "ReLU", **kargs):
        self.activation = [ "ReLU","Sigmoid","Tanh" , "Softplus"].index(activation)
        return in_shape

    def forward(self, x, **kargs):
        return [lambda x:x.relu(), lambda x:x.sigmoid(), lambda x:x.tanh(), lambda x:x.softplus()][self.activation](x)

    def neuronCount(self):
        return h.product(self.outShape)

    def printNet(self, f):
        pass

class ReLU(Activation):
    pass

class Identity(InferModule): # for feigning model equivelence when removing an op
    def init(self, in_shape, global_args = None, **kargs):
        return in_shape

    def forward(self, x, **kargs):
        return x

    def neuronCount(self):
        return 0

    def printNet(self, f):
        pass

class PrintActivation(Identity):
    def init(self, in_shape, global_args = None, activation = "ReLU", **kargs):
        self.activation = activation
        return in_shape

    def printNet(self, f):
        print(self.activation, file = f)

class PrintReLU(PrintActivation):
    pass

class Conv2D(InferModule):

    def init(self, in_shape, out_channels, kernel_size, stride = 1, global_args = None, bias=True, padding = 0, activation = "ReLU", **kargs):
        self.prev = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_softplus = h.default(global_args, 'use_softplus', False)
        
        weights_shape = (self.out_channels, self.in_channels, kernel_size, kernel_size)        
        self.weight = torch.nn.Parameter(h.dtype(*weights_shape))
        if bias:
            self.bias = torch.nn.Parameter(h.dtype(weights_shape[0]))
        else:
            self.bias = None # h.zeros(weights_shape[0])
            
        outshape = getShapeConv(in_shape, (out_channels, kernel_size, kernel_size), stride, padding)
        return outshape

        
    def forward(self, input, **kargs):
        return input.conv2d(self.weight, bias=self.bias, stride=self.stride, padding = self.padding )
    
    def printNet(self, f): # only complete if we've got stride=1
        print("Conv2D", file = f)
        sz = list(self.prev)
        print(self.activation + ", filters={}, kernel_size={}, input_shape={}, stride={}, padding={}".format(self.out_channels, [self.kernel_size, self.kernel_size], list(reversed(sz)), [self.stride, self.stride], self.padding ), file = f)
        print(h.printListsNumpy([[list(p) for p in l ] for l in self.weight.permute(2,3,1,0).data]) , file= f)
        print(h.printNumpy(self.bias if self.bias is not None else h.dtype(self.out_channels)), file= f)


    def neuronCount(self):
        return 0


class ConvTranspose2D(InferModule):

    def init(self, in_shape, out_channels, kernel_size, stride = 1, global_args = None, bias=True, padding = 0, out_padding=0, activation = "ReLU", **kargs):
        self.prev = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = out_padding
        self.activation = activation
        self.use_softplus = h.default(global_args, 'use_softplus', False)
        
        weights_shape = (self.in_channels, self.out_channels, kernel_size, kernel_size)        
        self.weight = torch.nn.Parameter(torch.Tensor(*weights_shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(weights_shape[0]))
        else:
            self.bias = None # h.zeros(weights_shape[0])
            
        outshape = getShapeConvTranspose(in_shape, (out_channels, kernel_size, kernel_size), stride, padding, out_padding)
        return outshape

        
    def forward(self, input, **kargs):
        return input.conv_transpose2d(self.weight, bias=self.bias, stride=self.stride, padding = self.padding, output_padding=self.out_padding)
    
    def printNet(self, f): # only complete if we've got stride=1
        print("ConvTranspose2D", file = f)
        print(self.activation + ", filters={}, kernel_size={}, input_shape={}".format(self.out_channels, list(self.kernel_size), list(self.prev) ), file = f)
        print(h.printListsNumpy([[list(p) for p in l ] for l in self.weight.permute(2,3,1,0).data]) , file= f)
        print(h.printNumpy(self.bias), file= f)

    def neuronCount(self):
        return 0

class MaxPool2D(InferModule):
    def init(self, in_shape, kernel_size, stride = None, **kargs):
        self.prev = in_shape
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        return getShapeConv(in_shape, (in_shape[0], kernel_size, kernel_size), stride)

    def forward(self, x, **kargs):
        return x.max_pool2d(self.kernel_size, self.stride)
    
    def printNet(self, f):
        print("MaxPool2D stride={}, kernel_size={}, input_shape={}".format(list(self.stride), list(self.shape[2:]), list(self.prev[1:]+self.prev[:1]) ), file = f)
        
    def neuronCount(self):
        return h.product(self.outShape)

class Normalize(InferModule):
    def init(self, in_shape, mean, std, **kargs):
        self.mean_v = mean
        self.std_v = std
        self.mean = h.dtype(mean)
        self.std = 1 / h.dtype(std)
        return in_shape

    def forward(self, x, **kargs):
        mean_ex = self.mean.view(self.mean.shape[0],1,1).expand(*x.size()[1:])
        std_ex = self.std.view(self.std.shape[0],1,1).expand(*x.size()[1:])
        return (x - mean_ex) * std_ex

    def neuronCount(self):
        return 0

    def printNet(self, f):
        print("Normalize mean={} std={}".format(self.mean_v, self.std_v), file = f)

class Flatten(InferModule):
    def init(self, in_shape, **kargs):
        return h.product(in_shape)
        
    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], h.product(s[1:]))

    def neuronCount(self):
        return 0

class Unflatten2d(InferModule):
    def init(self, in_shape, w, **kargs):
        self.w = w
        self.outChan = int(h.product(in_shape) / (w * w))
        
        return (self.outChan, self.w, self.w)
        
    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], self.outChan, self.w, self.w)

    def neuronCount(self):
        return 0


class View(InferModule):
    def init(self, in_shape, out_shape, **kargs):
        assert(h.product(in_shape) == h.product(out_shape))
        return out_shape
        
    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], *self.outShape)

    def neuronCount(self):
        return 0
    
class Seq(InferModule):
    def init(self, in_shape, *layers, **kargs):
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.prev = in_shape
        for s in layers:
            in_shape = s.infer(in_shape, **kargs).outShape
        return in_shape
    
    def forward(self, x, **kargs):
        for l in self.layers:
            x = l(x, **kargs)
        return x

    def clip_norm(self):
        for l in self.layers:
            l.clip_norm()

    def remove_norm(self):
        for l in self.layers:
            l.remove_norm()

    def printNet(self, f):
        for l in self.layers:
            l.printNet(f)

    def neuronCount(self):
        return sum([l.neuronCount() for l in self.layers ]) 
    
def FFNN(layers, last_lin = False, **kargs):
    starts = layers
    ends = []
    if last_lin:
        ends = [Seq(PrintActivation(activation = "Affine"), Linear(layers[-1],**kargs))]
        starts = layers[:-1]
    return Seq(*([ Seq(PrintActivation(**kargs), Linear(s, **kargs), Activation(**kargs)) for s in starts] + ends))

def Conv(*args, **kargs):
    return Seq(Conv2D(*args, **kargs), Activation(**kargs))

def ConvTranspose(*args, **kargs):
    return Seq(ConvTranspose2D(*args, **kargs), Activation(**kargs))

MP = MaxPool2D 

def LeNet(conv_layers, ly, bias = True, normal=False, **kargs):
    def transfer(tp):
        if isinstance(tp, InferModule):
            return tp
        if isinstance(tp[0], str):
            return MaxPool2D(*tp[1:])
        return Conv(out_channels = tp[0], kernel_size = tp[1], stride = tp[-1] if len(tp) == 4 else 1, bias=bias, normal=normal, **kargs)
                      
    return Seq(*[transfer(s) for s in conv_layers], FFNN(ly, **kargs, bias=bias))

def InvLeNet(ly, w, conv_layers, bias = True, normal=False, **kargs):
    def transfer(tp):
        return ConvTranspose(out_channels = tp[0], kernel_size = tp[1], stride = tp[2], padding = tp[3], out_padding = tp[4], bias=False, normal=normal)
                      
    return Seq(FFNN(ly, bias=bias), Unflatten2d(w),  *[transfer(s) for s in conv_layers])

class FromByteImg(InferModule):
    def init(self, in_shape, **kargs):
        return in_shape
    
    def forward(self, x, **kargs):
        return x.float()/ 256.

    def neuronCount(self):
        return 0
        
class Skip(InferModule):
    def init(self, in_shape, net1, net2, **kargs):
        self.net1 = net1.infer(in_shape, **kargs)
        self.net2 = net2.infer(in_shape, **kargs)
        assert(net1.outShape[1:] == net2.outShape[1:])
        return [ net1.outShape[0] + net2.outShape[0] ] + net1.outShape[1:]
    
    def forward(self, x, **kargs):
        r1 = self.net1(x, **kargs)
        r2 = self.net2(x, **kargs)
        return r1.cat(r2, dim=1)


    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def remove_norm(self):
        self.net1.remove_norm()
        self.net2.remove_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()

    def printNet(self, f):
        print("SkipNet1", file=f)
        self.net1.printNet(f)
        print("SkipNet2", file=f)
        self.net1.printNet(f)
        print("SkipCat dim=1", file=f)

class ParSum(InferModule):
    def init(self, in_shape, net1, net2, **kargs):
        self.net1 = net1.infer(in_shape, **kargs)
        self.net2 = net2.infer(in_shape, **kargs)
        assert(net1.outShape == net2.outShape)
        return net1.outShape
    
    def forward(self, x, **kargs):
        r1 = self.net1(x, **kargs)
        r2 = self.net2(x, **kargs)
        return r1 + r2 #h.cadd(r1,r2)

    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def remove_norm(self):
        self.net1.remove_norm()
        self.net2.remove_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()


def SkipNet(net1, net2, ffnn, **kargs):
    return Seq(Skip(net1,net2), FFNN(ffnn, **kargs))

def BasicBlock(in_planes, planes, stride=1, **kargs):
    block = Seq( Conv(planes, kernel_size = 3, stride = stride, padding = 1, bias=False, normal=True, **kargs)
               , Conv2D(planes, kernel_size = 3, stride = 1, padding = 1, bias=False, normal=True, **kargs))

    if stride != 1 or in_planes != planes:
        block = ParSum(block, Conv2D(planes, kernel_size=1, stride=stride, bias=False, normal=True, **kargs))
    return Seq(block, Activation(**kargs))


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


    
