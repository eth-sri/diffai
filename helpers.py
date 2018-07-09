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

import future
import builtins
import past
import six

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from forbiddenfruit import curse
#from torch.autograd import Variable

from timeit import default_timer as timer

class Timer:
    def __init__(self, activity = None, unit_name = None, units = 1, shouldPrint = True):
        self.activity = activity
        self.unit_name = unit_name
        self.units = units
        self.shouldPrint = shouldPrint
    def __enter__(self):
        self.start = timer()
        return self
    def getUnitTime(self):
        return (self.end - self.start) / self.units

    def __str__(self):
        return "Avg time to " + self.activity +" a "+self.unit_name+ ": "+str(self.getUnitTime())

    def __exit__(self, *args):
        self.end = timer()
        if self.shouldPrint:
            print(self)
            
def cudify(x):
    if use_cuda:
        return x.cuda(async=True)
    return x

def pyval(a, **kargs):
    return dtype([a], **kargs)

def ifThenElse(cond, a, b):
    cond = cond.float()
    return cond * a + (1 - cond) * b

def ifThenElseL(cond, a, b):
    return cond * a + (1 - cond) * b

def product(it):
    if isinstance(it,int):
        return it
    product = 1
    for x in it:
        if x >= 0:
            product *= x
    return product

def getEi(batches, num_elem):
    return eye(num_elem).expand(batches, num_elem,num_elem).permute(1,0,2)


def one_hot(batch,d):
    bs = batch.size()[0]
    indexes = [ list(range(bs)), batch]
    values = [ 1 for _ in range(bs) ]
    return cudify(torch.sparse.FloatTensor(ltypeCPU(indexes), dtypeCPU(values), torch.Size([bs,d])))

def seye(n, m = None): 
    if m is None:
        m = n
    mn = n if n < m else m
    indexes = [[ i for i in range(mn) ], [ i  for i in range(mn) ] ]
    values = [1 for i in range(mn) ]
    return cudify(torch.sparse.FloatTensor(ltypeCPU(indexes), dtypeCPU(values), torch.Size([n,m])))

dtypeCPU = torch.FloatTensor 
ltypeCPU = torch.LongTensor 


if torch.cuda.is_available() and not 'NOCUDA' in os.environ:
    print("using cuda")
    cuda_async = True
    dtype = lambda *args, **kargs: torch.cuda.FloatTensor(*args, **kargs).cuda(async=cuda_async)
    ltype = lambda *args, **kargs: torch.cuda.LongTensor(*args, **kargs).cuda(async=cuda_async)
    ones = lambda *args, **cargs: torch.ones(*args, **cargs).cuda(async=cuda_async)
    zeros = lambda *args, **cargs: torch.zeros(*args, **cargs).cuda(async=cuda_async)
    eye = lambda *args, **cargs: torch.eye(*args, **cargs).cuda(async=cuda_async)
    use_cuda = True
    print("set up cuda")
else:
    print("not using cuda")
    dtype = lambda *args, **kargs: torch.FloatTensor(*args, **kargs)
    ltype = lambda *args, **kargs: torch.LongTensor (*args, **kargs)
    ones = torch.ones
    zeros = torch.zeros
    eye = torch.eye
    use_cuda = False


def smoothmax(x, alpha, dim = 0):
    return x.mul(F.softmax(x * alpha, dim)).sum(dim + 1)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def flat(lst):
    lst_ = []
    for l in lst:
        lst_ += l
    return lst_


def printBoth(st, f = None):
    print(st)
    if not f is None:
        print(st, file=f)


def hasMethod(cl, mt):
    return callable(getattr(cl, mt, None))

def getMethodNames(Foo): 
    return [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]

def getMethods(Foo): 
    return [getattr(Foo, m) for m in getMethodNames(Foo)]

max_c_for_norm = 10000

def numel(arr):
    return product(arr.size())

def variable(Pt):
    class Point:
        def softplus(self): 
            return F.softplus(self)

        def log_softmax(self, *args, dim = 1, **kargs): 
            return F.log_softmax(self, dim = dim, *args, **kargs)
        def conv3d(self, *args, **kargs): 
            return F.conv3d(self, *args, **kargs)
        def conv2d(self, *args, **kargs): 
            return F.conv2d(self, *args, **kargs)
        def max_pool2d(self, *args, **kargs): 
            return F.max_pool2d(self, *args, **kargs)
        def conv1d(self, *args, **kargs): 
            return F.conv1d(self, *args, **kargs)
        def conv1d(self, *args, **kargs): 
            return F.conv1d(self, *args, **kargs)
        def cat(self, other, dim = 0, **kargs): 
            return torch.cat((self, other), dim = dim, **kargs)

        @staticmethod    
        def attack(model, radius, original, target): 
            return original
        @staticmethod    
        def box(original, *args, **kargs):
            return original
        @staticmethod    
        def line(original, other, w= None, *args, **kargs): 
            return (original + other) / 2

        def diameter(self): 
            return pyval(0) # dtype([[[10000]]])

        def lb(self): 
            return self
        def ub(self): 
            return self

        def cudify(self, cuda, cuda_async):
            return self.cuda(async=cuda_async) if cuda else self

    for nm in getMethodNames(Point):
        curse(Pt, nm, getattr(Point, nm))

variable(torch.autograd.Variable)
variable(torch.cuda.FloatTensor)
variable(torch.FloatTensor)
variable(torch.ByteTensor)
variable(torch.Tensor)

dtype.box = torch.Tensor.box
dtype.line = torch.Tensor.line
dtype.attack = torch.Tensor.attack


def default(dic, nm, d):
    if dic is not None and nm in dic:
        return dic[nm]
    return d




def softmaxBatchNP(x, epsilon, subtract = False):
    """Compute softmax values for each sets of scores in x."""
    x = x.astype(np.float32)
    ex = x / epsilon if epsilon is not None else x
    if subtract:
        ex -= ex.max(axis=1)[:,np.newaxis]    
    e_x = np.exp(ex)
    sm = (e_x / e_x.sum(axis=1)[:,np.newaxis])
    am = np.argmax(x, axis=1)
    bads = np.logical_not(np.isfinite(sm.sum(axis = 1)))

    if epsilon is None:
        sm[bads] = 0
        sm[bads, am[bads]] = 1
    else:
        epsilon *= (x.shape[1] - 1) / x.shape[1]
        sm[bads] = epsilon / (x.shape[1] - 1)
        sm[bads, am[bads]] = 1 - epsilon

    sm /= sm.sum(axis=1)[:,np.newaxis]
    return sm


def cadd(a,b):
    both = a.cat(b)
    a, b = both.split(a.size()[0])
    return a + b
