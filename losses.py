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
import torch.nn.functional as F
import torch.optim as optim

import helpers as h
import domains
from domains import *
import math


POINT_DOMAINS = [m for m in h.getMethods(domains) if h.hasMethod(m, "attack")] + [ torch.FloatTensor, torch.Tensor, torch.cuda.FloatTensor ] 
SYMETRIC_DOMAINS = [domains.Box] + POINT_DOMAINS

def domRes(outDom, target, **args): # TODO: make faster again by keeping sparse tensors sparse
    t = h.one_hot(target.data.long(), outDom.size()[1]).to_dense()
    tmat = t.unsqueeze(2).matmul(t.unsqueeze(1))
    
    tl = t.unsqueeze(2).expand(-1, -1, tmat.size()[1])
    
    inv_t = h.eye(tmat.size()[1]).expand(tmat.size()[0], -1, -1)
    inv_t = inv_t - tmat
    
    tl = tl.bmm(inv_t)
    
    fst = outDom.bmm(tl)
    snd = outDom.bmm(inv_t)
    diff = fst - snd
    return diff.lb() + t

def isSafeDom(outDom, target, **args):
    od,_ = torch.min(domRes(outDom, target, **args), 1)
    return od.gt(0.0).long().item()


def isSafeBox(target, net, inp, eps, dom):
    atarg = target.argmax(1)[0].unsqueeze(0)
    if hasattr(dom, "attack"):
        x = dom.attack(net, eps, inp, target)
        pred = net(x).argmax(1)[0].unsqueeze(0) # get the index of the max log-probability
        return pred.item() == atarg.item()
    else:
        outDom = net(dom.box(inp, eps))
        return isSafeDom(outDom, atarg)
