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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd

try:
    from . import helpers as h
except:
    import helpers as h

def catNonNullErrors(op): # the way of things is ugly
    def doop(er1, er2):
        erS, erL = (er1, er2)
        sS, sL = (erS.size()[0], erL.size()[0])

        if sS == sL: # TODO: here we know we used transformers on either side which didnt introduce new error terms (this is a hack for hybrid zonotopes and doesn't work with adaptive error term adding).
            return op(erS,erL)

        extrasS = h.zeros([sL] + list(erS.size()[1:]))
        extrasL = h.zeros([sS] + list(erL.size()[1:]))

        erL = torch.cat((extrasL, erL), dim=0)
        erS = torch.cat((erS, extrasS), dim=0)

        return op(erS,erL)
    return doop

class HybridZonotope:

    def isSafe(self, target):
        od,_ = torch.min(h.preDomRes(self,target).lb(), 1)
        return od.gt(0.0).long()

    def labels(self):
        target = torch.max(self.ub(), 1)[1]
        l = list(h.preDomRes(self,target).lb()[0])
        return [target.item()] + [ i for i,v in zip(range(len(l)), l) if v <= 0]

    def customRelu(self):
        return creluBoxy(self)

    def relu(self):
        return self.customRelu()
    
    def __init__(self, head, beta, errors):
        self.head = head
        self.errors = errors
        self.beta = beta

    def checkSizes(self):
        if not self.errors is None:
            if not self.errors.size()[1:] == self.head.size():
                raise Exception("Such bad sizes on error")
        if not self.beta is None:
            if not self.beta.size() == self.head.size():
                raise Exception("Such bad sizes on beta")
            if self.beta.lt(0.0).any():
                #raise Exception("Beta Below Zero")
                self.beta.abs_()
            
        return self
    
    def new(self, *args, **kargs):
        return self.__class__(*args, **kargs).checkSizes()

    def __mul__(self, flt):
        return self.new(self.head * flt, None if self.beta is None else self.beta * abs(flt), None if self.errors is None else self.errors * flt)
    
    def __truediv__(self, flt):
        flt = 1. / flt
        return self.new(self.head * flt, None if self.beta is None else self.beta * abs(flt), None if self.errors is None else self.errors * flt)

    def __add__(self, other):
        if isinstance(other, HybridZonotope):
            return self.new(self.head + other.head, h.msum(self.beta, other.beta, lambda a,b: a + b), h.msum(self.errors, other.errors, catNonNullErrors(lambda a,b: a + b)))
        else:
            # other has to be a standard variable or tensor
            return self.new(self.head + other, self.beta, self.errors)

    def __sub__(self, other):
        if isinstance(other, HybridZonotope):
            return self.new(self.head - other.head
                            , h.msum(self.beta, other.beta, lambda a,b: a + b)
                            , h.msum(self.errors, None if other.errors is None else -other.errors, catNonNullErrors(lambda a,b: a + b)))
        else:
            # other has to be a standard variable or tensor
            return self.new(self.head - other, self.beta, self.errors)

    def bmm(self, other):
        hd = self.head.bmm(other)
        bet = None if self.beta is None else self.beta.bmm(other.abs())

        if self.errors is None:
            er = None
        else:
            er = self.errors.matmul(other)
        return self.new(hd, bet, er)

    def conv(self, conv, weight, bias = None, **kargs):
        h = self.errors
        inter = h if h is None else h.view(-1, *h.size()[2:])
        hd = conv(self.head, weight, bias=bias, **kargs)
        res = h if h is None else conv(inter, weight, bias=None, **kargs)

        return self.new( hd
                       , None if self.beta is None else conv(self.beta, weight.abs(), bias = None, **kargs)
                       , h if h is None else res.view(h.size()[0], h.size()[1], *res.size()[1:]))

    def conv1d(self, *args, **kargs):
        return self.conv(lambda x, *args, **kargs: x.conv1d(*args,**kargs), *args, **kargs)
                   
    def conv2d(self, *args, **kargs):
        return self.conv(lambda x, *args, **kargs: x.conv2d(*args,**kargs), *args, **kargs)                   

    def conv3d(self, *args, **kargs):
        return self.conv(lambda x, *args, **kargs: x.conv3d(*args,**kargs), *args, **kargs)

    def conv_transpose1d(self, *args, **kargs):
        return self.conv(lambda x, *args, **kargs: x.conv_transpose1d(*args,**kargs), *args, **kargs)
                   
    def conv_transpose2d(self, *args, **kargs):
        return self.conv(lambda x, *args, **kargs: x.conv_transpose2d(*args,**kargs), *args, **kargs)                   

    def conv_transpose3d(self, *args, **kargs):
        return self.conv(lambda x, *args, **kargs: x.conv_transpose3d(*args,**kargs), *args, **kargs)
        
    def matmul(self, other):
        return self.new(self.head.matmul(other), None if self.beta is None else self.beta.matmul(other.abs()), None if self.errors is None else self.errors.matmul(other))

    def unsqueeze(self, i):
        return self.new(self.head.unsqueeze(i), None if self.beta is None else self.beta.unsqueeze(i), None if self.errors is None else self.errors.unsqueeze(i + 1))

    def squeeze(self, dim):
        return self.new(self.head.squeeze(dim),
                        None if self.beta is None else self.beta.squeeze(dim),
                        None if self.errors is None else self.errors.squeeze(dim + 1 if dim >= 0 else dim))    

    def float(self):
        return self # if we weren't already a float theres a problem
    
    def sum(self, dim=1):
        return self.new(torch.sum(self.head,dim=dim), None if self.beta is None else torch.sum(self.beta,dim=dim), None if self.errors is None else torch.sum(self.errors, dim= dim + 1 if dim >= 0 else dim))

    def view(self,*newshape):
        return self.new(self.head.view(*newshape), 
                        None if self.beta is None else self.beta.view(*newshape),
                        None if self.errors is None else self.errors.view(self.errors.size()[0], *newshape))

    def gather(self,dim, index):
        return self.new(self.head.gather(dim, index), 
                        None if self.beta is None else self.beta.gather(dim, index),
                        None if self.errors is None else self.errors.gather(dim + 1, index.expand([self.errors.size()[0]] + list(index.size()))))
    
    def concretize(self):
        if self.errors is None:
            return self

        return self.new(self.head, torch.sum(self.concreteErrors().abs(),0), None) # maybe make a box?
    
    def cat(self,other, dim=0):
        return self.new(self.head.cat(other.head, dim = dim), 
                        h.msum(other.beta, self.beta, lambda a,b: a.cat(b, dim = dim)),
                        h.msum(self.errors, other.errors, catNonNullErrors(lambda a,b: a.cat(b, dim+1))))


    def split(self, split_size, dim = 0):
        heads = list(self.head.split(split_size, dim))
        betas = list(self.beta.split(split_size, dim)) if not self.beta is None else None
        errorss = list(self.errors.split(split_size, dim + 1)) if not self.errors is None else None
        
        def makeFromI(i):
            return self.new( heads[i], 
                             None if betas is None else betas[i], 
                             None if errorss is None else errorss[i])
        return tuple(makeFromI(i) for i in range(len(heads)))

        
    
    def concreteErrors(self):
        if self.beta is None and self.errors is None:
            raise Exception("shouldn't have both beta and errors be none")
        if self.errors is None:
            return self.beta.unsqueeze(0)
        if self.beta is None:
            return self.errors
        return torch.cat([self.beta.unsqueeze(0),self.errors], dim=0)

    def softplus(self):
        if self.errors is None:
            if self.beta is None:
                return self.new(F.softplus(self.head), None , None)
            tp = F.softplus(self.head + self.beta)
            bt = F.softplus(self.head - self.beta)
            return self.new((tp + bt) / 2, (tp - bt) / 2 , None)

        errors = self.concreteErrors()
        o = h.ones(self.head.size())

        def sp(hd):
            return F.softplus(hd) # torch.log(o + torch.exp(hd))  # not very stable
        def spp(hd):
            ehd = torch.exp(hd)
            return ehd.div(ehd + o)
        def sppp(hd):
            ehd = torch.exp(hd)
            md = ehd + o
            return ehd.div(md.mul(md))

        fa = sp(self.head)
        fpa = spp(self.head)

        a = self.head

        k = torch.sum(errors.abs(), 0) 

        def evalG(r):
            return r.mul(r).mul(sppp(a + r))

        m = torch.max(evalG(h.zeros(k.size())), torch.max(evalG(k), evalG(-k)))
        m = h.ifThenElse( a.abs().lt(k), torch.max(m, torch.max(evalG(a), evalG(-a))), m)
        m /= 2
        
        return self.new(fa, m if self.beta is None else m + self.beta.mul(fpa), None if self.errors is None else self.errors.mul(fpa))

    def center(self):
        return self.head

    def vanillaTensorPart(self):
        return self.head

    def lb(self):
        return self.head - torch.sum(self.concreteErrors().abs(), 0)

    def ub(self):
        return self.head + torch.sum(self.concreteErrors().abs(), 0)        

    def size(self):
        return self.head.size()

    def diameter(self):
        abal = torch.abs(self.concreteErrors()).transpose(0,1)
        return abal.sum(1).sum(1) # perimeter




def creluBoxy(dom):
    if dom.errors is None:
        if dom.beta is None:
            return dom.new(F.relu(dom.head), None, None)
        er = dom.beta 
        mx = F.relu(dom.head + er)
        mn = F.relu(dom.head - er)
        return dom.new((mn + mx) / 2, (mx - mn) / 2 , None)

    aber = torch.abs(dom.errors)

    sm = torch.sum(aber, 0) 

    if not dom.beta is None:
        sm += dom.beta

    mx = dom.head + sm
    mn = dom.head - sm

    should_box = mn.lt(0) * mx.gt(0)
    gtz = dom.head.gt(0).float()
    mx /= 2
    newhead = h.ifThenElse(should_box, mx, gtz * dom.head)
    newbeta = h.ifThenElse(should_box, mx, gtz * (dom.beta if not dom.beta is None else 0))
    newerr = (1 - should_box.float()) * gtz * dom.errors

    return dom.new(newhead, newbeta , newerr)

def creluSwitch(dom):
    if dom.errors is None:
        if dom.beta is None:
            return dom.new(F.relu(dom.head), None, None)
        er = dom.beta 
        mx = F.relu(dom.head + er)
        mn = F.relu(dom.head - er)
        return dom.new((mn + mx) / 2, (mx - mn) / 2 , None)

    aber = torch.abs(dom.errors)

    sm = torch.sum(aber, 0) 

    if not dom.beta is None:
        sm += dom.beta

    mn = dom.head - sm
    mx = sm
    mx += dom.head

    should_box = mn.lt(0) * mx.gt(0)
    gtz = dom.head.gt(0)

    mn.neg_()
    should_boxer = mn.gt(mx)

    mn /= 2
    newhead = h.ifThenElse(should_box, h.ifThenElse(should_boxer, mx / 2, dom.head + mn ), gtz.float() * dom.head)
    zbet =  dom.beta if not dom.beta is None else 0
    newbeta = h.ifThenElse(should_box, h.ifThenElse(should_boxer, mx / 2, mn + zbet), gtz.float() * zbet)
    newerr  = h.ifThenElseL(should_box, 1 - should_boxer, gtz).float() * dom.errors

    return dom.new(newhead, newbeta , newerr)


def creluSmooth(dom):
    if dom.errors is None:
        if dom.beta is None:
            return dom.new(F.relu(dom.head), None, None)
        er = dom.beta 
        mx = F.relu(dom.head + er)
        mn = F.relu(dom.head - er)
        return dom.new((mn + mx) / 2, (mx - mn) / 2 , None)

    aber = torch.abs(dom.errors)

    sm = torch.sum(aber, 0) 

    if not dom.beta is None:
        sm += dom.beta

    mn = dom.head - sm
    mx = sm
    mx += dom.head


    nmn = F.relu(-1 * mn)

    zbet =  (dom.beta if not dom.beta is None else 0)
    newheadS = dom.head + nmn / 2
    newbetaS = zbet + nmn / 2
    newerrS = dom.errors

    mmx = F.relu(mx)

    newheadB = mmx / 2
    newbetaB = newheadB
    newerrB = 0

    eps = 0.0001
    t = nmn / (mmx + nmn + eps) # mn.lt(0).float() * F.sigmoid(nmn - nmx)

    shouldnt_zero = mx.gt(0).float()

    newhead = shouldnt_zero * ( (1 - t) * newheadS + t * newheadB)
    newbeta = shouldnt_zero * ( (1 - t) * newbetaS + t * newbetaB)
    newerr =  shouldnt_zero * ( (1 - t) * newerrS  + t * newerrB)

    return dom.new(newhead, newbeta , newerr)


def creluNIPS(dom):
    if dom.errors is None:
        if dom.beta is None:
            return dom.new(F.relu(dom.head), None, None)
        er = dom.beta 
        mx = F.relu(dom.head + er)
        mn = F.relu(dom.head - er)
        return dom.new((mn + mx) / 2, (mx - mn) / 2 , None)
    
    aber = torch.abs(dom.errors)

    sm = torch.sum(aber, 0) 

    if not dom.beta is None:
        sm += dom.beta

    mn = dom.head - sm
    mx = sm
    mx += dom.head

    mngz = mn >= 0

    zs = h.zeros(dom.head.shape)

    lam = torch.where(mx > 0, mx / (mx - mn), zs)
    mu = lam * mn * (-0.5)

    betaz = zs if dom.beta is None else dom.beta 
    
    newhead = torch.where(mngz, dom.head , lam * dom.head + mu)
    newbeta = torch.where(mngz, betaz    , lam * betaz + mu ) # mu is always positive on this side
    newerr = torch.where(mngz, dom.errors, lam * dom.errors )
    return dom.new(newhead, newbeta, newerr)


class Zonotope(HybridZonotope):
    def applySuper(self, ret):
        batches = ret.head.size()[0]
        num_elem = h.product(ret.head.size()[1:])
        ei = h.getEi(batches, num_elem)

        if len(ret.head.size()) > 2:
            ei = ei.contiguous().view(num_elem, *ret.head.size())

        ret.errors = torch.cat([ ret.errors, ei * ret.beta ]) if not ret.beta is None else ret.errors
        ret.beta = None
        return ret.checkSizes()
    
    def softplus(self):
        return self.applySuper(super(Zonotope,self).softplus())

    def relu(self):
        return self.applySuper(super(Zonotope,self).relu())


def mysign(x):
    e = h.dtype(x.eq(0).float())
    r = h.dtype(x.sign())
    return r + e

def mulIfEq(grad,out,target):
    pred = out.max(1, keepdim=True)[1]
    is_eq = pred.eq(target.view_as(pred)).float()
    is_eq = is_eq.view([-1] + [1 for _ in grad.size()[1:]]).expand_as(grad)
    return is_eq
    
def stdLoss(out, target):
    return F.nll_loss(out.log_softmax(dim=1), target, size_average = False, reduce = False).sum()

class ListDomain(object):

    def __init__(self, al):
        self.al = list(al)

    def isSafe(self,*args,**kargs):
        raise "Domain Not Suitable For Testing"

    def labels(self):
        raise "Domain Not Suitable For Testing"

    def __mul__(self, flt):
        return ListDomain(a.__mul__(flt) for a in self.al)

    def __truediv__(self, flt):
        return ListDomain(a.__truediv__(flt) for a in self.al)

    def __add__(self, other):
        if isinstance(other, ListDomain):
            return ListDomain(a.__add__(o) for a,o in zip(self.al, other.al))
        else:
            return ListDomain(a.__add__(other) for a in self.al)

    def __sub__(self, other):
        if isinstance(other, ListDomain):
            return ListDomain(a.__sub__(o) for a,o in zip(self.al, other.al))
        else:
            return ListDomain(a.__sub__(other) for a in self.al)

    def bmm(self, other):
        return ListDomain(a.bmm(other) for a in self.al)

    def matmul(self, other):
        return ListDomain(a.matmul(other) for a in self.al)

    def conv(self, *args, **kargs):
        return ListDomain(a.conv(*args, **kargs) for a in self.al)

    def conv1d(self, *args, **kargs):
        return ListDomain(a.conv1d(*args, **kargs) for a in self.al)

    def conv2d(self, *args, **kargs):
        return ListDomain(a.conv2d(*args, **kargs) for a in self.al)

    def conv3d(self, *args, **kargs):
        return ListDomain(a.conv3d(*args, **kargs) for a in self.al)

    def unsqueeze(self, *args, **kargs):
        return ListDomain(a.unsqueeze(*args, **kargs) for a in self.al)

    def squeeze(self, *args, **kargs):
        return ListDomain(a.squeeze(*args, **kargs) for a in self.al)

    def view(self, *args, **kargs):
        return ListDomain(a.view(*args, **kargs) for a in self.al)

    def gather(self, *args, **kargs):
        return ListDomain(a.gather(*args, **kargs) for a in self.al)

    def sum(self, *args, **kargs):
        return ListDomain(a.sum(*args,**kargs) for a in self.al)

    def concretize(self):
        return ListDomain(a.concretize() for a in self.al)

    def float(self):
        return ListDomain(a.float() for a in self.al)

    def vanillaTensorPart(self):
        return self.al[0].vanillaTensorPart()

    def center(self):
        return ListDomain(a.center() for a in self.al)

    def ub(self):
        return ListDomain(a.ub() for a in self.al)

    def lb(self):
        return ListDomain(a.lb() for a in self.al)

    def relu(self):
        return ListDomain(a.relu() for a in self.al)

    def softplus(self):
        return ListDomain(a.softplus() for a in self.al)


    def cat(self, other, *args, **kargs):
        return ListDomain(a.cat(o, *args, **kargs) for a,o in zip(self.al, other.al))


    def split(self, *args, **kargs):
        return [ListDomain(*z) for z in zip(a.split(*args, **kargs) for a in self.al)]

    def size(self):
        return self.al[0].size()





class TaggedDomain(object):

    def __init__(self, a, tag):
        self.tag = tag
        self.a = a

    def isSafe(self,*args,**kargs):
        raise "Domain Not Suitable For Testing"

    def labels(self):
        raise "Domain Not Suitable For Testing"

    def __mul__(self, flt):
        return TaggedDomain(self.a.__mul__(flt), self.tag)

    def __truediv__(self, flt):
        return TaggedDomain(self.a.__truediv__(flt), self.tag)

    def __add__(self, other):
        if isinstance(other, TaggedDomain):
            return TaggedDomain(self.a.__add__(other.a), self.tag)
        else:
            return TaggedDomain(self.a.__add__(other), self.tag)

    def __sub__(self, other):
        if isinstance(other, TaggedDomain):
            return TaggedDomain(self.a.__sub__(other.a), self.tag)
        else:
            return TaggedDomain(self.a.__sub__(other.a), self.tag)

    def bmm(self, other):
        return TaggedDomain(self.a.bmm(other), self.tag)

    def matmul(self, other):
        return TaggedDomain(self.a.matmul(other), self.tag)

    def conv(self, *args, **kargs):
        return TaggedDomain(self.a.conv(*args, **kargs) , self.tag)

    def conv1d(self, *args, **kargs):
        return TaggedDomain(self.a.conv1d(*args, **kargs), self.tag)

    def conv2d(self, *args, **kargs):
        return TaggedDomain(self.a.conv2d(*args, **kargs), self.tag)

    def conv3d(self, *args, **kargs):
        return TaggedDomain(self.a.conv3d(*args, **kargs), self.tag)

    def unsqueeze(self, *args, **kargs):
        return TaggedDomain(self.a.unsqueeze(*args, **kargs), self.tag)

    def squeeze(self, *args, **kargs):
        return TaggedDomain(self.a.squeeze(*args, **kargs), self.tag)

    def view(self, *args, **kargs):
        return TaggedDomain(self.a.view(*args, **kargs), self.tag)

    def gather(self, *args, **kargs):
        return TaggedDomain(self.a.gather(*args, **kargs), self.tag)

    def sum(self, *args, **kargs):
        return TaggedDomain(self.a.sum(*args,**kargs), self.tag)

    def concretize(self):
        return TaggedDomain(self.a.concretize(), self.tag)

    def float(self):
        return TaggedDomain(self.a.float(), self.tag)

    def vanillaTensorPart(self):
        return self.a.vanillaTensorPart()

    def center(self):
        return TaggedDomain(self.a.center(), self.tag)

    def ub(self):
        return TaggedDomain(self.a.ub(), self.tag)

    def lb(self):
        return TaggedDomain(self.a.lb(), self.tag)

    def relu(self):
        return TaggedDomain(self.a.relu(), self.tag)

    def softplus(self):
        return TaggedDomain(self.a.softplus(), self.tag)


    def cat(self, other, *args, **kargs):
        return TaggedDomain(self.a.cat(other.a, *args, **kargs), self.tag)


    def split(self, *args, **kargs):
        return [TaggedDomain(z, self.tag) for z in self.a.split(*args, **kargs)]

    def size(self):
        return self.a.size()
