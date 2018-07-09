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
import helpers as h

Point = h.dtype

class FGSM(object):
    @staticmethod    
    def attack(model, epsilon, x, target):
        xn = Point(x.data)
        xn.requires_grad_()
        model.optimizer.zero_grad()
        loss = model.stdLoss(xn, None, target).sum()
        loss.backward()
        r = x + Point(epsilon * torch.sign(xn.grad.data))
        model.optimizer.zero_grad()
        return r

class PGD(object):
    @staticmethod    
    def attack(model, epsilon, xo, target, k = 5):
        epsilon /= k
        x = xo
        for _ in range(k):
            x = FGSM.attack(model, epsilon, x, target)
        return x

class LPGD(object):
    @staticmethod
    def attack(*args):
        return PGD.attack(*args, k = 20)

class MI_FGSM(object):
    @staticmethod    
    def attack(model, epsilon, x, target, k = 20, mu = 0.5):
        epsilon /= k
        x = Point(x.data, requires_grad=True)
        gradorg = Point(h.zeros(x.shape))
        for _ in range(k):
            model.optimizer.zero_grad()
            loss = model.stdLoss(x, None, target).sum()
            loss.backward()
            oth = x.grad / torch.norm(x.grad, p=1)
            gradorg = gradorg * mu + oth
            x.data = (x + epsilon * torch.sign(gradorg)).data
        return x

class HBox(object):

    def creluBoxy(self):
        if self.errors is None:
            if self.beta is None:
                return self.new(F.relu(self.head), None, None)
            er = self.beta 
            mx = F.relu(self.head + er)
            mn = F.relu(self.head - er)
            return self.new((mn + mx) / 2, (mx - mn) / 2 , None)
    
        aber = torch.abs(self.errors)
    
        sm = torch.sum(aber, 0) 
    
        if not self.beta is None:
            sm += self.beta
    
        mx = self.head + sm
        mn = self.head - sm
        
        should_box = mn.lt(0) * mx.gt(0)
        gtz = self.head.gt(0).float()
        mx /= 2
        newhead = h.ifThenElse(should_box, mx, gtz * self.head)
        newbeta = h.ifThenElse(should_box, mx, gtz * (self.beta if not self.beta is None else 0))
        newerr = (1 - should_box.float()) * gtz * self.errors
    
        return self.new(newhead, newbeta , newerr)
    
    def creluSwitch(self):
        if self.errors is None:
            if self.beta is None:
                return self.new(F.relu(self.head), None, None)
            er = self.beta 
            mx = F.relu(self.head + er)
            mn = F.relu(self.head - er)
            return self.new((mn + mx) / 2, (mx - mn) / 2 , None)
    
        aber = torch.abs(self.errors)
    
        sm = torch.sum(aber, 0) 
    
        if not self.beta is None:
            sm += self.beta
    
        mn = self.head - sm
        mx = sm
        mx += self.head
        
        should_box = mn.lt(0) * mx.gt(0)
        gtz = self.head.gt(0)
        
        mn.neg_()
        should_boxer = mn.gt(mx)

        mn /= 2
        newhead = h.ifThenElse(should_box, h.ifThenElse(should_boxer, mx / 2, self.head + mn ), gtz.float() * self.head)
        zbet =  self.beta if not self.beta is None else 0
        newbeta = h.ifThenElse(should_box, h.ifThenElse(should_boxer, mx / 2, mn + zbet), gtz.float() * zbet)
        newerr  = h.ifThenElseL(should_box, 1 - should_boxer, gtz).float() * self.errors
    
        return self.new(newhead, newbeta , newerr)


    def creluSmooth(self):
        if self.errors is None:
            if self.beta is None:
                return self.new(F.relu(self.head), None, None)
            er = self.beta 
            mx = F.relu(self.head + er)
            mn = F.relu(self.head - er)
            return self.new((mn + mx) / 2, (mx - mn) / 2 , None)
    
        aber = torch.abs(self.errors)
    
        sm = torch.sum(aber, 0) 
    
        if not self.beta is None:
            sm += self.beta
    
        mn = self.head - sm
        mx = sm
        mx += self.head
        

        nmn = F.relu(-1 * mn)
                
        zbet =  (self.beta if not self.beta is None else 0)
        newheadS = self.head + nmn / 2
        newbetaS = zbet + nmn / 2
        newerrS = self.errors

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

        return self.new(newhead, newbeta , newerr)

    def customRelu(self):
        return self.creluBoxy()

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
        assert type(flt) in [float, int]
        return self.new(self.head * flt, None if self.beta is None else self.beta * flt, None if self.errors is None else self.errors * flt)
    
    def __truediv__(self, flt):
        assert type(flt) in [float, int]
        flt = 1. / flt
        return self.new(self.head * flt, None if self.beta is None else self.beta * flt, None if self.errors is None else self.errors * flt)

    def __add__(self, other):
        if isinstance(other, HBox):
            return self.new(self.head + other.head, other.beta if self.beta is None else (self.beta if other.beta is None else self.beta + other.beta), other.errors if self.errors is None else (self.errors if other.errors is None else self.errors + other.errors))
        else:
            # other has to be a standard variable or tensor
            return self.new(self.head + other, self.beta, self.errors)

    def __sub__(self, other):
        if isinstance(other, HBox):
            return self.new(self.head - other.head
                            , other.beta if self.beta is None else (self.beta if other.beta is None else self.beta + other.beta),
                            (None if other.errors is None else -other.errors) if self.errors is None else (self.errors if other.errors is None else self.errors - other.errors))
        else:
            # other has to be a standard variable or tensor
            return self.new(self.head - other, self.beta, self.errors)

    def bmm(self, other):
        
        hd = self.head.unsqueeze(1).bmm(other).squeeze(1)
        bet = None if self.beta is None else self.beta.unsqueeze(1).bmm(other.abs()).squeeze(1)

        if self.errors is None:
            er = None
        else:
            bigOther = other.expand(self.errors.size()[0], -1, -1, -1)
            h = self.errors
            inter = h.view(-1, *h.size()[2:]).unsqueeze(1)
            bigOther = bigOther.contiguous().view(-1, *bigOther.size()[2:])
            er = inter.bmm(bigOther)
            er = er.view(*h.size()[:-1], -1)
        return self.new(hd, bet, er)

    def conv(self, conv, weight, bias = None, stride = None, **kargs):
        h = self.errors
        inter = h if h is None else h.view(-1, *h.size()[2:])
        hd = conv(self.head, weight, bias, stride = stride, **kargs)
        res = h if h is None else conv(inter, weight, bias=None, stride = stride, **kargs)

        return self.new( hd
                       , None if self.beta is None else conv(self.beta, weight.abs(), bias = None, stride = stride, **kargs)
                       , h if h is None else res.view(h.size()[0], h.size()[1], *res.size()[1:]))

    def conv1d(self, *args, **kargs):
        return self.conv(F.conv1d, *args, **kargs)
                   
    def conv2d(self, *args, **kargs):
        return self.conv(F.conv2d, *args, **kargs)                   

    def conv3d(self, *args, **kargs):
        return self.conv(F.conv3d, *args, **kargs)
        
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
    
    def sum(self):
        return self.new(torch.sum(self.head,1), None if self.beta is None else torch.sum(self.beta,1), None if self.errors is None else torch.sum(self.errors, 2))

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

        def catNonNullErrors(er1, er2): # the way of things is ugly
            erS, erL = (er1, er2)
            sS, sL = (erS.size()[0], erL.size()[0])

            if sS == sL: # here we know we used transformers on either side which didnt introduce new error terms (this is a hack).
                return erS.cat(erL, dim + 1)

            extrasS = h.zeros([sL] + list(erS.size()[1:]))
            extrasL = h.zeros([sS] + list(erL.size()[1:]))

            erL = torch.cat((extrasL, erL), dim=0)
            erS = torch.cat((erS, extrasS), dim=0)

            return erS.cat(erL, dim + 1)
        
        return self.new(torch.cat((self.head, other.head), dim = dim), 
                        other.beta   if self.beta   is None else (self.beta if other.beta is None else # this cant work here, other.errors has the wrong shape 
                                                                  torch.cat((self.beta, other.beta), dim = dim)),
                        other.errors if self.errors is None else (self.errors if other.errors is None else
                                                                  catNonNullErrors(self.errors, other.errors)))


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

    def lb(self):
        return self.head - torch.sum(self.concreteErrors().abs(), 0)

    def ub(self):
        return self.head + torch.sum(self.concreteErrors().abs(), 0)        

    def size(self):
        return self.head.size()

    def diameter(self):
        abal = torch.abs(self.concreteErrors()).permute(1,0,2)
        sm = torch.sum(abal, 1)
        return torch.sum(sm, 1) # perimeter

    @staticmethod
    def box(original, radius):
        """
        This version of it is slow, but keeps correlation down the line.
        """
        batches = original.size()[0]
        num_elem = h.product(original.size()[1:])
        ei = h.getEi(batches,num_elem)
        
        if len(original.size()) > 2:
            ei = ei.contiguous().view(num_elem, *original.size())

        return HBox(original, None, ei * radius).checkSizes()

    @staticmethod
    def line(o1, o2, w = None):
        ln = ((o2 - o1) / 2).unsqueeze(0)
        if not w is None and w > 0.0:
            batches = o1.size()[0]
            num_elem = h.product(o1.size()[1:])
            ei = h.getEi(batches,num_elem)
            if len(o1.size()) > 2:
                ei = ei.contiguous().view(num_elem, *o1.size())
            ln = torch.cat([ln, ei * w])
        return HBox((o1 + o2) / 2, None, ln ).checkSizes()

class Box(HBox):
    def __init__(self, *args, **kargs):
        super(Box, self).__init__(*args, **kargs)

    @staticmethod
    def box(original, diameter):  
        """
        This version of it takes advantage of betas being uncorrelated.  
        Unfortunately they stay uncorrelated forever.  
        Counterintuitively, tests show more accuracy - this is because the other box
        creates lots of 0 errors which get accounted for by the calcultion of the newhead in relu 
        which is apparently worse than not accounting for errors.
        """
        return Box(original, h.ones(original.size()) * diameter, None).checkSizes()
    
    @staticmethod
    def line(o1, o2, w = None):
        return Box((o1 + o2) / 2, ((o2 - o1) / 2).abs(), None).checkSizes()


class ZBox(HBox):
    def __init__(self, *args, **kargs):
        super(ZBox, self).__init__(*args, **kargs)

    @staticmethod
    def copy(hbox):
        return ZBox(hbox.head, hbox.beta, hbox.errors)
    
    @staticmethod
    def box(*args, **kargs):
        return ZBox.copy(HBox.box(*args, **kargs))

    @staticmethod
    def line(*args, **kargs):
        return ZBox.copy(HBox.line(*args, **kargs))

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
        return self.applySuper(super(ZBox,self).softplus())

    def relu(self):
        return self.applySuper(super(ZBox,self).relu())


class HSwitch(HBox):
    def __init__(self, *args, **kargs):
        super(HSwitch, self).__init__(*args, **kargs)

    def customRelu(self):
        return self.creluSwitch()

    @staticmethod
    def copy(hbox):
        return HSwitch(hbox.head, hbox.beta, hbox.errors)
    
    @staticmethod
    def box(*args, **kargs):
        return HSwitch.copy(HBox.box(*args, **kargs))

    @staticmethod
    def line(*args, **kargs):
        return HSwitch.copy(HBox.line(*args, **kargs))
    
class ZSwitch(ZBox):
    def __init__(self, *args, **kargs):
        super(ZSwitch, self).__init__(*args, **kargs)

    def customRelu(self):
        return self.creluSwitch()

    @staticmethod
    def copy(hbox):
        return ZSwitch(hbox.head, hbox.beta, hbox.errors)
    
    @staticmethod
    def box(*args, **kargs):
        return ZSwitch.copy(HBox.box(*args, **kargs))

    @staticmethod
    def line(*args, **kargs):
        return ZSwitch.copy(HBox.line(*args, **kargs))

class HSmooth(HBox):
    def __init__(self, *args, **kargs):
        super(HSmooth, self).__init__(*args, **kargs)

    def customRelu(self):
        return self.creluSmooth()

    @staticmethod
    def copy(hbox):
        return HSmooth(hbox.head, hbox.beta, hbox.errors)
    
    @staticmethod
    def box(*args, **kargs):
        return HSmooth.copy(HBox.box(*args, **kargs))

    @staticmethod
    def line(*args, **kargs):
        return HSmooth.copy(HBox.line(*args, **kargs))


    
class ZSmooth(ZBox):
    def __init__(self, *args, **kargs):
        super(ZSmooth, self).__init__(*args, **kargs)

    def customRelu(self):
        return self.creluSmooth()

    @staticmethod
    def copy(hbox):
        return ZSmooth(hbox.head, hbox.beta, hbox.errors)
    
    @staticmethod
    def box(*args, **kargs):
        return ZSmooth.copy(HBox.box(*args, **kargs))

    @staticmethod
    def line(*args, **kargs):
        return ZSmooth.copy(HBox.line(*args, **kargs))



