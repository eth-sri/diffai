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

import math
import numpy as np

try:
    from . import helpers as h
    from . import ai
except:
    import helpers as h
    import ai

class DList(object):
    Domain = ai.ListDomain
    def __init__(self, *al):
        if len(al) == 0:
            al = [("Point()", 1.0), ("Box()", 0.1)]

        self.al = [(eval(a) if type(a) is str else a, float(aw)) for a,aw in al]
        self.div = 1.0 / sum(aw for _,aw in self.al)

    def box(self, *args, **kargs):
        return self.Domain(a.box(*args, **kargs) for a,_ in self.al)

    def boxBetween(self, *args, **kargs):
        return self.Domain(a.box(*args, **kargs) for a,_ in self.al)

    def line(self, *args, **kargs):
        return self.Domain(a.line(*args, **kargs) for a,_ in self.al)

    def loss(self, dom, *args, **kargs):
        return sum(a.loss(ad, *args, **kargs) * aw for (a, aw), ad in zip(self.al, dom.al)) * self.div
        
    def widthLoss(self, dom, **args):
        return sum(a.widthLoss(ad, **args) * aw for (a, aw), ad in zip(self.al, dom.al)) * self.div

    def regLoss(self, dom, *args, **kargs):
        return sum(a.regLoss(ad, *args, **kargs) * aw for (a, aw), ad in zip(self.al, dom.al)) * self.div

    def combinedLoss(self, dom, *args, **kargs):
        return sum(a.combinedLoss(ad, *args, **kargs) * aw for (a, aw), ad in zip(self.al, dom.al)) * self.div

    def __str__(self):
        return "DList(%s)" % h.sumStr("("+str(a)+","+str(w)+")" for a,w in self.al)

class Mix(DList):
    def __init__(self, a="Point()", b="Box()", aw = 1.0, bw = 0.1):
        super(Mix, self).__init__((a,aw), (b,bw))

class DProb(object):
    Domain = ai.TaggedDomain
    def __init__(self, *doms):
        if len(doms) == 0:
            doms = [("Point()", 0.8), ("Box()", 0.2)]
        div = 1.0 / sum(float(aw) for _,aw in doms)
        self.domains = [eval(a) if type(a) is str else a for a,_ in doms]
        self.probs = [ div * float(aw)  for _,aw in doms]

    def chooseDom(self):
        return self.domains[np.random.choice(len(self.domains), p = self.probs)]

    def box(self, *args, **kargs):
        domain = self.chooseDom()
        return self.Domain(domain.box(*args, **kargs), tag = domain)

    def line(self, *args, **kargs):
        domain = self.chooseDom()
        return self.Domain(domain.line(*args, **kargs), tag = domain)

    def loss(self, dom, target, **args):
        return dom.tag.loss(dom.a, target, **args)
        
    def widthLoss(self, dom, **args):
        return dom.tag.widthLoss(dom.a, **args)

    def regLoss(self, dom, *args, **kargs):
        return dom.tag.regLoss(dom.a, *args, **kargs)

    def combinedLoss(self, dom, *args, **kargs):
        return dom.tag.combinedLoss(dom.a, *args, **kargs)

    def __str__(self):
        return "DProb(%s)" % h.sumStr("("+str(a)+","+str(w)+")" for a,w in zip(self.domains, self.probs))

class Coin(DProb):
    def __init__(self, a="Point()", b="Box()", ap = 0.8, bp = 0.2):
        super(Coin, self).__init__((a,ap), (b,bp))

class Point(object):
    Domain = h.dtype
    def __init__(self, **kargs):
        pass

    def box(self, original, *args, **kargs):
        return original

    def line(self, original, other, *args, **kargs): 
        return (original + other) / 2

    def loss(self, dom, target, **args):
        return F.nll_loss(dom.log_softmax(dim=1), target, size_average = False, reduce = False)

    def widthLoss(self, dom, **args):
        return 0
    
    def combinedLoss(self, dom, target, loss_fn, *args, **kargs):
        return loss_fn(dom, target)

    def regLoss(self, dom, target, loss_fn, *args, **kargs):
        return loss_fn(dom, target)

    def boxBetween(self, o1, o2, *args, **kargs):
        return (o1 + o2) / 2

    def __str__(self):
        return "Point()"

class PointA(Point):
    def boxBetween(self, o1, o2, *args, **kargs):
        return o1

    def __str__(self):
        return "PointA()"

class PointB(Point):
    def boxBetween(self, o1, o2, *args, **kargs):
        return o2

    def __str__(self):
        return "PointB()"


class Normal(Point):
    def __init__(self, w = None, **kargs):
        self.epsilon = w
        
    def box(self, original, epsilon, *args, **kargs):
        """ original = mu = mean, epsilon = variance"""
        if not self.epsilon is None:
            epsilon = self.epsilon

        inter = torch.randn_like(original, device = h.device) * epsilon
        return original + inter

    def __str__(self):
        return "Normal(%s)" % ("" if self.epsilon is None else str(self.epsilon))

class MI_FGSM(Point):

    def __init__(self, w = None, r = 20.0, k = 100, mu = 0.8, should_end = True, restart = None, searchable=False,**kargs):
        self.epsilon = w
        self.k = k
        self.mu = mu
        self.r = float(r)
        self.should_end = should_end
        self.restart = restart
        self.searchable = searchable

    def box(self, original, epsilon, model, target = None, untargeted = False, **kargs):
        if target is None:
            untargeted = True
            with torch.no_grad():
                target = model(original).max(1)[1]
        return self.attack(model, epsilon, original, untargeted, target, **kargs)

    def boxBetween(self, o1, o2, model, target = None, *args, **kargs):
        return self.attack(model, (o1 - o2).abs() / 2, (o1 + o2) / 2, target, **kargs)

    def attack(self, model, epsilon, xo, untargeted, target, loss_function=ai.stdLoss):
        if not self.epsilon is None:
            epsilon = self.epsilon
        x = nn.Parameter(xo.clone(), requires_grad=True)
        gradorg = h.zeros(x.shape)
        is_eq = 1
        for i in range(self.k):
            if self.restart is not None and i % int(self.k / self.restart) == 0:
                x = is_eq * (torch.randn_like(xo) * epsilon + xo) + (1 - is_eq) * x
                x = nn.Parameter(x, requires_grad = True)

            model.optimizer.zero_grad()

            out = model(x)
            loss = loss_function(out, target)

            loss.backward()
            with torch.no_grad():
                oth = x.grad / torch.norm(x.grad, p=1)
                gradorg *= self.mu 
                gradorg += oth
                grad = (self.r * epsilon / self.k) * ai.mysign(gradorg)
                if self.should_end:
                    is_eq = ai.mulIfEq(grad, out, target)
                x = (x + grad * is_eq) if untargeted else (x - grad * is_eq)
                x = xo + torch.clamp(x - xo, -epsilon, epsilon)
                x.requires_grad_()

        model.optimizer.zero_grad()
        return x

    def boxBetween(self, o1, o2, model, target, *args, **kargs):
        raise "Not boxBetween is not yet supported by MI_FGSM"

    def __str__(self):
        return "MI_FGSM(%s)" % (("" if self.epsilon is None else "w="+str(self.epsilon)+",")
                                + ("" if self.k == 5 else "k="+str(self.k)+",")
                                + ("" if self.r == 5.0 else "r="+str(self.r)+",")
                                + ("" if self.mu == 0.8 else "r="+str(self.mu)+",")
                                + ("" if self.should_end else "should_end=False"))


class PGDK(MI_FGSM):
    def __init__(self, w = None, r = 5.0, k = 5, **kargs):
        super(PGDK,self).__init__(w=w, r=r, k = k, mu = 0, **kargs)

    def __str__(self):
        return "PGDK(%s)" % (("" if self.epsilon is None else "w="+str(self.epsilon)+",")
                              + ("" if self.k == 5 else "k="+str(self.k)+",")
                              + ("" if self.r == 5.0 else "r="+str(self.r)+",")
                              + ("" if self.should_end else "should_end=False"))

class PGD(PGDK):

    def __init__(self, k = 5, **kargs):
        super(PGD, self).__init__(r = 1, k=k, **kargs)

    def __str__(self):
        return "PGD(%s)" % (("" if self.epsilon is None else "w="+str(self.epsilon)+",")
                              + ("" if self.k == 5 else "k="+str(self.k)+",")
                              + ("" if self.should_end else "should_end=False"))

class NormalAdv(Point):
    def __init__(self, a="PGD()", w = None):
        self.a = (eval(a) if type(a) is str else a)
        self.epsilon = w

    def box(self, original, epsilon, *args, **kargs):
        if not self.epsilon is None:
            epsilon = self.epsilon
        epsilon = torch.randn(original.size()[0:1], device = h.device)[0] * epsilon
        return self.a.box(original, epsilon,*args, **kargs)

    def __str__(self):
        return "NormalAdv(%s)" % ( str(self.a) + ("" if self.epsilon is None else ",w="+str(self.epsilon)))


class AdvDom(Point):
    def __init__(self, a="PGD()", b="Box()", width = None):
        self.a = (eval(a) if type(a) is str else a)
        self.b = (eval(b) if type(b) is str else b)
        self.width = width

    def box(self, original, epsilon, *args, **kargs):
        adv = self.a.box(original, epsilon,*args, **kargs)
        return self.b.boxBetween(original, adv.ub(), *args, w=self.width, **kargs)

    def loss(self, *args, **kargs):
        return self.b.loss(*args, **kargs)

    def widthLoss(self, *args, **kargs):
        return self.b.widthLoss(*args, **kargs)

    def regLoss(self, *args, **kargs):
        return self.b.regLoss(*args, **kargs)

    def combinedLoss(self, *args, **kargs):
        return self.b.combinedLoss(*args, **kargs)

    def boxBetween(self, o1, o2, *args, **kargs):
        original = (o1 + o2) / 2
        adv = self.a.boxBetween(o1, o2, *args, **kargs)
        return self.b.boxBetween(original, adv.ub(), *args, **kargs)

    def __str__(self):
        return "AdvDom(%s)" % (("" if self.width is None else "width="+str(self.width)+",")
                               + str(self.a) + "," + str(self.b))

class BiAdv(AdvDom):
    def box(self, original, epsilon, **kargs):
        adv = self.a.box(original, epsilon,**kargs)
        extreme = (adv.ub() - original).abs()
        return self.b.boxBetween(original - extreme, original + extreme, **kargs)
    
    def boxBetween(self, o1, o2, *args, **kargs):
        original = (o1 + o2) / 2
        adv = self.a.boxBetween(o1, o2, *args, **kargs)
        extreme = (adv.ub() - original).abs()
        return self.b.boxBetween(original - extreme, original + extreme, *args, **kargs)

    def __str__(self):
        return "BiAdv" + AdvDom.__str__(self)[6:]


class HBox(object):
    Domain = ai.HybridZonotope

    def __init__(self, w = None, tot_weight = None, width_weight = None, pow_loss = None, log_loss = False, searchable = True, **kargs):
        self.w = w
        self.tot_weight = tot_weight
        self.width_weight = width_weight
        self.pow_loss = pow_loss
        self.searchable = searchable
        self.log_loss = log_loss

    def __str__(self):
        return "HBox(%s)" % ("" if self.w is None else "w="+str(self.w))

    def boxBetween(self, o1, o2, *args, **kargs):
        batches = o1.size()[0]
        num_elem = h.product(ei.size()[1:])
        ei = h.getEi(batches, num_elem)
        
        if len(o1.size()) > 2:
            ei = ei.contiguous().view(num_elem, *original.size())

        return self.Domain((o1 + o2) / 2, None, ei * (o1 - o2).abs() / 2).checkSizes()

    def box(self, original, radius, **kargs):
        """
        This version of it is slow, but keeps correlation down the line.
        """
        if not self.w is None:
            radius = self.w

        batches = original.size()[0]
        num_elem = h.product(original.size()[1:])
        ei = h.getEi(batches,num_elem)
        
        if len(original.size()) > 2:
            ei = ei.contiguous().view(num_elem, *original.size())

        return self.Domain(original, None, ei * radius).checkSizes()

    def line(self, o1, o2, w = None, **kargs):
        if not self.w is None:
            w = self.w

        ln = ((o2 - o1) / 2).unsqueeze(0)
        if not w is None and w > 0.0:
            batches = o1.size()[0]
            num_elem = h.product(o1.size()[1:])
            ei = h.getEi(batches,num_elem)
            if len(o1.size()) > 2:
                ei = ei.contiguous().view(num_elem, *o1.size())
            ln = torch.cat([ln, ei * w])
        return self.Domain((o1 + o2) / 2, None, ln ).checkSizes()

    def loss(self, dom, target, width_weight = 0, tot_weight = 1, **args):
        if not self.width_weight is None:
            width_weight = self.width_weight
        if not self.tot_weight is None:
            tot_weight = self.tot_weight
        
        r = -h.preDomRes(dom, target).lb()
        tot = F.softplus(r.max(1)[0])      # kinda works        

        if self.log_loss:
            tot = (tot + 1).log()
        if self.pow_loss is not None and self.pow_loss > 0 and self.pow_loss != 1:
            tot = tot.pow(self.pow_loss)

        ls = tot * tot_weight
        if width_weight > 0:
            ls += dom.diameter() * width_weight

        return ls / (width_weight + tot_weight)

    def widthLoss(self, dom, **kargs):
        return dom.diameter()

    def regLoss(self, dom, target, loss_fn, *args, **kargs):
        return torch.max(loss_fn(dom.lb(), target), 
                         loss_fn(dom.ub(), target))

    def combinedLoss(self, dom, target, loss_fn, *args, only_max=False, **kargs):
        mx = target.argmax(1)
        ups = dom.ub()
        r = range(ups.size()[0])
        if only_max:
            ups[r,mx] = ups.min(1)[0]
            ups_vals = ups.max(1)[0]
            diff = F.softplus(ups_vals - dom.lb()[r,mx])
            return diff * diff
        else:
            ups[r,mx] = dom.lb()[r,mx]
            return loss_fn(ups, target)


class Box(HBox):
    def __str__(self):
        return "Box(%s)" % ("" if self.w is None else "w="+str(self.w))

    def box(self, original, radius, **kargs):  
        """
        This version of it takes advantage of betas being uncorrelated.  
        Unfortunately they stay uncorrelated forever.  
        Counterintuitively, tests show more accuracy - this is because the other box
        creates lots of 0 errors which get accounted for by the calcultion of the newhead in relu 
        which is apparently worse than not accounting for errors.
        """
        if not self.w is None:
            radius = self.w
        return self.Domain(original, h.ones(original.size()) * radius, None).checkSizes()
    
    def line(self, o1, o2, w = 0, **kargs):
        if not self.w is None:
            w = self.w
        return self.Domain((o1 + o2) / 2, ((o2 - o1) / 2).abs() + h.ones(o2.size()) * w, None).checkSizes()

    def boxBetween(self, o1, o2, *args, **kargs):
        return self.line(o1, o2, **kargs)


class ZBox(HBox):

    def __str__(self):
        return "ZBox(%s)" % ("" if self.w is None else "w="+str(self.w))

    Domain = ai.Zonotope

class HSwitch(HBox):
    def __str__(self):
        return "HSwitch(%s)" % ("" if self.w is None else "w="+str(self.w))

    class Domain(ai.HybridZonotope):
        customRelu = ai.creluSwitch
    
class ZSwitch(ZBox):

    def __str__(self):
        return "ZSwitch(%s)" % ("" if self.w is None else "w="+str(self.w))
    class Domain(ai.Zonotope):
        customRelu = ai.creluSwitch


class ZNIPS(ZBox):

    def __str__(self):
        return "ZSwitch(%s)" % ("" if self.w is None else "w="+str(self.w))
    class Domain(ai.Zonotope):
        customRelu = ai.creluNIPS
    
class HSmooth(HBox):
    def __str__(self):
        return "HSmooth(%s)" % ("" if self.w is None else "w="+str(self.w))

    class Domain(ai.HybridZonotope):
        customRelu = ai.creluSmooth

    
class HNIPS(HBox):
    def __str__(self):
        return "HSmooth(%s)" % ("" if self.w is None else "w="+str(self.w))

    class Domain(ai.HybridZonotope):
        customRelu = ai.creluNIPS


class ZSmooth(ZBox):
    def __str__(self):
        return "ZSmooth(%s)" % ("" if self.w is None else "w="+str(self.w))

    class Domain(ai.Zonotope):
        customRelu = ai.creluSmooth
