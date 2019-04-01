import future
import builtins
import past
import six

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import components as comp
from torch.distributions import multinomial, categorical

import math
import numpy as np

try:
    from . import helpers as h
    from . import ai
    from . import scheduling as S
except:
    import helpers as h
    import ai
    import scheduling as S



class WrapDom(object):
    def __init__(self, a):
        self.a = eval(a) if type(a) is str else a

    def box(self, *args, **kargs):
        return self.Domain(self.a.box(*args, **kargs))

    def boxBetween(self, *args, **kargs):
        return self.Domain(self.a.boxBetween(*args, **kargs))

    def line(self, *args, **kargs):
        return self.Domain(self.a.line(*args, **kargs))

class DList(object):
    Domain = ai.ListDomain
    class MLoss():
        def __init__(self, aw):
            self.aw = aw
        def loss(self, dom, *args, lr = 1, **kargs):
            if self.aw <= 0.0:
                return 0
            return self.aw * dom.loss(*args, lr = lr * self.aw, **kargs)

    def __init__(self, *al):
        if len(al) == 0:
            al = [("Point()", 1.0), ("Box()", 0.1)]

        self.al = [(eval(a) if type(a) is str else a, S.Const.initConst(aw)) for a,aw in al]

    def getDiv(self, **kargs):
        return 1.0 / sum(aw.getVal(**kargs) for _,aw in self.al)

    def box(self, *args, **kargs):
        m = self.getDiv(**kargs)
        return self.Domain(ai.TaggedDomain(a.box(*args, **kargs), DList.MLoss(aw.getVal(**kargs) * m)) for a,aw in self.al)

    def boxBetween(self, *args, **kargs):
        
        m = self.getDiv(**kargs)
        return self.Domain(ai.TaggedDomain(a.boxBetween(*args, **kargs), DList.MLoss(aw.getVal(**kargs) * m)) for a,aw in self.al)

    def line(self, *args, **kargs):
        m = self.getDiv(**kargs)
        return self.Domain(ai.TaggedDomain(a.line(*args, **kargs), DList.MLoss(aw.getVal(**kargs) * m)) for a,aw in self.al)
        
    def __str__(self):
        return "DList(%s)" % h.sumStr("("+str(a)+","+str(w)+")" for a,w in self.al)

class Mix(DList):
    def __init__(self, a="Point()", b="Box()", aw = 1.0, bw = 0.1):
        super(Mix, self).__init__((a,aw), (b,bw))

class LinMix(DList):
    def __init__(self, a="Point()", b="Box()", bw = 0.1):
        super(LinMix, self).__init__((a,S.Complement(bw)), (b,bw))

class DProb(object):
    def __init__(self, *doms):
        if len(doms) == 0:
            doms = [("Point()", 0.8), ("Box()", 0.2)]
        div = 1.0 / sum(float(aw) for _,aw in doms)
        self.domains = [eval(a) if type(a) is str else a for a,_ in doms]
        self.probs = [ div * float(aw)  for _,aw in doms]

    def chooseDom(self):
        return self.domains[np.random.choice(len(self.domains), p = self.probs)] if len(self.domains) > 1 else self.domains[0]

    def box(self, *args, **kargs):
        domain = self.chooseDom()
        return domain.box(*args, **kargs)

    def line(self, *args, **kargs):
        domain = self.chooseDom()
        return domain.line(*args, **kargs)

    def __str__(self):
        return "DProb(%s)" % h.sumStr("("+str(a)+","+str(w)+")" for a,w in zip(self.domains, self.probs))

class Coin(DProb):
    def __init__(self, a="Point()", b="Box()", ap = 0.8, bp = 0.2):
        super(Coin, self).__init__((a,ap), (b,bp))

class Point(object):
    Domain = h.dten
    def __init__(self, **kargs):
        pass

    def box(self, original, *args, **kargs):
        return original

    def line(self, original, other, *args, **kargs): 
        return (original + other) / 2

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


class NormalPoint(Point):
    def __init__(self, w = None, **kargs):
        self.epsilon = w
        
    def box(self, original, w, *args, **kargs):
        """ original = mu = mean, epsilon = variance"""
        if not self.epsilon is None:
            w = self.epsilon

        inter = torch.randn_like(original, device = h.device) * w
        return original + inter

    def __str__(self):
        return "NormalPoint(%s)" % ("" if self.epsilon is None else str(self.epsilon))



class MI_FGSM(Point):

    def __init__(self, w = None, r = 20.0, k = 100, mu = 0.8, should_end = True, restart = None, searchable=False,**kargs):
        self.epsilon = S.Const.initConst(w)
        self.k = k
        self.mu = mu
        self.r = float(r)
        self.should_end = should_end
        self.restart = restart
        self.searchable = searchable

    def box(self, original, model, target = None, untargeted = False, **kargs):
        if target is None:
            untargeted = True
            with torch.no_grad():
                target = model(original).max(1)[1]
        return self.attack(model, original, untargeted, target, **kargs)

    def boxBetween(self, o1, o2, model, target = None, *args, **kargs):
        return self.attack(model, (o1 - o2).abs() / 2, (o1 + o2) / 2, target, **kargs)


    def attack(self, model, xo, untargeted, target, w, loss_function=ai.stdLoss, **kargs):
        w = self.epsilon.getVal(c = w, **kargs)

        x = nn.Parameter(xo.clone(), requires_grad=True)
        gradorg = h.zeros(x.shape)
        is_eq = 1

        w = h.ones(x.shape) * w
        for i in range(self.k):
            if self.restart is not None and i % int(self.k / self.restart) == 0:
                x = is_eq * (torch.rand_like(xo) * w + xo) + (1 - is_eq) * x
                x = nn.Parameter(x, requires_grad = True)

            model.optimizer.zero_grad()

            out = model(x).vanillaTensorPart()
            loss = loss_function(out, target)

            loss.sum().backward(retain_graph=True)
            with torch.no_grad():
                oth = x.grad / torch.norm(x.grad, p=1)
                gradorg *= self.mu 
                gradorg += oth
                grad = (self.r * w / self.k) * ai.mysign(gradorg)
                if self.should_end:
                    is_eq = ai.mulIfEq(grad, out, target)
                x = (x + grad * is_eq) if untargeted else (x - grad * is_eq)

                x = xo + torch.min(torch.max(x - xo, -w),w)
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


class PGD(MI_FGSM):
    def __init__(self, r = 5.0, k = 5, **kargs):
        super(PGD,self).__init__(r=r, k = k, mu = 0, **kargs)

    def __str__(self):
        return "PGD(%s)" % (("" if self.epsilon is None else "w="+str(self.epsilon)+",")
                            + ("" if self.k == 5 else "k="+str(self.k)+",")
                            + ("" if self.r == 5.0 else "r="+str(self.r)+",")
                            + ("" if self.should_end else "should_end=False"))

class IFGSM(PGD):

    def __init__(self, k = 5, **kargs):
        super(IFGSM, self).__init__(r = 1, k=k, **kargs)

    def __str__(self):
        return "IFGSM(%s)" % (("" if self.epsilon is None else "w="+str(self.epsilon)+",")
                              + ("" if self.k == 5 else "k="+str(self.k)+",")
                              + ("" if self.should_end else "should_end=False"))

class NormalAdv(Point): 
    def __init__(self, a="IFGSM()", w = None):
        self.a = (eval(a) if type(a) is str else a)
        self.epsilon = S.Const.initConst(w)

    def box(self, original, w, *args, **kargs):
        epsilon = self.epsilon.getVal(c = w, shape = original.shape[:1], **kargs)
        assert (0 <= h.dten(epsilon)).all()
        epsilon = torch.randn(original.size()[0:1], device = h.device)[0] * epsilon
        return self.a.box(original, w = epsilon, *args, **kargs)

    def __str__(self):
        return "NormalAdv(%s)" % ( str(self.a) + ("" if self.epsilon is None else ",w="+str(self.epsilon)))


class InclusionSample(Point):
    def __init__(self, sub, a="Box()", normal = False, w = None, **kargs):
        self.sub = S.Const.initConst(sub)  # sub is the fraction of w to use.
        self.w = S.Const.initConst(w)
        self.normal = normal
        self.a = (eval(a) if type(a) is str else a)

    def box(self, original, w, *args, **kargs):
        w = self.w.getVal(c = w, shape = original.shape[:1], **kargs)
        sub = self.sub.getVal(c = 1, shape = original.shape[:1], **kargs)

        assert (0 <= h.dten(w)).all()
        assert (h.dten(sub) <= 1).all()
        assert (0 <= h.dten(sub)).all() 
        if self.normal:
            inter = torch.randn_like(original, device = h.device)
        else:
            inter = (torch.rand_like(original, device = h.device) * 2 - 1) 

        inter = inter * w * (1 - sub)
        
        return self.a.box(original + inter, w = w * sub, *args, **kargs)

    def boxBetween(self, o1, o2, *args, **kargs):
        w = (o2 - o1).abs()
        return self.box( (o2 + o1)/2 , w = w, *args, **kargs)

    def __str__(self):
        return "InclusionSample(%s, %s)" % (str(self.sub), str(self.a) + ("" if self.epsilon is None else ",w="+str(self.epsilon)))

InSamp = InclusionSample


class AdvInclusion(InclusionSample):
    def __init__(self, sub, a="IFGSM()", b="Box()", w = None, **kargs):
        self.sub = S.Const.initConst(sub)  # sub is the fraction of w to use.
        self.w = S.Const.initConst(w)
        self.a = (eval(a) if type(a) is str else a)
        self.b = (eval(b) if type(b) is str else b)

    def box(self, original, w, *args, **kargs):
        w = self.w.getVal(c = w, shape = original.shape, **kargs)
        sub = self.sub.getVal(c = 1, shape = original.shape, **kargs)

        assert (0 <= h.dten(w)).all()
        assert (h.dten(sub) <= 1).all()
        assert (0 <= h.dten(sub)).all() 

        if h.dten(w).sum().item() <= 0.0:
            inter = original
        else:
            inter = self.a.box(original, w = w * (1 - sub), *args, **kargs)

        return self.b.box(inter, w = w * sub, *args, **kargs)

    def __str__(self):
        return "AdvInclusion(%s, %s, %s)" % (str(self.sub), str(self.a), str(self.b) + ("" if self.epsilon is None else ",w="+str(self.epsilon)))


class AdvDom(Point):
    def __init__(self, a="IFGSM()", b="Box()"):
        self.a = (eval(a) if type(a) is str else a)
        self.b = (eval(b) if type(b) is str else b)

    def box(self, original,*args, **kargs):
        adv = self.a.box(original, *args, **kargs)
        return self.b.boxBetween(original, adv.ub(), *args, **kargs)

    def boxBetween(self, o1, o2, *args, **kargs):
        original = (o1 + o2) / 2
        adv = self.a.boxBetween(o1, o2, *args, **kargs)
        return self.b.boxBetween(original, adv.ub(), *args, **kargs)

    def __str__(self):
        return "AdvDom(%s)" % (("" if self.width is None else "width="+str(self.width)+",")
                               + str(self.a) + "," + str(self.b))



class BiAdv(AdvDom):
    def box(self, original, **kargs):
        adv = self.a.box(original, **kargs)
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

    def domain(self, *args, **kargs):
        return ai.TaggedDomain(self.Domain(*args, **kargs), self)

    def __init__(self, w = None, tot_weight = 1, width_weight = 0, pow_loss = None, log_loss = False, searchable = True, cross_loss = True, **kargs):
        self.w = S.Const.initConst(w)
        self.tot_weight = S.Const.initConst(tot_weight)
        self.width_weight = S.Const.initConst(width_weight)
        self.pow_loss = pow_loss
        self.searchable = searchable
        self.log_loss = log_loss
        self.cross_loss = cross_loss

    def __str__(self):
        return "HBox(%s)" % ("" if self.w is None else "w="+str(self.w))

    def boxBetween(self, o1, o2,  *args, **kargs):
        batches = o1.size()[0]
        num_elem = h.product(o1.size()[1:])
        ei = h.getEi(batches, num_elem)
        
        if len(o1.size()) > 2:
            ei = ei.contiguous().view(num_elem, *o1.size())

        return self.domain((o1 + o2) / 2, None, ei * (o2 - o1).abs() / 2).checkSizes()

    def box(self, original, w, **kargs):
        """
        This version of it is slow, but keeps correlation down the line.
        """
        radius = self.w.getVal(c = w, **kargs)

        batches = original.size()[0]
        num_elem = h.product(original.size()[1:])
        ei = h.getEi(batches,num_elem)
        
        if len(original.size()) > 2:
            ei = ei.contiguous().view(num_elem, *original.size())

        return self.domain(original, None, ei * radius).checkSizes()

    def line(self, o1, o2, **kargs):
        w = self.w.getVal(c = 0, **kargs)

        ln = ((o2 - o1) / 2).unsqueeze(0)
        if not w is None and w > 0.0:
            batches = o1.size()[0]
            num_elem = h.product(o1.size()[1:])
            ei = h.getEi(batches,num_elem)
            if len(o1.size()) > 2:
                ei = ei.contiguous().view(num_elem, *o1.size())
            ln = torch.cat([ln, ei * w])
        return self.domain((o1 + o2) / 2, None, ln ).checkSizes()

    def loss(self, dom, target, *args, **kargs):
        width_weight = self.width_weight.getVal(**kargs)
        tot_weight = self.tot_weight.getVal(**kargs)
        
        if self.cross_loss:
            r = dom.ub()
            inds = torch.arange(r.shape[0], device=h.device, dtype=h.ltype)
            r[inds,target] = dom.lb()[inds,target]
            tot = r.loss(target, *args, **kargs)
        else:
            tot = dom.loss(target, *args, **kargs)

        if self.log_loss:
            tot = (tot + 1).log()
        if self.pow_loss is not None and self.pow_loss > 0 and self.pow_loss != 1:
            tot = tot.pow(self.pow_loss)

        ls = tot * tot_weight
        if width_weight > 0:
            ls += dom.diameter() * width_weight

        return ls / (width_weight + tot_weight)

class Box(HBox):
    def __str__(self):
        return "Box(%s)" % ("" if self.w is None else "w="+str(self.w))

    def box(self, original, w, **kargs):  
        """
        This version of it takes advantage of betas being uncorrelated.  
        Unfortunately they stay uncorrelated forever.  
        Counterintuitively, tests show more accuracy - this is because the other box
        creates lots of 0 errors which get accounted for by the calcultion of the newhead in relu 
        which is apparently worse than not accounting for errors.
        """
        radius = self.w.getVal(c = w, **kargs)
        return self.domain(original, h.ones(original.size()) * radius, None).checkSizes()
    
    def line(self, o1, o2, **kargs):
        w = self.w.getVal(c = 0, **kargs)
        return self.domain((o1 + o2) / 2, ((o2 - o1) / 2).abs() + h.ones(o2.size()) * w, None).checkSizes()

    def boxBetween(self, o1, o2, *args, **kargs):
        return self.line(o1, o2, **kargs)

class ZBox(HBox):

    def __str__(self):
        return "ZBox(%s)" % ("" if self.w is None else "w="+str(self.w))

    def Domain(self, *args, **kargs):
        return ai.Zonotope(*args, **kargs)

class HSwitch(HBox):
    def __str__(self):
        return "HSwitch(%s)" % ("" if self.w is None else "w="+str(self.w))
    
    def Domain(self, *args, **kargs):
        return ai.HybridZonotope(*args, customRelu = ai.creluSwitch, **kargs)
    
class ZSwitch(ZBox):

    def __str__(self):
        return "ZSwitch(%s)" % ("" if self.w is None else "w="+str(self.w))
    def Domain(self, *args, **kargs):
        return ai.Zonotope(*args, customRelu = ai.creluSwitch, **kargs)


class ZNIPS(ZBox):

    def __str__(self):
        return "ZSwitch(%s)" % ("" if self.w is None else "w="+str(self.w))

    def Domain(self, *args, **kargs):
        return ai.Zonotope(*args, customRelu = ai.creluNIPS, **kargs)
    
class HSmooth(HBox):
    def __str__(self):
        return "HSmooth(%s)" % ("" if self.w is None else "w="+str(self.w))

    def Domain(self, *args, **kargs):
        return ai.HybridZonotope(*args, customRelu = ai.creluSmooth, **kargs)
    
class HNIPS(HBox):
    def __str__(self):
        return "HSmooth(%s)" % ("" if self.w is None else "w="+str(self.w))

    def Domain(self, *args, **kargs):
        return ai.HybridZonotope(*args, customRelu = ai.creluNIPS, **kargs)

class ZSmooth(ZBox):
    def __str__(self):
        return "ZSmooth(%s)" % ("" if self.w is None else "w="+str(self.w))

    def Domain(self, *args, **kargs):
        return ai.Zonotope(*args, customRelu = ai.creluSmooth, **kargs)





# stochastic correlation
class HRand(WrapDom):
    # domain must be an ai style domain like hybrid zonotope.
    def __init__(self, num_correlated, a = "HSwitch()", **kargs):
        super(HRand, self).__init__(Box())
        self.num_correlated = num_correlated
        self.dom = eval(a) if type(a) is str else a
        
    def Domain(self, d):
        with torch.no_grad():
            out = d.abstractApplyLeaf('stochasticCorrelate', self.num_correlated)
            out = self.dom.Domain(out.head, out.beta, out.errors)
        return out

    def __str__(self):
        return "HRand(%s, domain = %s)" % (str(self.num_correlated), str(self.a))
