import future
import builtins
import past
import six

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd

from functools import reduce

try:
    from . import helpers as h
except:
    import helpers as h



def catNonNullErrors(op, ref_errs=None): # the way of things is ugly
    def doop(er1, er2):
        erS, erL = (er1, er2)
        sS, sL = (erS.size()[0], erL.size()[0])

        if sS == sL: # TODO: here we know we used transformers on either side which didnt introduce new error terms (this is a hack for hybrid zonotopes and doesn't work with adaptive error term adding).
            return op(erS,erL)

        if ref_errs is not None:
            sz = ref_errs.size()[0]
        else:
            sz = min(sS, sL)
    
        p1 = op(erS[:sz], erL[:sz])
        erSrem = erS[sz:]
        erLrem = erS[sz:]
        p2 = op(erSrem, h.zeros(erSrem.shape))
        p3 = op(h.zeros(erLrem.shape), erLrem)
        return torch.cat((p1,p2,p3), dim=0)
    return doop

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
    gtz = dom.head.gt(0).to_dtype()
    mx /= 2
    newhead = h.ifThenElse(should_box, mx, gtz * dom.head)
    newbeta = h.ifThenElse(should_box, mx, gtz * (dom.beta if not dom.beta is None else 0))
    newerr = (1 - should_box.to_dtype()) * gtz * dom.errors

    return dom.new(newhead, newbeta , newerr)


def creluBoxySound(dom):
    if dom.errors is None:
        if dom.beta is None:
            return dom.new(F.relu(dom.head), None, None)
        er = dom.beta 
        mx = F.relu(dom.head + er)
        mn = F.relu(dom.head - er)
        return dom.new((mn + mx) / 2, (mx - mn) / 2 + 2e-6 , None)

    aber = torch.abs(dom.errors)

    sm = torch.sum(aber, 0) 

    if not dom.beta is None:
        sm += dom.beta

    mx = dom.head + sm
    mn = dom.head - sm

    should_box = mn.lt(0) * mx.gt(0)
    gtz = dom.head.gt(0).to_dtype()
    mx /= 2
    newhead = h.ifThenElse(should_box, mx, gtz * dom.head)
    newbeta = h.ifThenElse(should_box, mx + 2e-6, gtz * (dom.beta if not dom.beta is None else 0))
    newerr = (1 - should_box.to_dtype()) * gtz * dom.errors

    return dom.new(newhead, newbeta, newerr)


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
    newhead = h.ifThenElse(should_box, h.ifThenElse(should_boxer, mx / 2, dom.head + mn ), gtz.to_dtype() * dom.head)
    zbet =  dom.beta if not dom.beta is None else 0
    newbeta = h.ifThenElse(should_box, h.ifThenElse(should_boxer, mx / 2, mn + zbet), gtz.to_dtype() * zbet)
    newerr  = h.ifThenElseL(should_box, 1 - should_boxer, gtz).to_dtype() * dom.errors

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
    t = nmn / (mmx + nmn + eps) # mn.lt(0).to_dtype() * F.sigmoid(nmn - nmx)

    shouldnt_zero = mx.gt(0).to_dtype()

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
    
    sm = torch.sum(torch.abs(dom.errors), 0) 

    if not dom.beta is None:
        sm += dom.beta

    mn = dom.head - sm
    mx =  dom.head + sm

    mngz = mn >= 0.0

    zs = h.zeros(dom.head.shape)

    diff = mx - mn

    lam = torch.where((mx > 0) & (diff > 0.0), mx / diff, zs)
    mu = lam * mn * (-0.5)

    betaz = zs if dom.beta is None else dom.beta 

    newhead = torch.where(mngz, dom.head , lam * dom.head + mu)
    mngz += diff <= 0.0
    newbeta = torch.where(mngz, betaz    , lam * betaz + mu ) # mu is always positive on this side
    newerr = torch.where(mngz, dom.errors, lam * dom.errors )
    return dom.new(newhead, newbeta, newerr)




class MaxTypes:

    @staticmethod
    def ub(x):
        return x.ub()

    @staticmethod
    def only_beta(x):
        return x.beta if x.beta is not None else x.head * 0

    @staticmethod
    def head_beta(x):
        return MaxTypes.only_beta(x) + x.head

class HybridZonotope:

    def isSafe(self, target):
        od,_ = torch.min(h.preDomRes(self,target).lb(), 1)
        return od.gt(0.0).long()

    def isPoint(self):
        return False

    def labels(self):
        target = torch.max(self.ub(), 1)[1]
        l = list(h.preDomRes(self,target).lb()[0])
        return [target.item()] + [ i for i,v in zip(range(len(l)), l) if v <= 0]

    def relu(self):
        return self.customRelu(self)
    
    def __init__(self, head, beta, errors, customRelu = creluBoxy, **kargs):
        self.head = head
        self.errors = errors
        self.beta = beta
        self.customRelu = creluBoxy if customRelu is None else customRelu

    def new(self, *args, customRelu = None, **kargs):
        return self.__class__(*args, **kargs, customRelu = self.customRelu if customRelu is None else customRelu).checkSizes()

    def zono_to_hybrid(self, *args, **kargs): # we are already a hybrid zono.
        return self.new(self.head, self.beta, self.errors, **kargs)

    def hybrid_to_zono(self, *args, correlate=True, customRelu = None, **kargs):
        beta = self.beta
        errors = self.errors
        if correlate and beta is not None:
            batches = beta.shape[0]
            num_elem = h.product(beta.shape[1:])
            ei = h.getEi(batches, num_elem)
        
            if len(beta.shape) > 2:
                ei = ei.contiguous().view(num_elem, *beta.shape)
            err = ei * beta
            errors = torch.cat((err, errors), dim=0) if errors is not None else err
            beta = None

        return Zonotope(self.head, beta, errors if errors is not None else (self.beta * 0).unsqueeze(0) , customRelu = self.customRelu if customRelu is None else None)



    def abstractApplyLeaf(self, foo, *args, **kargs):
        return getattr(self, foo)(*args, **kargs)

    def decorrelate(self, cc_indx_batch_err): # keep these errors
        if self.errors is None:
            return self

        batch_size = self.head.shape[0]
        num_error_terms = self.errors.shape[0]

        

        beta = h.zeros(self.head.shape).to_dtype() if self.beta is None  else self.beta
        errors = h.zeros([0] + list(self.head.shape)).to_dtype() if self.errors is None else self.errors

        inds_i = torch.arange(self.head.shape[0], device=h.device).unsqueeze(1).long()
        errors = errors.to_dtype().permute(1,0, *list(range(len(self.errors.shape)))[2:])
        
        sm = errors.clone()
        sm[inds_i, cc_indx_batch_err] = 0
        
        beta = beta.to_dtype() + sm.abs().sum(dim=1)

        errors = errors[inds_i, cc_indx_batch_err]
        errors = errors.permute(1,0, *list(range(len(self.errors.shape)))[2:]).contiguous()
        return self.new(self.head, beta, errors)
    
    def dummyDecorrelate(self, num_decorrelate):
        if num_decorrelate == 0 or self.errors is None:
            return self
        elif num_decorrelate >= self.errors.shape[0]:
            beta = self.beta
            if self.errors is not None:
                errs = self.errors.abs().sum(dim=0)
                if beta is None:
                    beta = errs
                else:
                    beta += errs
            return self.new(self.head, beta, None)
        return None

    def stochasticDecorrelate(self, num_decorrelate, choices = None, num_to_keep=False):
        dummy = self.dummyDecorrelate(num_decorrelate)
        if dummy is not None:
            return dummy
        num_error_terms = self.errors.shape[0]
        batch_size = self.head.shape[0]

        ucc_mask = h.ones([batch_size, self.errors.shape[0]]).long()
        cc_indx_batch_err = h.cudify(torch.multinomial(ucc_mask.to_dtype(), num_decorrelate if num_to_keep else num_error_terms - num_decorrelate, replacement=False)) if choices is None else choices
        return self.decorrelate(cc_indx_batch_err)

    def decorrelateMin(self, num_decorrelate, num_to_keep=False):
        dummy = self.dummyDecorrelate(num_decorrelate)
        if dummy is not None:
            return dummy

        num_error_terms = self.errors.shape[0]
        batch_size = self.head.shape[0]

        error_sum_b_e = self.errors.abs().view(self.errors.shape[0], batch_size, -1).sum(dim=2).permute(1,0)
        cc_indx_batch_err = error_sum_b_e.topk(num_decorrelate if num_to_keep else num_error_terms - num_decorrelate)[1]
        return self.decorrelate(cc_indx_batch_err)
      
    def correlate(self, cc_indx_batch_beta): # given in terms of the flattened matrix.
        num_correlate = h.product(cc_indx_batch_beta.shape[1:])
        
        beta = h.zeros(self.head.shape).to_dtype() if self.beta is None  else self.beta
        errors = h.zeros([0] + list(self.head.shape)).to_dtype() if self.errors is None else self.errors

        batch_size = beta.shape[0]
        new_errors = h.zeros([num_correlate] + list(self.head.shape)).to_dtype()
        
        inds_i = torch.arange(batch_size, device=h.device).unsqueeze(1).long()

        nc = torch.arange(num_correlate, device=h.device).unsqueeze(1).long()

        new_errors = new_errors.permute(1,0, *list(range(len(new_errors.shape)))[2:]).contiguous().view(batch_size, num_correlate, -1)
        new_errors[inds_i, nc.unsqueeze(0).expand([batch_size]+list(nc.shape)).squeeze(2), cc_indx_batch_beta] = beta.view(batch_size,-1)[inds_i, cc_indx_batch_beta]

        new_errors = new_errors.permute(1,0, *list(range(len(new_errors.shape)))[2:]).contiguous().view(num_correlate, batch_size, *beta.shape[1:])
        errors = torch.cat((errors, new_errors), dim=0)
            
        beta.view(batch_size, -1)[inds_i, cc_indx_batch_beta] = 0
        
        return self.new(self.head, beta, errors)

    def stochasticCorrelate(self, num_correlate, choices = None):
        if num_correlate == 0:
            return self

        domshape = self.head.shape
        batch_size = domshape[0]
        num_pixs = h.product(domshape[1:])
        num_correlate = min(num_correlate, num_pixs)
        ucc_mask = h.ones([batch_size, num_pixs ]).long()

        cc_indx_batch_beta = h.cudify(torch.multinomial(ucc_mask.to_dtype(), num_correlate, replacement=False)) if choices is None else choices
        return self.correlate(cc_indx_batch_beta)


    def correlateMaxK(self, num_correlate):
        if num_correlate == 0:
            return self
        
        domshape = self.head.shape
        batch_size = domshape[0]
        num_pixs = h.product(domshape[1:])
        num_correlate = min(num_correlate, num_pixs)

        concrete_max_image = self.ub().view(batch_size, -1)

        cc_indx_batch_beta = concrete_max_image.topk(num_correlate)[1]
        return self.correlate(cc_indx_batch_beta)

    def correlateMaxPool(self, *args, max_type = MaxTypes.ub , max_pool = F.max_pool2d, **kargs):
        domshape = self.head.shape
        batch_size = domshape[0]
        num_pixs = h.product(domshape[1:])

        concrete_max_image = max_type(self)

        cc_indx_batch_beta = max_pool(concrete_max_image, *args, return_indices=True, **kargs)[1].view(batch_size, -1)

        return self.correlate(cc_indx_batch_beta)

    def checkSizes(self):
        if not self.errors is None:
            if not self.errors.size()[1:] == self.head.size():
                raise Exception("Such bad sizes on error:", self.errors.shape, " head:", self.head.shape)
            if torch.isnan(self.errors).any():
                raise Exception("Such nan in errors")
        if not self.beta is None:
            if not self.beta.size() == self.head.size():
                raise Exception("Such bad sizes on beta")

            if torch.isnan(self.beta).any():
                raise Exception("Such nan in errors")
            if self.beta.lt(0.0).any():
                self.beta = self.beta.abs()
            
        return self

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

    def addPar(self, a, b):
        return self.new(a.head + b.head, h.msum(a.beta, b.beta, lambda a,b: a + b), h.msum(a.errors, b.errors, catNonNullErrors(lambda a,b: a + b, self.errors)))

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


    def getBeta(self):
        return self.head * 0 if self.beta is None else self.beta

    def getErrors(self):
        return (self.head * 0).unsqueeze(0) if self.beta is None else self.errors

    def merge(self, other, ref = None): # the vast majority of the time ref should be none here.  Not for parallel computation with powerset
        s_beta = self.getBeta() # so that beta is never none

        sbox_u = self.head + s_beta
        sbox_l = self.head - s_beta
        o_u = other.ub()
        o_l = other.lb()
        o_in_s = (o_u <= sbox_u) & (o_l >= sbox_l)

        s_err_mx = self.errors.abs().sum(dim=0)

        if not isinstance(other, HybridZonotope):
            new_head = (self.head + other.center()) / 2
            new_beta = torch.max(sbox_u + s_err_mx,o_u) - new_head
            return self.new(torch.where(o_in_s, self.head, new_head), torch.where(o_in_s, self.beta,new_beta), o_in_s.float() * self.errors)
        
        # TODO: could be more efficient if one of these doesn't have beta or errors but thats okay for now.
        s_u = sbox_u + s_err_mx
        s_l = sbox_l - s_err_mx

        obox_u = o_u - other.head
        obox_l = o_l + other.head

        s_in_o = (s_u <= obox_u) & (s_l >= obox_l)
        
        # TODO: could theoretically still do something better when one is contained partially in the other
        new_head = (self.head + other.center()) / 2
        new_beta = torch.max(sbox_u + self.getErrors().abs().sum(dim=0),o_u) - new_head

        return self.new(torch.where(o_in_s, self.head, torch.where(s_in_o, other.head, new_head))
                        , torch.where(o_in_s, s_beta,torch.where(s_in_o, other.getBeta(), new_beta))
                        , h.msum(o_in_s.float() * self.errors, s_in_o.float() * other.errors, catNonNullErrors(lambda a,b: a + b, ref_errs = ref.errors if ref is not None else ref))) # these are both zero otherwise
    

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

    def double(self):
        return self.new(self.head.double(), self.beta.double()  if self.beta is not None else None, self.errors.double() if self.errors is not None else None) 

    def float(self):
        return self.new(self.head.float(), self.beta.float()  if self.beta is not None else None, self.errors.float() if self.errors is not None else None) 

    def to_dtype(self):
        return self.new(self.head.to_dtype(), self.beta.to_dtype()  if self.beta is not None else None, self.errors.to_dtype() if self.errors is not None else None) 
    
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


    def applyMonotone(self, foo, *args, **kargs):
        if self.beta is None and self.errors is None:
            return self.new(foo(self.head), None , None)

        beta = self.concreteErrors().abs().sum(dim=0)

        tp = foo(self.head + beta, *args, **kargs)
        bt = foo(self.head - beta, *args, **kargs)

        new_hybrid = self.new((tp + bt) / 2, (tp - bt) / 2 , None)


        if self.errors is not None:
            return new_hybrid.correlateMaxK(self.errors.shape[0])
        return new_hybrid

    def avg_pool2d(self, *args, **kargs):
        nhead = F.avg_pool2d(self.head, *args, **kargs)
        return self.new(nhead, 
                        None if self.beta is None else F.avg_pool2d(self.beta, *args, **kargs), 
                        None if self.errors is None else F.avg_pool2d(self.errors.view(-1, *self.head.shape[1:]), *args, **kargs).view(-1,*nhead.shape)) 

    def adaptive_avg_pool2d(self, *args, **kargs):
        nhead = F.adaptive_avg_pool2d(self.head, *args, **kargs)
        return self.new(nhead, 
                        None if self.beta is None else F.adaptive_avg_pool2d(self.beta, *args, **kargs), 
                        None if self.errors is None else F.adaptive_avg_pool2d(self.errors.view(-1, *self.head.shape[1:]), *args, **kargs).view(-1,*nhead.shape)) 

    def elu(self):
        return self.applyMonotone(F.elu)

    def selu(self):
        return self.applyMonotone(F.selu)

    def sigm(self):
        return self.applyMonotone(F.sigmoid)

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
        return self.head - self.concreteErrors().abs().sum(dim=0)

    def ub(self):
        return self.head + self.concreteErrors().abs().sum(dim=0)

    def size(self):
        return self.head.size()

    def diameter(self):
        abal = torch.abs(self.concreteErrors()).transpose(0,1)
        return abal.sum(1).sum(1) # perimeter

    def loss(self, target, **args):
        r = -h.preDomRes(self, target).lb()
        return F.softplus(r.max(1)[0])

    def deep_loss(self, act = F.relu, *args, **kargs):
        batch_size = self.head.shape[0]
        inds = torch.arange(batch_size, device=h.device).unsqueeze(1).long()

        def dl(l,u):
            ls, lsi = torch.sort(l, dim=1)
            ls_u = u[inds, lsi]

            def slidingMax(a): # using maxpool
                k = a.shape[1]
                ml = a.min(dim=1)[0].unsqueeze(1)

                inp = torch.cat((h.zeros([batch_size, k]), a - ml), dim=1)
                mpl = F.max_pool1d(inp.unsqueeze(1) ,  kernel_size = k, stride=1, padding = 0, return_indices=False).squeeze(1)
                return mpl[:,:-1] + ml
            
            return act(slidingMax(ls_u) - ls).sum(dim=1)

        l = self.lb().view(batch_size, -1)
        u = self.ub().view(batch_size, -1)
        return ( dl(l,u) + dl(-u,-l) ) / (2 * l.shape[1]) # make it easier to regularize against



class Zonotope(HybridZonotope):
    def applySuper(self, ret):
        batches = ret.head.size()[0]
        num_elem = h.product(ret.head.size()[1:])
        ei = h.getEi(batches, num_elem)

        if len(ret.head.size()) > 2:
            ei = ei.contiguous().view(num_elem, *ret.head.size())

        ret.errors = torch.cat( (ret.errors, ei * ret.beta) ) if not ret.beta is None else ret.errors
        ret.beta = None
        return ret.checkSizes()

    def zono_to_hybrid(self, *args, customRelu = None, **kargs): # we are already a hybrid zono.
        return HybridZonotope(self.head, self.beta, self.errors, customRelu = self.customRelu if customRelu is None else customRelu)

    def hybrid_to_zono(self, *args, **kargs):
        return self.new(self.head, self.beta, self.errors, **kargs)

    def applyMonotone(self, *args, **kargs):
        return self.applySuper(super(Zonotope,self).applyMonotone(*args, **kargs))

    def softplus(self):
        return self.applySuper(super(Zonotope,self).softplus())

    def relu(self):
        return self.applySuper(super(Zonotope,self).relu())

    def splitRelu(self, *args, **kargs):
        return [self.applySuper(a) for a in super(Zonotope, self).splitRelu(*args, **kargs)]


def mysign(x):
    e = x.eq(0).to_dtype()
    r = x.sign().to_dtype()
    return r + e

def mulIfEq(grad,out,target):
    pred = out.max(1, keepdim=True)[1]
    is_eq = pred.eq(target.view_as(pred)).to_dtype()
    is_eq = is_eq.view([-1] + [1 for _ in grad.size()[1:]]).expand_as(grad)
    return is_eq
    

def stdLoss(out, target):
    if torch.__version__[0] == "0":
        return F.cross_entropy(out, target, reduce = False)
    else:
        return F.cross_entropy(out, target, reduction='none')



class ListDomain(object):

    def __init__(self, al, *args, **kargs):
        self.al = list(al)

    def new(self, *args, **kargs):
        return self.__class__(*args, **kargs)

    def isSafe(self,*args,**kargs):
        raise "Domain Not Suitable For Testing"

    def labels(self):
        raise "Domain Not Suitable For Testing"

    def isPoint(self):
        return all(a.isPoint() for a in self.al)

    def __mul__(self, flt):
        return self.new(a.__mul__(flt) for a in self.al)

    def __truediv__(self, flt):
        return self.new(a.__truediv__(flt) for a in self.al)

    def __add__(self, other):
        if isinstance(other, ListDomain):
            return self.new(a.__add__(o) for a,o in zip(self.al, other.al))
        else:
            return self.new(a.__add__(other) for a in self.al)

    def merge(self, other, ref = None):
        if ref is None:
            return self.new(a.merge(o) for a,o in zip(self.al,other.al) )
        return self.new(a.merge(o, ref = r) for a,o,r in zip(self.al,other.al, ref.al))

    def addPar(self, a, b):
        return self.new(s.addPar(av,bv) for s,av,bv in zip(self.al, a.al, b.al))

    def __sub__(self, other):
        if isinstance(other, ListDomain):
            return self.new(a.__sub__(o) for a,o in zip(self.al, other.al))
        else:
            return self.new(a.__sub__(other) for a in self.al)

    def abstractApplyLeaf(self, *args, **kargs):
        return self.new(a.abstractApplyLeaf(*args, **kargs) for a in self.al)

    def bmm(self, other):
        return self.new(a.bmm(other) for a in self.al)

    def matmul(self, other):
        return self.new(a.matmul(other) for a in self.al)

    def conv(self, *args, **kargs):
        return self.new(a.conv(*args, **kargs) for a in self.al)

    def conv1d(self, *args, **kargs):
        return self.new(a.conv1d(*args, **kargs) for a in self.al)

    def conv2d(self, *args, **kargs):
        return self.new(a.conv2d(*args, **kargs) for a in self.al)

    def conv3d(self, *args, **kargs):
        return self.new(a.conv3d(*args, **kargs) for a in self.al)

    def max_pool2d(self, *args, **kargs):
        return self.new(a.max_pool2d(*args, **kargs) for a in self.al)

    def avg_pool2d(self, *args, **kargs):
        return self.new(a.avg_pool2d(*args, **kargs) for a in self.al)

    def adaptive_avg_pool2d(self, *args, **kargs):
        return self.new(a.adaptive_avg_pool2d(*args, **kargs) for a in self.al)

    def unsqueeze(self, *args, **kargs):
        return self.new(a.unsqueeze(*args, **kargs) for a in self.al)

    def squeeze(self, *args, **kargs):
        return self.new(a.squeeze(*args, **kargs) for a in self.al)

    def view(self, *args, **kargs):
        return self.new(a.view(*args, **kargs) for a in self.al)

    def gather(self, *args, **kargs):
        return self.new(a.gather(*args, **kargs) for a in self.al)

    def sum(self, *args, **kargs):
        return self.new(a.sum(*args,**kargs) for a in self.al)

    def double(self):
        return self.new(a.double() for a in self.al)

    def float(self):
        return self.new(a.float() for a in self.al)

    def to_dtype(self):
        return self.new(a.to_dtype() for a in self.al)

    def vanillaTensorPart(self):
        return self.al[0].vanillaTensorPart()

    def center(self):
        return self.new(a.center() for a in self.al)

    def ub(self):
        return self.new(a.ub() for a in self.al)

    def lb(self):
        return self.new(a.lb() for a in self.al)

    def relu(self):
        return self.new(a.relu() for a in self.al)

    def splitRelu(self, *args, **kargs):
        return self.new(a.splitRelu(*args, **kargs) for a in self.al)

    def softplus(self):
        return self.new(a.softplus() for a in self.al)

    def elu(self):
        return self.new(a.elu() for a in self.al)

    def selu(self):
        return self.new(a.selu() for a in self.al)

    def sigm(self):
        return self.new(a.sigm() for a in self.al)

    def cat(self, other, *args, **kargs):
        return self.new(a.cat(o, *args, **kargs) for a,o in zip(self.al, other.al))


    def split(self, *args, **kargs):
        return [self.new(*z) for z in zip(a.split(*args, **kargs) for a in self.al)]

    def size(self):
        return self.al[0].size()

    def loss(self, *args, **kargs):
        return sum(a.loss(*args, **kargs) for a in self.al)

    def deep_loss(self, *args, **kargs):
        return sum(a.deep_loss(*args, **kargs) for a in self.al)

    def checkSizes(self):
        for a in self.al:
            a.checkSizes()
        return self


class TaggedDomain(object):


    def __init__(self, a, tag = None):
        self.tag = tag
        self.a = a

    def isSafe(self,*args,**kargs):
        return self.a.isSafe(*args, **kargs)

    def isPoint(self):
        return self.a.isPoint()

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

    def addPar(self, a,b):
        return TaggedDomain(self.a.addPar(a.a, b.a), self.tag)

    def __sub__(self, other):
        if isinstance(other, TaggedDomain):
            return TaggedDomain(self.a.__sub__(other.a), self.tag)
        else:
            return TaggedDomain(self.a.__sub__(other), self.tag)

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

    def max_pool2d(self, *args, **kargs):
        return TaggedDomain(self.a.max_pool2d(*args, **kargs), self.tag)

    def avg_pool2d(self, *args, **kargs):
        return TaggedDomain(self.a.avg_pool2d(*args, **kargs), self.tag)

    def adaptive_avg_pool2d(self, *args, **kargs):
        return TaggedDomain(self.a.adaptive_avg_pool2d(*args, **kargs), self.tag)


    def unsqueeze(self, *args, **kargs):
        return TaggedDomain(self.a.unsqueeze(*args, **kargs), self.tag)

    def squeeze(self, *args, **kargs):
        return TaggedDomain(self.a.squeeze(*args, **kargs), self.tag)

    def abstractApplyLeaf(self, *args, **kargs):
        return TaggedDomain(self.a.abstractApplyLeaf(*args, **kargs), self.tag)

    def view(self, *args, **kargs):
        return TaggedDomain(self.a.view(*args, **kargs), self.tag)

    def gather(self, *args, **kargs):
        return TaggedDomain(self.a.gather(*args, **kargs), self.tag)

    def sum(self, *args, **kargs):
        return TaggedDomain(self.a.sum(*args,**kargs), self.tag)

    def double(self):
        return TaggedDomain(self.a.double(), self.tag)

    def float(self):
        return TaggedDomain(self.a.float(), self.tag)

    def to_dtype(self):
        return TaggedDomain(self.a.to_dtype(), self.tag)

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

    def splitRelu(self, *args, **kargs):
        return TaggedDomain(self.a.splitRelu(*args, **kargs), self.tag)

    def diameter(self):
        return self.a.diameter()

    def softplus(self):
        return TaggedDomain(self.a.softplus(), self.tag)

    def elu(self):
        return TaggedDomain(self.a.elu(), self.tag)

    def selu(self):
        return TaggedDomain(self.a.selu(), self.tag)

    def sigm(self):
        return TaggedDomain(self.a.sigm(), self.tag)


    def cat(self, other, *args, **kargs):
        return TaggedDomain(self.a.cat(other.a, *args, **kargs), self.tag)

    def split(self, *args, **kargs):
        return [TaggedDomain(z, self.tag) for z in self.a.split(*args, **kargs)]

    def size(self):
        
        return self.a.size()

    def loss(self, *args, **kargs):
        return self.tag.loss(self.a, *args, **kargs)

    def deep_loss(self, *args, **kargs):
        return self.a.deep_loss(*args, **kargs)

    def checkSizes(self):
        self.a.checkSizes()
        return self

    def merge(self, other, ref = None):
        return TaggedDomain(self.a.merge(other.a, ref = None if ref is None else ref.a), self.tag)
