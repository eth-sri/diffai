import future
import builtins
import past
import six
import inspect
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import decimal
import PIL
from torchvision import datasets, transforms
from datetime import datetime

from forbiddenfruit import curse
#from torch.autograd import Variable

from timeit import default_timer as timer

class Timer:
    def __init__(self, activity = None, units = 1, shouldPrint = True, f = None):
        self.activity = activity
        self.units = units
        self.shouldPrint = shouldPrint
        self.f = f
    def __enter__(self):
        self.start = timer()
        return self
    def getUnitTime(self):
        return (self.end - self.start) / self.units

    def __str__(self):
        return "Avg time to " + self.activity + ": "+str(self.getUnitTime())

    def __exit__(self, *args):
        self.end = timer()
        if self.shouldPrint:
            printBoth(self, f = self.f)
            
def cudify(x):
    if use_cuda:
        return x.cuda(async=True)
    return x

def pyval(a, **kargs):
    return dten([a], **kargs)

def ifThenElse(cond, a, b):
    cond = cond.to_dtype()
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
    return cudify(torch.sparse.FloatTensor(ltenCPU(indexes), ftenCPU(values), torch.Size([bs,d])))

def seye(n, m = None): 
    if m is None:
        m = n
    mn = n if n < m else m
    indexes = [[ i for i in range(mn) ], [ i  for i in range(mn) ] ]
    values = [1 for i in range(mn) ]
    return cudify(torch.sparse.ByteTensor(ltenCPU(indexes), dtenCPU(values), torch.Size([n,m])))

dtype = torch.float32
ftype = torch.float32
ltype = torch.int64
btype = torch.uint8

torch.set_default_dtype(dtype)

cpu = torch.device("cpu")

cuda_async = True

ftenCPU = lambda *args, **kargs: torch.tensor(*args, dtype=ftype, device=cpu, **kargs)
dtenCPU = lambda *args, **kargs: torch.tensor(*args, dtype=dtype, device=cpu, **kargs)
ltenCPU = lambda *args, **kargs: torch.tensor(*args, dtype=ltype, device=cpu, **kargs)
btenCPU = lambda *args, **kargs: torch.tensor(*args, dtype=btype, device=cpu, **kargs)

if torch.cuda.is_available() and not 'NOCUDA' in os.environ:
    print("using cuda")
    device = torch.device("cuda")
    ften = lambda *args, **kargs: torch.tensor(*args, dtype=ftype, device=device, **kargs).cuda(non_blocking=cuda_async)
    dten = lambda *args, **kargs: torch.tensor(*args, dtype=dtype, device=device, **kargs).cuda(non_blocking=cuda_async)
    lten = lambda *args, **kargs: torch.tensor(*args, dtype=ltype, device=device, **kargs).cuda(non_blocking=cuda_async)
    bten = lambda *args, **kargs: torch.tensor(*args, dtype=btype, device=device, **kargs).cuda(non_blocking=cuda_async)
    ones = lambda *args, **cargs: torch.ones(*args, **cargs).cuda(non_blocking=cuda_async)
    zeros = lambda *args, **cargs: torch.zeros(*args, **cargs).cuda(non_blocking=cuda_async)
    eye = lambda *args, **cargs: torch.eye(*args, **cargs).cuda(non_blocking=cuda_async)
    use_cuda = True
    print("set up cuda")
else:
    print("not using cuda")
    ften = ftenCPU
    dten = dtenCPU
    lten = ltenCPU
    bten = btenCPU
    ones = torch.ones
    zeros = torch.zeros
    eye = torch.eye
    use_cuda = False
    device = cpu

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


def printBoth(*st, f = None):
    print(*st)
    if not f is None:
        print(*st, file=f)


def hasMethod(cl, mt):
    return callable(getattr(cl, mt, None))

def getMethodNames(Foo): 
    return [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]

def getMethods(Foo): 
    return [getattr(Foo, m) for m in getMethodNames(Foo)]

max_c_for_norm = 10000

def numel(arr):
    return product(arr.size())

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def loadDataset(dataset, batch_size, train, transform = True):
    oargs = {}
    if dataset in ["MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "PhotoTour"]:
        oargs['train'] = train
    elif dataset in ["STL10", "SVHN"] :
        oargs['split'] = 'train' if train else 'test'
    elif dataset in ["LSUN"]:
        oargs['classes'] = 'train' if train else 'test'
    elif dataset in ["Imagenet12"]:
        pass
    else:
        raise Exception(dataset + " is not yet supported")

    if dataset in ["MNIST"]:
        transformer = transforms.Compose([ transforms.ToTensor()]
                                         + ([transforms.Normalize((0.1307,), (0.3081,))] if transform else []))
    elif dataset in ["CIFAR10", "CIFAR100"]:
        transformer = transforms.Compose(([ #transforms.RandomCrop(32, padding=4), 
                                            transforms.RandomAffine(0, (0.125, 0.125), resample=PIL.Image.BICUBIC) ,
                                            transforms.RandomHorizontalFlip(), 
                                            #transforms.RandomRotation(15, resample = PIL.Image.BILINEAR) 
                                          ] if train else [])
                                         + [transforms.ToTensor()] 
                                         + ([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] if transform else []))
    elif dataset in ["SVHN"]:
        transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))])
    else:
        transformer = transforms.ToTensor()

    if dataset in ["Imagenet12"]:
        # https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
        train_set = datasets.ImageFolder(
            '../data/Imagenet12/train' if train else '../data/Imagenet12/val',
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                normalize,
            ]))
    else:
        train_set = getattr(datasets, dataset)('../data', download=True, transform=transformer, **oargs)
    return torch.utils.data.DataLoader(
        train_set
        , batch_size=batch_size
        , shuffle=True, 
        **({'num_workers': 1, 'pin_memory': True} if use_cuda else {}))


def variable(Pt):
    class Point:
        def isSafe(self,target):
            pred = self.max(1, keepdim=True)[1] # get the index of the max log-probability
            return pred.eq(target.data.view_as(pred))

        def isPoint(self):
            return True

        def labels(self):
            return [self[0].max(1)[1]] # get the index of the max log-probability
            
        def softplus(self): 
            return F.softplus(self)

        def elu(self): 
            return F.elu(self)

        def selu(self): 
            return F.selu(self)

        def sigm(self): 
            return F.sigmoid(self)
        
        def conv3d(self, *args, **kargs): 
            return F.conv3d(self, *args, **kargs)
        def conv2d(self, *args, **kargs): 
            return F.conv2d(self, *args, **kargs)
        def conv1d(self, *args, **kargs): 
            return F.conv1d(self, *args, **kargs)

        def conv_transpose3d(self, *args, **kargs): 
            return F.conv_transpose3d(self, *args, **kargs)
        def conv_transpose2d(self, *args, **kargs): 
            return F.conv_transpose2d(self, *args, **kargs)
        def conv_transpose1d(self, *args, **kargs): 
            return F.conv_transpose1d(self, *args, **kargs)

        def max_pool2d(self, *args, **kargs): 
            return F.max_pool2d(self, *args, **kargs)

        def avg_pool2d(self, *args, **kargs): 
            return F.avg_pool2d(self, *args, **kargs)

        def adaptive_avg_pool2d(self, *args, **kargs): 
            return F.adaptive_avg_pool2d(self, *args, **kargs)


        def cat(self, other, dim = 0, **kargs): 
            return torch.cat((self, other), dim = dim, **kargs)

        def addPar(self, a, b):
            return a + b

        def abstractApplyLeaf(self, foo, *args, **kargs):
            return self
        
        def diameter(self): 
            return pyval(0)

        def to_dtype(self): 
            return self.type(dtype=dtype, non_blocking=cuda_async)

        def loss(self, target, **kargs):
            if torch.__version__[0] == "0":
                return F.cross_entropy(self, target, reduce = False)
            else:
                return F.cross_entropy(self, target, reduction='none')

        def deep_loss(self, *args, **kargs):
            return 0

        def merge(self, *args, **kargs):
            return self

        def splitRelu(self, *args, **kargs):
            return self

        def lb(self): 
            return self
        def vanillaTensorPart(self):
            return self
        def center(self): 
            return self
        def ub(self): 
            return self

        def cudify(self, cuda_async = True):
            return self.cuda(non_blocking=cuda_async) if use_cuda else self
    
    def log_softmax(self, *args, dim = 1, **kargs): 
        return F.log_softmax(self, *args,dim = dim, **kargs)       

    if torch.__version__[0] == "0" and torch.__version__ != "0.4.1":
        Point.log_softmax = log_softmax


    def log_softmax(self, *args, dim = 1, **kargs): 
        return F.log_softmax(self, *args,dim = dim, **kargs)       

    if torch.__version__[0] == "0" and torch.__version__ != "0.4.1":
        Point.log_softmax = log_softmax

    for nm in getMethodNames(Point):
        curse(Pt, nm, getattr(Point, nm))

variable(torch.autograd.Variable)
variable(torch.cuda.DoubleTensor)
variable(torch.DoubleTensor)
variable(torch.cuda.FloatTensor)
variable(torch.FloatTensor)
variable(torch.ByteTensor)
variable(torch.Tensor)


def default(dic, nm, d):
    if dic is not None and nm in dic:
        return dic[nm]
    return d




def softmaxBatchNP(x, epsilon, subtract = False):
    """Compute softmax values for each sets of scores in x."""
    x = x.astype(np.float64)
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

def msum(a,b, l):
    if a is None:
        return b
    if b is None:
        return a
    return l(a,b)

class SubAct(argparse.Action):
    def __init__(self, sub_choices, *args, **kargs):
        super(SubAct,self).__init__(*args, nargs='+', **kargs)
        self.sub_choices = sub_choices
        self.sub_choices_names = None if sub_choices is None else getMethodNames(sub_choices)

    def __call__(self, parser, namespace, values, option_string=None):
        if self.sub_choices_names is not None and not values[0] in self.sub_choices_names:
            msg = 'invalid choice: %r (choose from %s)' % (values[0], self.sub_choices_names)
            raise argparse.ArgumentError(self, msg)

        prev = getattr(namespace, self.dest)
        setattr(namespace, self.dest, prev + [values])

def catLists(val):
    if isinstance(val, list):
        v = []
        for i in val:
            v += catLists(i)
        return v
    return [val]

def sumStr(val):
    s = ""
    for v in val:
        s += v
    return s

def catStrs(val):
    s = val[0]
    if len(val) > 1:
        s += "("
    for v in val[1:2]:
        s += v
    for v in val[2:]:
        s += ", "+v
    if len(val) > 1:
        s += ")"
    return s

def printNumpy(x):
    return "[" + sumStr([decimal.Decimal(float(v)).__format__("f") + ", " for v in x.data.cpu().numpy()])[:-2]+"]"

def printStrList(x):
    return "[" + sumStr(v + ", " for v in x)[:-2]+"]"

def printListsNumpy(val):
    if isinstance(val, list):
        return printStrList(printListsNumpy(v) for v in val)
    return printNumpy(val)

def parseValues(values, methods, *others):
    if len(values) == 1 and values[0]:
        x = eval(values[0], dict(pair for l in ([methods] + list(others)) for pair in l.__dict__.items()) )

        return x() if inspect.isclass(x) else x
    args = []
    kargs = {}
    for arg in values[1:]:
        if '=' in arg:
            k = arg.split('=')[0]
            v = arg[len(k)+1:]
            try:
                kargs[k] = eval(v)
            except:
                kargs[k] = v
        else:
            args += [eval(arg)]
    return getattr(methods, values[0])(*args, **kargs)

def preDomRes(outDom, target): # TODO: make faster again by keeping sparse tensors sparse
    t = one_hot(target.long(), outDom.size()[1]).to_dense().to_dtype()
    tmat = t.unsqueeze(2).matmul(t.unsqueeze(1))

    tl = t.unsqueeze(2).expand(-1, -1, tmat.size()[1])

    inv_t = eye(tmat.size()[1]).expand(tmat.size()[0], -1, -1)
    inv_t = inv_t - tmat

    tl = tl.bmm(inv_t)

    fst = outDom.unsqueeze(1).matmul(tl).squeeze(1)
    snd = outDom.unsqueeze(1).matmul(inv_t).squeeze(1)
        
    return (fst - snd) + t

def mopen(shouldnt, *args, **kargs):
    if shouldnt:
        import contextlib
        return contextlib.suppress()
    return open(*args, **kargs)
        
def file_timestamp():
    return str(datetime.now()).replace(":","").replace(" ", "")

def prepareDomainNameForFile(s):
    return s.replace(" ", "_").replace(",", "").replace("(", "_").replace(")", "_").replace("=", "_")

# delimited only
def callCC(foo):
    class RV(BaseException):
        def __init__(self, v):
            self.v = v

    def cc(x):
        raise RV(x)

    try:
        return foo(cc)
    except RV as rv:
        return rv.v
