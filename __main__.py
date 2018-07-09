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

import future
import builtins
import past
import six

from timeit import default_timer as timer
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import inspect
from inspect import getargspec
import os
import helpers as h
from helpers import Timer
import copy
import random

from components import *
import models

import domains
from domains import *
import math

from losses import *

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Top(nn.Module):
    def __init__(self, args, net, ty = Point):
        super(Top, self).__init__()
        self.net = net
        self.ty = ty
        self.cw = args.width_weight

        self.tw = args.tot_weight
        self.sw = args.std_weight
        self.time_mult = args.time_mult * 100000
        self.mix_mult = args.mix_mult * 30000
        self.w = args.width
        self.global_num = 0
        self.getSpec = getattr(self, args.spec)
        self.sub_batch_size = args.sub_batch_size
        self.curve_width = args.curve_width
        self.use_log = args.use_log
        self.regularize = args.regularize
        self.log_ai_loss = args.log_ai_loss

        self.min_tw = args.tot_weight_min
        self.tw_time_pow = args.tot_weight_time_pow
        self.tw_time_falloff = pow(float(args.tot_weight_time), self.tw_time_pow)

        self.speedCount = 0
        self.speed = 0.0

        

    def addSpeed(self, s):
        self.speed = (s + self.speed * self.speedCount) / (self.speedCount + 1)
        self.speedCount += 1

    def forward(self, x):
        return self.net(x)

    def clip_norm(self):
        self.net.clip_norm()
    
    def getPred(self, x):
        return self(x).log_softmax()

    def widthL(self, outDom):
        return outDom.diameter()

    def widthDomL(self, dom):
        return self.widthL(self(dom))

    def boxSpec(self, x, target):
        if h.hasMethod(self.ty, "attack"):
            x = self.ty.attack(self , self.w, x, target)
            return [(x,None,target)]
        return [(x, self.ty.box(x, self.w), target)]

    def lineSpec(self, x, target):
        eps = h.ones(x.size()) * self.w
        return [(x, self.ty.line(x - eps, x + eps, None), target)]

    def mixSpec(self, x, target):
        if self.ty in SYMETRIC_DOMAINS:
            return self.boxSpec(x, target)
        tm = float(self.global_num) / float(self.time_mult)
        lg = math.log10(tm * tm + 10)
        pr_hbox = 1 - 1. / (lg * lg)
        s = random.uniform(0,1)
        if s < pr_hbox:
            return [(x, domains.HBox.box(x,self.w), target)]
        else:
            return [(x, domains.Box.box(x,self.w), target)]

    def curveSpec(self, x, target):
        if self.ty in SYMETRIC_DOMAINS:
            return self.boxSpec(x,target)
        

        batch_size = x.size()[0]

        newTargs = [ None for i in range(batch_size) ]
        newSpecs = [ None for i in range(batch_size) ]
        bestSpecs = [ None for i in range(batch_size) ]

        for i in range(batch_size):
            newTarg = target[i]
            newTargs[i] = newTarg
            newSpec = x[i]

            best_x = newSpec
            best_dist = float("inf")
            for j in range(batch_size):
                potTarg = target[j] 
                potSpec = x[j]
                if (not newTarg.data.equal(potTarg.data)) or i == j:
                    continue
                curr_dist = (newSpec - potSpec).norm(1).item()  # must experiment with the type of norm here
                if curr_dist <= best_dist:
                    best_x = potSpec

            newSpecs[i] = newSpec
            bestSpecs[i] = best_x
                
        new_batch_size = self.sub_batch_size
        batchedTargs = chunks(newTargs, new_batch_size)
        batchedSpecs = chunks(newSpecs, new_batch_size)
        batchedBest = chunks(bestSpecs, new_batch_size)

        def batch(t,s,b):
            t = h.ltype(t)
            s = torch.stack(s)
            b = torch.stack(b)

            if h.use_cuda:
                t.cuda()
                s.cuda()
                b.cuda()

            m = self.ty.line(s, b, self.curve_width)
            return (s, m , t)

        return [ batch(t,s,b) for t,s,b in zip(batchedTargs, batchedSpecs, batchedBest)]


    def stdLoss(self, x, dom, target, **args):
        return F.nll_loss(self.getPred(x), target, **args, size_average = False, reduce = False) # definitely includes averaging



    def preDomRes(self, outDom, target, **args): # TODO: make faster again by keeping sparse tensors sparse
        t = h.one_hot(target.data.long(), outDom.size()[1]).to_dense()
        tmat = t.unsqueeze(2).matmul(t.unsqueeze(1))

        tl = t.unsqueeze(2).expand(-1, -1, tmat.size()[1])

        inv_t = h.eye(tmat.size()[1]).expand(tmat.size()[0], -1, -1)
        inv_t = inv_t - tmat

        tl = tl.bmm(inv_t)

        fst = outDom.bmm(tl)
        snd = outDom.bmm(inv_t)
        
        return (fst - snd) + t
    def domRes(self, outDom, target, **args):
        return self.preDomRes(outDom, target, **args).lb()

    def outDomL(self, outDom, target, **args):
        r = -self.domRes(outDom,target, **args)
        return torch.exp( torch.clamp(r, -1000, 25) )

    def outRDomL(self, outDom, target, **args):
        r = -self.domRes(outDom,target, **args)
        out = F.softplus(r.max(1)[0])      # kinda works        
        return (out).pow(self.log_ai_loss) if not self.log_ai_loss is None and self.log_ai_loss > 0 else out


    def isSafeDom(self, outDom, target, **args):
        od,_ = torch.min(self.domRes(outDom,target, **args), 1)
        return od.gt(0.0).long()

    def regLoss(self):
        if self.regularize is None:
            return 0
        reg_loss = 0
        for param in self.parameters():
            reg_loss += param.norm(2)
        return self.regularize * reg_loss
        
    def domLoss(self, x, dom, target, **args):
        if self.ty in POINT_DOMAINS:
            return self.regLoss() + self.stdLoss(x,dom, target,**args)
        outDom = self(dom)
        return self.outDomL(outDom, target, **args) + self.regLoss()

    def domRLoss(self, x, dom, target, **args):
        if self.ty in POINT_DOMAINS:
            return self.regLoss() + self.stdLoss(x,dom, target,**args)
        outDom = self(dom)
        return self.outRDomL(outDom, target, **args) + self.regLoss()

    def getMult(self):
        time_passage = float(self.global_num) / float(self.time_mult)
        m = torch.log(h.pyval(time_passage + 2)) if self.use_log else h.pyval(1)
        return m * m * m * m * m * m
    
    def getTW(self):
        tm = pow(float(self.global_num), self.tw_time_pow)
        
        if tm >= self.tw_time_falloff:
            return self.tw
        return (self.tw - self.min_tw) * tm / self.tw_time_falloff + self.min_tw
        


    def totalLoss(self, x, dom, target, **args):
        if self.ty in POINT_DOMAINS:
            return self.regLoss() + self.stdLoss(x,dom,target,**args)
        outDom = self(dom)
        tw = self.tw * self.getMult()
        cw = self.cw
        sw = self.sw
        return self.regLoss() + ((self.outRDomL(outDom, target, **args) + 1 / tw) * ((self.widthL(outDom) * cw + 1) * self.stdLoss(x, dom, target, **args) + 1 / sw) - 1 / ( sw * tw) )

    def aiLoss(self, x, dom, target, **args):
        if self.ty in POINT_DOMAINS:
            return self.regLoss()  +  self.stdLoss(x,dom,target,**args)
        outDom = self(dom)
        cw = self.cw / self.getMult()
        sw = self.sw
        tw = self.getTW() * self.getMult()

        if cw > 0:
            ww = self.widthL(outDom) * cw
        else:
            ww = 0

        if tw > 0:
            tt = self.outRDomL(outDom, target, **args) * tw
        else:
            tt = 0

        return self.regLoss() + (tt + ww + self.stdLoss(x, dom, target, **args) * sw) / (tw +  sw + cw)

    def fgsmLoss(self, x, dom, target, **args):
        if self.ty in POINT_DOMAINS:
            return self.regLoss()  +  self.stdLoss(x,dom,target,**args)
        atk = domains.FGSM.attack(self, self.w, x, target)
        outDom = self(dom)
        sw = self.sw
        tw = self.tw * self.getMult()
        return (self.outRDomL(outDom, target, **args) * tw 
                + self.stdLoss(atk, None, target, **args) * sw) / (tw + sw)

    def printNet(self, f):
        self.net.printNet(f)



        
# Training settings
parser = argparse.ArgumentParser(description='PyTorch DiffAI Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='input batch size for training')
parser.add_argument('--test-freq', type=int, default=1, metavar='N', help='number of epochs to skip before testing')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N', help='input batch size for testing')
parser.add_argument('--sub-batch-size', type=int, default=3, metavar='N', help='input batch size for curve specs')

parser.add_argument('--test', type=str, default=None, metavar='net', help='Saved net to use, in addition to any other nets you specify with -n')

parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--log-freq', type=int, default=10, metavar='N', help='The frequency with which log statistics are printed')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--threshold', type=float, default=-0.01, metavar='TH', help='threshold for lr schedule')
parser.add_argument('--patience', type=int, default=0, metavar='PT', help='patience for lr schedule')
parser.add_argument('--factor', type=float, default=0.5, metavar='R', help='reduction multiplier for lr schedule')
parser.add_argument('--max-norm', type=float, default=10000, metavar='MN', help='the maximum norm allowed in weight distribution')
parser.add_argument('--time-mult', type=float, default=1, metavar='MN', help='the time falloff for standard training')
parser.add_argument('--mix-mult', type=float, default=1, metavar='MN', help='the time falloff for mix-spec training')
parser.add_argument('--width-weight', type=float, default=0.0, metavar='CW', help='the weight of width in a combined loss')
parser.add_argument('--curve-width', type=float, default=None, metavar='CW', help='the width of the curve spec')

parser.add_argument('--tot-weight', type=float, default=0.001, metavar='TW', help='the weight of domain total in a combined loss')

parser.add_argument('--tot-weight-min', type=float, default=0, metavar='TW', help='the minimum weight of domain total in a combined loss')
parser.add_argument('--tot-weight-time', type=int, default=0, metavar='TW', help='the number of timesteps to use for domain weight scheduling')
parser.add_argument('--tot-weight-time-pow', type=float, default=1, metavar='TW', help='the time-power to use for domain weight scheduling')

parser.add_argument('--std-weight', type=float, default=1, metavar='TW', help='the weight of standard loss in a combined loss')
parser.add_argument('--width', type=float, default=0.01, metavar='CW', help='the width of either the line or box')
parser.add_argument('--loss', choices = [ x for x in dir(Top) if x[-4:] == "Loss" and len(getargspec(getattr(Top, x)).args) == 4]
                    , default="aiLoss", help='picks which loss function to use for training')
parser.add_argument('--spec', choices = [ x for x in dir(Top) if x[-4:] == "Spec" and len(getargspec(getattr(Top, x)).args) == 3]
                    , default="boxSpec", help='picks which spec builder function to use for training')


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument("--use-schedule", type=h.str2bool, nargs='?', 
                    const=True, default=False,
                    help="activate learning rate schedule")
parser.add_argument("--use-log", type=h.str2bool, nargs='?',
                    const=True, default=False,
                    help="Activate log falloff of time")

parser.add_argument('-d', '--domain', choices = h.getMethodNames(domains), action = 'append'
                    , default=[], help='picks which abstract domains to use for training', required=True)

parser.add_argument('-t', '--test-domain', choices = h.getMethodNames(domains), action = 'append'
                    , default=[], help='picks which abstract domains to use for testing', required=True)

parser.add_argument('-n', '--net', choices = h.getMethodNames(models), action = 'append'
                    , default=[], help='picks which net to use for training')  # one net for now

parser.add_argument('-D', '--dataset', choices = [n for (n,k) in inspect.getmembers(datasets, inspect.isclass) if issubclass(k, Dataset)]
                    , default="MNIST", help='picks which dataset to use.')

parser.add_argument('-o', '--out', default="out/", help='picks which net to use for training')
parser.add_argument('--test-size', type=int, default=2000, help='number of examples to test with')

parser.add_argument('-r', '--regularize', type=float, default=None, help='use regularization')
parser.add_argument('--log-ai-loss', type=float, default=None, help='use the log of the output. Must be non-negative. ')

args = parser.parse_args()

args.log_interval = int(50000 / (args.batch_size * args.log_freq))

h.max_c_for_norm = args.max_norm

if h.use_cuda:
    torch.cuda.manual_seed(1 + args.seed)    
else:
    torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if h.use_cuda else {}

def loadDataset(train):
    oargs = {}
    if args.dataset in ["MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "PhotoTour"]:
        oargs['train'] = train
    elif args.dataset in ["STL10", "SVHN"] :
        oargs['split'] = 'train' if train else 'test'
    elif args.dataset in ["LSUN"]:
        oargs['classes'] = 'train' if train else 'test'
    else:
        raise Exception(args.dataset + " is not yet supported")

    if args.dataset in ["MNIST"]:
        transformer = transforms.Compose([ transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    elif args.dataset in ["CIFAR10", "CIFAR100"]:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset in ["SVHN"]:
        transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2))])
    else:
        transformer = transforms.ToTensor()

    return torch.utils.data.DataLoader(
        getattr(datasets, args.dataset)('../data', download=True,
                                        transform=transformer, **oargs)
        , batch_size=args.batch_size if train else args.test_batch_size
        , shuffle=True, **kwargs)

train_loader = loadDataset(True)
test_loader = loadDataset(False)

input_dims = train_loader.dataset[0][0].size()
num_classes = int(max(getattr(train_loader.dataset, 'train_labels' if args.dataset != "SVHN" else 'labels'))) + 1

print("input_dims: ", input_dims)
print("Num classes: ", num_classes)
def train(epoch, models):
    for model in models:
        model.train()

    ep_tot = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if h.use_cuda:
            data, target = data.cuda(), target.cuda()

        for model in models:
            model.global_num += data.size()[0]

            timer = Timer("train", "sample from " + model.name + " with " + model.ty.name, data.size()[0], False)
            lossy = 0
            with timer:
                for s in model.getSpec(data,target):
                    model.optimizer.zero_grad()
                    loss = getattr(model, args.loss)(*s).sum() / data.size()[0]
                    lossy += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    model.optimizer.step()
                    model.clip_norm()
            
            model.addSpeed(timer.getUnitTime())

            if batch_idx % args.log_interval == 0:
                print('Train Epoch {:12} {:20}: {:3} [{:7}/{} ({:.0f}%)] \tAvg sec/ex {:1.8f}\tMult {:.6f}\tTW {:.12f}\tLoss: {:.6f}'.format(
                    model.name,  model.ty.name,
                    epoch, 
                    batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                    model.speed,
                    model.getMult().data[0],
                    model.getTW(),
                    lossy))

    
                
def test(models, epoch, f = None):
    class MStat:
        def __init__(self, model):
            model.eval()
            self.model = model
            self.correct = 0
            self.test_loss = 0    
            class Stat:
                def __init__(self, d, dnm):
                    self.domain = d
                    self.name = dnm
                    self.width = 0
                    self.safe = 0
                    self.proved = 0
                    self.time = 0
            self.domains = [ Stat(getattr(domains,d), d) for d in args.test_domain ]
    model_stats = [ MStat(m) for m in models ]
        
    num_its = 0
    for data, target in test_loader:
        if num_its >= args.test_size:
            break
        num_its += data.size()[0]
        if h.use_cuda:
            data, target = data.cuda(), target.cuda()

        for m in model_stats:
            with torch.no_grad():
                m.test_loss += m.model.stdLoss(data, None, target).sum().item() # sum up batch loss

            tyorg = m.model.ty

            with torch.no_grad():
                pred = m.model(data).data.max(1, keepdim=True)[1] # get the index of the max log-probability
                m.correct += pred.eq(target.data.view_as(pred)).sum()

            for stat in m.domains:
                timer = Timer(shouldPrint = False)
                with timer:
                    m.model.ty = stat.domain
                    def calcData(data, target):
                        box = m.model.boxSpec(data, target)[0]
                        with torch.no_grad():
                            if m.model.ty in POINT_DOMAINS:
                                preder = m.model(box[0]).data
                                pred = preder.max(1, keepdim=True)[1] # get the index of the max log-probability
                                org = m.model(data).max(1,keepdim=True)[1]
                                stat.proved += float(org.eq(pred).sum())
                                stat.safe += float(pred.eq(target.data.view_as(pred)).sum())
                            else: 
                                bs = m.model(box[1])
                                stat.width += m.model.widthL(bs).data[0] # sum up batch loss
                                stat.safe  += m.model.isSafeDom(bs, target).sum().item()
                                stat.proved += sum([ m.model.isSafeDom(bs, (h.ones(target.size()) * n).long() ).sum().item() for n in range(num_classes) ])
                    if m.model.net.neuronCount() < 5000 or stat.domain in SYMETRIC_DOMAINS:
                        calcData(data, target)
                    else:
                        for d,t in zip(data, target):
                            calcData(d.unsqueeze(0),t.unsqueeze(0))
                stat.time += timer.getUnitTime()
            m.model.ty = tyorg

                
    l = num_its # len(test_loader.dataset)
    for m in model_stats:

        pr_corr = float(m.correct) / float(l)
        if args.use_schedule:
            m.model.lrschedule.step(1 - pr_corr)
        
        h.printBoth('Test: {:12} trained with {:8} - Mult {:1.8f}, Avg sec/ex {:1.12f}, Average loss: {:8.4f}, Accuracy: {}/{} ({:3.1f}%)'.format(
            m.model.name, m.model.ty.name,
            m.model.getMult().data[0],
            m.model.speed,
            m.test_loss / l, 
            m.correct, l, 100. * pr_corr), f)
        
        model_stat_rec = ""
        for stat in m.domains:
            pr_safe = stat.safe / l
            pr_proved = stat.proved / l
            pr_corr_given_proved = pr_safe / pr_proved if pr_proved > 0 else 0.0
            h.printBoth("\t{:10} - Width: {:<22.4f} Pr[Proved]={:<1.3f}  Pr[Corr and Proved]={:<1.3f}  Pr[Corr|Proved]={:<1.3f}    Time = {:<7.5f}".format(
                stat.name, stat.width / l, pr_proved, pr_safe, pr_corr_given_proved, stat.time), f)
            model_stat_rec += "{}_{:1.3f}_{:1.3f}_{:1.3f}__".format(stat.name, pr_proved, pr_safe, pr_corr_given_proved)
        net_file = os.path.join(out_dir, m.model.name + "_checkpoint_"+str(epoch)+"_with_{:1.3f}".format(pr_corr)+"__"+model_stat_rec+".net")

        h.printBoth("\tSaving netfile: {}\n".format(net_file), f)

        if epoch == 1 or epoch % 10 == 0:
            torch.save(m.model.net, net_file)


def createModel(net, domain, domain_name):
    net_weights, net_create = net
    domain.name = domain_name

    net = net_create(num_classes).infer(input_dims)
    net.load_state_dict(net_weights.state_dict())

    model = Top(args, net, domain)
    model.clip_norm()
    if h.use_cuda:
        model.cuda()

    model.optimizer = optim.Adam(model.parameters(), lr=args.lr if not domain in POINT_DOMAINS else 1e-4)
    model.lrschedule = optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer,
        'min',
        patience=args.patience,
        threshold= args.threshold,
        min_lr=0.000001,
        factor=args.factor,
        verbose=True)

    model.name = net_create.__name__

    return model


out_dir = os.path.join(args.out, args.dataset, str(args.net)[1:-1].replace(", ","_").replace("'",""),
                       args.spec, "width_"+str(args.width), str(datetime.now()).replace(":","").replace(" ", "") )

print("Saving to:", out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Starting Training with:")
with open(os.path.join(out_dir, "config.txt"), "w") as f:
    for k in sorted(vars(args)):
        h.printBoth("\t"+k+": "+str(getattr(args,k)), f)
print("")

if not args.test is None:
    net = torch.load(args.test)
    nets = [ lambda *args, **kargs: net ]
elif args.net == []:
    raise Exception("Need to specify at least one net with either -n or --test")
else:
    nets = []

nets += [ getattr(models,n) for n in args.net ] 


nets = [ (n(num_classes).infer(input_dims),n) for n in nets ]

for net, net_create in nets:
    print("Name: ", net_create.__name__)
    print("Number of Neurons (relus): ", net.neuronCount())
    print("Number of Parameters: ", sum([h.product(s.size()) for s in net.parameters()]))
    print()


if args.domain == []:
    models = [ createModel(net, domains.Box, "Box") for net in nets]
else:
    models = h.flat([[createModel(net, getattr(domains,d), d) for net in nets] for d in args.domain])

with open(os.path.join(out_dir, "log.txt"), "w") as f:

    startTime = timer()
    for epoch in range(1, args.epochs + 1):
        if (epoch - 1) % args.test_freq == 0:
            with Timer("test before epoch "+str(epoch),"sample", 10000):
                test(models, epoch, f)
        h.printBoth("Elapsed-Time: {:.2f}s\n".format(timer() - startTime), f)
        with Timer("train","sample", 60000):
            train(epoch, models)
