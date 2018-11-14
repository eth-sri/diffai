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
import copy

from timeit import default_timer as timer
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import Dataset
import decimal
import torch.onnx


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

import warnings
from torch.serialization import SourceChangeWarning

POINT_DOMAINS = [m for m in h.getMethods(domains) if m.Domain is h.dtype]
SYMETRIC_DOMAINS = [domains.Box] + POINT_DOMAINS


class Top(nn.Module):
    def __init__(self, args, net, ty = Point):
        super(Top, self).__init__()
        self.net = net
        self.ty = ty
        self.w = args.width
        self.global_num = 0
        self.getSpec = getattr(self, args.spec)
        self.sub_batch_size = args.sub_batch_size
        self.curve_width = args.curve_width
        self.regularize = args.regularize


        self.speedCount = 0
        self.speed = 0.0

    def addSpeed(self, s):
        self.speed = (s + self.speed * self.speedCount) / (self.speedCount + 1)
        self.speedCount += 1

    def forward(self, x):
        return self.net(x)

    def clip_norm(self):
        self.net.clip_norm()

    def boxSpec(self, x, target):
        return [(self.ty.box(x, self.w, model=self, target=target, untargeted=True), target)]

    def curveSpec(self, x, target):
        if self.ty.__class__ in SYMETRIC_DOMAINS:
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
        batchedTargs = h.chunks(newTargs, new_batch_size)
        batchedSpecs = h.chunks(newSpecs, new_batch_size)
        batchedBest = h.chunks(bestSpecs, new_batch_size)

        def batch(t,s,b):
            t = h.ltype(t)
            s = torch.stack(s)
            b = torch.stack(b)

            if h.use_cuda:
                t.cuda()
                s.cuda()
                b.cuda()

            m = self.ty.line(s, b, self.curve_width)
            return (m , t)

        return [ batch(t,s,b) for t,s,b in zip(batchedTargs, batchedSpecs, batchedBest)]


    def regLoss(self):
        if self.regularize is None:
            return 0
        reg_loss = 0
        for param in self.parameters():
            reg_loss += param.norm(2)
        return self.regularize * reg_loss
        
    def aiLoss(self, dom, target, **args):
        return self.regLoss()  +  self.ty.loss(self(dom), target, **args)

    def printNet(self, f):
        self.net.printNet(f)

        
# Training settings
parser = argparse.ArgumentParser(description='PyTorch DiffAI Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='input batch size for training')
parser.add_argument('--test-freq', type=int, default=1, metavar='N', help='number of epochs to skip before testing')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N', help='input batch size for testing')
parser.add_argument('--sub-batch-size', type=int, default=3, metavar='N', help='input batch size for curve specs')

parser.add_argument('--test', type=str, default=None, metavar='net', help='Saved net to use, in addition to any other nets you specify with -n')
parser.add_argument('--update-test-net',type=h.str2bool, nargs='?', const=True, default=False, help="should update test net")
parser.add_argument('--onyx', type=h.str2bool, nargs='?', const=True, default=False, help="should output onyx")
parser.add_argument('--update-test-net-name', type=str, choices = h.getMethodNames(models), default=None, help="update test net name")

parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--log-freq', type=int, default=10, metavar='N', help='The frequency with which log statistics are printed')
parser.add_argument('--save-freq', type=int, default=1, metavar='N', help='The frequency with which nets and images are saved')
parser.add_argument('--number-save-images', type=int, default=0, metavar='N', help='The number of images to save. Should be smaller than test-size.')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--threshold', type=float, default=-0.01, metavar='TH', help='threshold for lr schedule')
parser.add_argument('--patience', type=int, default=0, metavar='PT', help='patience for lr schedule')
parser.add_argument('--factor', type=float, default=0.5, metavar='R', help='reduction multiplier for lr schedule')
parser.add_argument('--max-norm', type=float, default=10000, metavar='MN', help='the maximum norm allowed in weight distribution')

parser.add_argument('--curve-width', type=float, default=None, metavar='CW', help='the width of the curve spec')

parser.add_argument('--width-weight', type=float, default=0, metavar='CW', help='the weight of width in a combined loss')
parser.add_argument('--tot-weight', type=float, default=1, metavar='TW', help='the weight of domain total in a combined loss')

parser.add_argument('--width', type=float, default=0.01, metavar='CW', help='the width of either the line or box')
parser.add_argument('--spec', choices = [ x for x in dir(Top) if x[-4:] == "Spec" and len(getargspec(getattr(Top, x)).args) == 3]
                    , default="boxSpec", help='picks which spec builder function to use for training')


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument("--use-schedule", type=h.str2bool, nargs='?', 
                    const=True, default=False,
                    help="activate learning rate schedule")

parser.add_argument('-d', '--domain', sub_choices = None, action = h.SubAct
                    , default=[], help='picks which abstract domains to use for training', required=True)

parser.add_argument('-t', '--test-domain', sub_choices = None, action = h.SubAct
                    , default=[], help='picks which abstract domains to use for testing.  Examples include ' + str(domains), required=True)

parser.add_argument('-n', '--net', choices = h.getMethodNames(models), action = 'append'
                    , default=[], help='picks which net to use for training')  # one net for now

parser.add_argument('-D', '--dataset', choices = [n for (n,k) in inspect.getmembers(datasets, inspect.isclass) if issubclass(k, Dataset)]
                    , default="MNIST", help='picks which dataset to use.')

parser.add_argument('-o', '--out', default="out/", help='picks which net to use for training')
parser.add_argument('--dont-write', type=h.str2bool, nargs='?', const=True, default=False, help='dont write anywhere if this flag is on')
parser.add_argument('--test-size', type=int, default=2000, help='number of examples to test with')

parser.add_argument('-r', '--regularize', type=float, default=None, help='use regularization')


args = parser.parse_args()

largest_domain = max([len(h.catStrs(d)) for d in (args.domain)] )
largest_test_domain = max([len(h.catStrs(d)) for d in (args.test_domain)] )

args.log_interval = int(50000 / (args.batch_size * args.log_freq))

h.max_c_for_norm = args.max_norm

if h.use_cuda:
    torch.cuda.manual_seed(1 + args.seed)    
else:
    torch.manual_seed(args.seed)

train_loader = h.loadDataset(args.dataset, args.batch_size, True, False)
test_loader = h.loadDataset(args.dataset, args.test_batch_size, False, False)

input_dims = train_loader.dataset[0][0].size()
num_classes = int(max(getattr(train_loader.dataset, 'train_labels' if args.dataset != "SVHN" else 'labels'))) + 1

print("input_dims: ", input_dims)
print("Num classes: ", num_classes)

vargs = vars(args)

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

                    loss = model.aiLoss(*s, **vargs).sum() / data.size()[0]
                    lossy += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    model.optimizer.step()
                    model.clip_norm()
            
            model.addSpeed(timer.getUnitTime())

            if batch_idx % args.log_interval == 0:
                print(('Train Epoch {:12} {:'+ str(largest_domain) +'}: {:3} [{:7}/{} ({:.0f}%)] \tAvg sec/ex {:1.8f}\tLoss: {:.6f}').format(
                    model.name,  model.ty.name,
                    epoch, 
                    batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                    model.speed,
                    lossy))

    
num_tests = 0                
def test(models, epoch, f = None):
    global num_tests
    num_tests += 1
    class MStat:
        def __init__(self, model):
            model.eval()
            self.model = model
            self.correct = 0
            class Stat:
                def __init__(self, d, dnm):
                    self.domain = d
                    self.name = dnm
                    self.width = 0
                    self.max_eps = 0
                    self.safe = 0
                    self.proved = 0
                    self.time = 0
            self.domains = [ Stat(h.parseValues(domains,d), h.catStrs(d)) for d in args.test_domain ]
    model_stats = [ MStat(m) for m in models ]
        
    num_its = 0
    saved_data_target = []
    for data, target in test_loader:
        if num_its >= args.test_size:
            break

        if num_tests == 1:
            saved_data_target += list(zip(list(data), list(target)))
        
        num_its += data.size()[0]
        if h.use_cuda:
            data, target = data.cuda(), target.cuda()

        for m in model_stats:

            with torch.no_grad():
                pred = m.model(data).data.max(1, keepdim=True)[1] # get the index of the max log-probability
                m.correct += pred.eq(target.data.view_as(pred)).sum()

            for stat in m.domains:
                timer = Timer(shouldPrint = False)
                with timer:
                    def calcData(data, target):
                        box = stat.domain.box(data, m.model.w, model=m.model, untargeted = True, target=target)
                        with torch.no_grad():
                            bs = m.model(box)
                            org = m.model(data).max(1,keepdim=True)[1]
                            stat.width += bs.diameter().sum().item() # sum up batch loss
                            stat.proved += bs.isSafe(org).sum().item()
                            stat.safe += bs.isSafe(target).sum().item()
                            stat.max_eps += 0 # TODO: calculate max_eps

                    if m.model.net.neuronCount() < 5000 or stat.domain in SYMETRIC_DOMAINS:
                        calcData(data, target)
                    else:
                        for d,t in zip(data, target):
                            calcData(d.unsqueeze(0),t.unsqueeze(0))
                stat.time += timer.getUnitTime()
                
    l = num_its # len(test_loader.dataset)
    for m in model_stats:

        pr_corr = float(m.correct) / float(l)
        if args.use_schedule:
            m.model.lrschedule.step(1 - pr_corr)
        
        h.printBoth(('Test: {:12} trained with {:'+ str(largest_domain) +'} - Avg sec/ex {:1.12f}, Accuracy: {}/{} ({:3.1f}%)').format(
            m.model.name, m.model.ty.name,
            m.model.speed,
            m.correct, l, 100. * pr_corr), f = f)
        
        model_stat_rec = ""
        for stat in m.domains:
            pr_safe = stat.safe / l
            pr_proved = stat.proved / l
            pr_corr_given_proved = pr_safe / pr_proved if pr_proved > 0 else 0.0
            h.printBoth(("\t{:" + str(largest_test_domain)+"} - Width: {:<36.16f} Pr[Proved]={:<1.3f}  Pr[Corr and Proved]={:<1.3f}  Pr[Corr|Proved]={:<1.3f} AvgMaxEps: {:1.10f} Time = {:<7.5f}").format(
                stat.name, 
                stat.width / l, 
                pr_proved, 
                pr_safe, pr_corr_given_proved, 
                stat.max_eps / l,
                stat.time), f = f)
            model_stat_rec += "{}_{:1.3f}_{:1.3f}_{:1.3f}__".format(stat.name, pr_proved, pr_safe, pr_corr_given_proved)
        prepedname = m.model.ty.name.replace(" ", "_").replace(",", "").replace("(", "_").replace(")", "_").replace("=", "_")
        net_file = os.path.join(out_dir, m.model.name +"__" +prepedname + "_checkpoint_"+str(epoch)+"_with_{:1.3f}".format(pr_corr))

        h.printBoth("\tSaving netfile: {}\n".format(net_file + ".net"), f = f)

        if num_tests % args.save_freq == 1 or args.save_freq == 1 and not args.dont_write:
            torch.save(m.model.net, net_file + ".pynet")
            
            with h.mopen(args.dont_write, net_file + ".net", "w") as f2:
                m.model.net.printNet(f2)
                f2.close()
            if args.onyx:
                nn = copy.deepcopy(m.model.net)
                nn.remove_norm()
                torch.onnx.export(nn, h.zeros([1] + list(input_dims)), net_file + ".onyx", 
                                  verbose=False, input_names=["actual_input"] + ["param"+str(i) for i in range(len(list(nn.parameters())))], output_names=["output"])


    if num_tests == 1 and not args.dont_write:
        img_dir = os.path.join(out_dir, "images")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for img_num,(img,target) in zip(range(args.number_save_images), saved_data_target[:args.number_save_images]):
            sz = ""
            for s in img.size():
                sz += str(s) + "x"
            sz = sz[:-1]

            img_file = os.path.join(img_dir, args.dataset + "_" + sz + "_"+ str(img_num))
            if img_num == 0:
                print("Saving image to: ", img_file + ".img")
            with open(img_file + ".img", "w") as imgfile:
                flatimg = img.view(h.product(img.size()))
                for t in flatimg.cpu():
                    print(decimal.Decimal(float(t)).__format__("f"), file=imgfile)
            with open(img_file + ".class" , "w") as imgfile:
                print(int(target.item()), file=imgfile)

def createModel(net, domain, domain_name):
    net_weights, net_create = net
    domain.name = domain_name

    net = net_create()
    net.load_state_dict(net_weights.state_dict())

    model = Top(args, net, domain)
    model.clip_norm()
    if h.use_cuda:
        model.cuda()

    model.optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.lrschedule = optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer,
        'min',
        patience=args.patience,
        threshold= args.threshold,
        min_lr=0.000001,
        factor=args.factor,
        verbose=True)

    net.name = net_create.__name__
    model.name = net_create.__name__

    return model

out_dir = os.path.join(args.out, args.dataset, str(args.net)[1:-1].replace(", ","_").replace("'",""),
                       args.spec, "width_"+str(args.width), str(datetime.now()).replace(":","").replace(" ", "") )

print("Saving to:", out_dir)

if not os.path.exists(out_dir) and not args.dont_write:
    os.makedirs(out_dir)

print("Starting Training with:")
with h.mopen(args.dont_write, os.path.join(out_dir, "config.txt"), "w") as f:
    for k in sorted(vars(args)):
        h.printBoth("\t"+k+": "+str(getattr(args,k)), f = f)
print("")

def buildNet(n):
    n = n(num_classes)
    if args.dataset in ["MNIST"]:
        n = Seq(Normalize([0.1307], [0.3081] ), n)
    elif args.dataset in ["CIFAR10", "CIFAR100"]:
        n = Seq(Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), n)
    elif dataset in ["SVHN"]:
        n = Seq(Normalize([0.5,0.5,0.5], [0.2, 0.2, 0.2]), n)

    n = n.infer(input_dims)
    n.clip_norm()
    return n

if not args.test is None:

    test_name = None

    def loadedNet():
        if test_name is not None:
            n = getattr(models,test_name)
            n = buildNet(n)
            n.clip_norm()
            return n
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SourceChangeWarning)
                return torch.load(args.test)

    net = loadedNet()
    if args.update_test_net_name is not None:
        test_name = args.update_test_net_name
    elif args.update_test_net and '__name__' in dir(net):
        test_name = net.__name__

    if test_name is not None:
        loadedNet.__name__ = test_name

    nets = [ (net, loadedNet) ]

elif args.net == []:
    raise Exception("Need to specify at least one net with either -n or --test")
else:
    nets = []

for n in args.net:
    m = getattr(models,n)
    net_create = (lambda m: lambda: buildNet(m))(m) # why doesn't python do scoping right?  This is a thunk.  It is bad.
    net_create.__name__ = n
    net = buildNet(m)
    net.__name__ = n
    nets += [ (net, net_create) ]

    print("Name: ", net_create.__name__)
    print("Number of Neurons (relus): ", net.neuronCount())
    print("Number of Parameters: ", sum([h.product(s.size()) for s in net.parameters()]))
    print()


if args.domain == []:
    models = [ createModel(net, domains.Box(args.width), "Box") for net in nets]
else:
    models = h.flat([[createModel(net, h.parseValues(domains,d), h.catStrs(d)) for net in nets] for d in args.domain])


with h.mopen(args.dont_write, os.path.join(out_dir, "log.txt"), "w") as f:
    startTime = timer()
    for epoch in range(1, args.epochs + 1):
        if (epoch - 1) % args.test_freq == 0:
            with Timer("test before epoch "+str(epoch),"sample", 10000):
                test(models, epoch, f)
        h.printBoth("Elapsed-Time: {:.2f}s\n".format(timer() - startTime), f = f)
        if args.epochs <= args.test_freq:
            break
        with Timer("train","sample", 60000):
            train(epoch, models)
