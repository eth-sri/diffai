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
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset

import inspect
from inspect import getargspec
import os
import helpers as h
from helpers import Timer
import copy
import random
from itertools import count

from components import *
import models

import goals
from goals import *
import math

from torch.serialization import SourceChangeWarning
import warnings


parser = argparse.ArgumentParser(description='Convert a pickled PyTorch DiffAI net to an abstract onyx net which returns the interval concretization around the final logits.  The first dimension of the output is the natural center, the second dimension is the lb, the third is the ub',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-n', '--net', type=str, default=None, metavar='N', help='Saved and pickled net to use, in pynet format', required=True)
parser.add_argument('-d', '--domain', type=str, default="Point()", help='picks which abstract goals to use for testing.  Uses box.  Doesn\'t use time, so don\'t use Lin.  Unless point, should specify a width w.')
parser.add_argument('-b', '--batch-size', type=int, default=1, help='The batch size to export.  Not sure this matters.')

parser.add_argument('-o', '--out', type=str, default="convert_out/", metavar='F', help='Where to save the net.')

parser.add_argument('--update-net', type=h.str2bool, nargs='?', const=True, default=False, help="should update test net")
parser.add_argument('--net-name', type=str, choices = h.getMethodNames(models), default=None, help="update test net name")

parser.add_argument('--save-name', type=str, default=None, help="name to save the net with.  Defaults to <domain>___<netfile-.pynet>.onyx")

parser.add_argument('-D', '--dataset', choices = [n for (n,k) in inspect.getmembers(datasets, inspect.isclass) if issubclass(k, Dataset)]
                    , default="MNIST", help='picks which dataset to use.')

parser.add_argument('--map-to-cpu', type=h.str2bool, nargs='?', const=True, default=False, help="map cuda operations in save back to cpu; enables to run on a computer without a GPU")

parser.add_argument('--tf-input', type=h.str2bool, nargs='?', const=True, default=False, help="change the shape of the input data from batch-channels-height-width (standard in pytroch) to batch-height-width-channels (standard in tf)")

args = parser.parse_args()

out_dir = args.out

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", SourceChangeWarning)
    if args.map_to_cpu:
        net = torch.load(args.net, map_location='cpu')
    else:
        net = torch.load(args.net)

net_name = None

if args.net_name is not None:
    net_name = args.net_name
elif args.update_net and 'name' in dir(net):
    net_name = net.name
    

def buildNet(n, input_dims, num_classes):
    n = n(num_classes)
    if args.dataset in ["MNIST"]:
        n = Seq(Normalize([0.1307], [0.3081] ), n)
    elif args.dataset in ["CIFAR10", "CIFAR100"]:
        n = Seq(Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), n)
    elif dataset in ["SVHN"]:
        n = Seq(Normalize([0.5,0.5,0.5], [0.2, 0.2, 0.2]), n)
    elif dataset in ["Imagenet12"]:
        n = Seq(Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]), n)

    n = n.infer(input_dims)
    n.clip_norm()
    return n


if net_name is not None:
    n = getattr(models,net_name)
    n = buildNet(n, net.inShape, net.outShape)
    n.load_state_dict(net.state_dict())
    net = n

net = net.to(h.device)
net.remove_norm()

domain = eval(args.domain)

if args.save_name is None:
    save_name = h.prepareDomainNameForFile(args.domain) + "___" + os.path.basename(args.net)[:-6] + ".onyx"  
else:
    save_name = args.save_name

def abstractNet(inpt):
    if args.tf_input:
        inpt = inpt.permute(0, 3, 1, 2)
    dom = domain.box(inpt, w = None)
    o = net(dom, onyx=True).unsqueeze(1)

    out = torch.cat([o.vanillaTensorPart(), o.lb().vanillaTensorPart(), o.ub().vanillaTensorPart()], dim=1)
    return out

input_shape = [args.batch_size] + list(net.inShape)
if args.tf_input:
    input_shape = [args.batch_size] + list(net.inShape)[1:]  + [net.inShape[0]]
dummy = h.zeros(input_shape)

abstractNet(dummy)

class AbstractNet(nn.Module):
    def __init__(self, domain, net, abstractNet):
        super(AbstractNet, self).__init__()
        self.net = net
        self.abstractNet = abstractNet
        if hasattr(domain, "net") and domain.net is not None:
            self.netDom = domain.net

    def forward(self, inpt):
        return self.abstractNet(inpt)

absNet = AbstractNet(domain, net, abstractNet)

out_path = os.path.join(out_dir,  save_name)
print("Saving:", out_path)

param_list = ["param"+str(i) for i in range(len(list(absNet.parameters())))]

torch.onnx.export(absNet, dummy, out_path, verbose=False, input_names=["actual_input"] + param_list, output_names=["output"])

