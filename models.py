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

import components as n


def ffnn(c, **kargs):
    return n.FFNN([100, 100, 100, 100, 100,c], **kargs)

def convSmall(c, **kargs):
    return n.LeNet([ (16,4,4,2), (32,4,4,2) ], [100,c], last_lin = True, **kargs)

def convMed(c, **kargs):
    return n.LeNet([ (16,4,4,2), (32,4,4,2) ], [100,c], padding = 1, last_lin = True, **kargs)

def convBig(c, **kargs):
    return n.LeNet([ (32,3,3,1), (32,4,4,2) , (64,3,3,1), (64,4,4,2)], [512, 512,c], padding = 1, last_lin = True, **kargs)

def convSuper(c, **kargs):
    return n.LeNet([ (32,3,3,1), (32,4,4,1) , (64,3,3,1), (64,4,4,1)], [512, 512,c], last_lin = True, **kargs)

def convHuge(c, **kargs):
    return n.LeNet([ (128, 3, 3, 1), (128,4,4,1), (256,3,3,1), (256,4,4,1)], [512,512,c], padding=1, normal=True, bias=False, last_lin = True, **kargs)

def skip(c, **kargs):
    return n.SkipNet( n.LeNet([ (16,3,3), (16,3,3), (32,3,3) ], [200])
                    , n.LeNet([ (32,4,4), (32,4,4) ], [200])
                    , [200,c], **kargs )


def resnet18small(c, **kargs):
    return n.Seq(n.ResNet([2,2,2]), n.FFNN([100, c], bias=True, last_lin=False, **kargs))

def resnet18(c, **kargs):
    return n.Seq(n.ResNet([2,2,2,2]), n.FFNN([512, 512, c], bias=False, last_lin=True, **kargs))

def resnet34(c, **kargs):
    return n.Seq(n.ResNet([3,4,6,3]), n.FFNN([512, 512, c], bias=False, last_lin=True, **kargs))
