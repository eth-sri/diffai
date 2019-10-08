import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import multinomial, categorical
import torch.optim as optim

import math

try:
    from . import helpers as h
    from . import ai
    from . import scheduling as S
except:
    import helpers as h
    import ai
    import scheduling as S

import math
import abc

from torch.nn.modules.conv import _ConvNd
from enum import Enum


class InferModule(nn.Module):
    def __init__(self, *args, normal=False, ibp_init=False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.infered = False
        self.normal = normal
        self.ibp_init = ibp_init

    def infer(self, in_shape, global_args=None):
        """ this is really actually stateful. """

        if self.infered:
            return self
        self.infered = True

        super(InferModule, self).__init__()
        self.inShape = list(in_shape)
        self.outShape = list(self.init(list(in_shape), *self.args, global_args=global_args, **self.kwargs))
        if self.outShape is None:
            raise "init should set the out_shape"

        self.reset_parameters()
        return self

    def reset_parameters(self):
        if not hasattr(self, 'weight') or self.weight is None:
            return
        n = h.product(self.weight.size()) / self.outShape[0]
        stdv = 1 / math.sqrt(n)

        if self.ibp_init:
            torch.nn.init.orthogonal_(self.weight.data)
        elif self.normal:
            self.weight.data.normal_(0, stdv)
            self.weight.data.clamp_(-1, 1)
        else:
            self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            if self.ibp_init:
                self.bias.data.zero_()
            elif self.normal:
                self.bias.data.normal_(0, stdv)
                self.bias.data.clamp_(-1, 1)
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def clip_norm(self):
        if not hasattr(self, "weight"):
            return
        if not hasattr(self, "weight_g"):
            if torch.__version__[0] == "0":
                nn.utils.weight_norm(self, dim=None)
            else:
                nn.utils.weight_norm(self)

        self.weight_g.data.clamp_(-h.max_c_for_norm, h.max_c_for_norm)

        if torch.__version__[0] != "0":
            self.weight_v.data.clamp_(-h.max_c_for_norm * 10000, h.max_c_for_norm * 10000)
            if hasattr(self, "bias"):
                self.bias.data.clamp_(-h.max_c_for_norm * 10000, h.max_c_for_norm * 10000)

    def regularize(self, p):
        reg = 0
        if torch.__version__[0] == "0":
            for param in self.parameters():
                reg += param.norm(p)
        else:
            if hasattr(self, "weight_g"):
                reg += self.weight_g.norm().sum()
                reg += self.weight_v.norm().sum()
            elif hasattr(self, "weight"):
                reg += self.weight.norm().sum()

            if hasattr(self, "bias"):
                reg += self.bias.view(-1).norm(p=p).sum()

        return reg

    def remove_norm(self):
        if hasattr(self, "weight_g"):
            torch.nn.utils.remove_weight_norm(self)

    def showNet(self, t=""):
        print(t + self.__class__.__name__)

    def printNet(self, f):
        print(self.__class__.__name__, file=f)

    @abc.abstractmethod
    def forward(self, *args, **kargs):
        pass

    def __call__(self, *args, onyx=False, **kargs):
        if onyx:
            return self.forward(*args, onyx=onyx, **kargs)
        else:
            return super(InferModule, self).__call__(*args, **kargs)

    @abc.abstractmethod
    def neuronCount(self):
        pass

    def depth(self):
        return 0


def getShapeConv(in_shape, conv_shape, stride=1, padding=0):
    # print(in_shape)
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)


def getShapeConvTranspose(in_shape, conv_shape, stride=1, padding=0, out_padding=0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = (inH - 1) * stride - 2 * padding + kH + out_padding
    outW = (inW - 1) * stride - 2 * padding + kW + out_padding
    return (outChan, outH, outW)


class Embedding(InferModule):
    def init(self, in_shape, vocab, dim, **kargs):
        self.vocab = vocab
        self.dim = dim
        self.in_shape = in_shape
        self.embed = nn.Embedding(vocab, dim)

        return [1, in_shape[0], dim]

    def forward(self, x, **kargs):
        y = self.embed(x.long()).view(-1, 1, self.in_shape[0], self.dim)
        # print(x)
        return y

    def neuronCount(self):
        return 0

    def showNet(self, t=""):
        print(t + "Embedding out=" + str(self.in_shape[0]) + " * " + str(self.dim))

    def printNet(self, f):
        print("Embedding(" + str(self.in_shape[0]) + ", " + str(self.dim) + ")")

        print(h.printNumpy(self.embed.weight), file=f)


class EmbeddingWithSub(InferModule):
    def init(self, in_shape, vocab, dim, **kargs):
        self.vocab = vocab
        self.dim = dim
        self.in_shape = in_shape
        self.all_possible_sub = in_shape[0] * 2 - 2 + 1
        self.embed = nn.Embedding(vocab, dim)

        return [1, in_shape[0], dim]

    def forward(self, x, **kargs):
        if isinstance(x, ai.TaggedDomain):  # convert to Box (HybirdZonotope), if the input is Box
            x = x.center().vanillaTensorPart()
            x = x.repeat((1, self.all_possible_sub))
            for i in x:
                for j in range(1, self.all_possible_sub):
                    swap_id = j // 2
                    if swap_id == 0:
                        i[swap_id] = x[0][swap_id + 1]
                    elif swap_id == self.in_shape[0] - 1:
                        i[swap_id] = x[0][swap_id - 1]
                    else:
                        i[swap_id] = x[0][swap_id - 1 + (j % 2) * 2]
            # every position has 2 options, except for the first and the last one. Also, it can remain the same.
            # After which it becomes a batch * (max_length * 2 - 2 + 1) * max_length tensor
            y = self.embed(x.long()).view(-1, 1, self.in_shape[0], self.dim)
            return y
        else:  # convert to Point, if the input is Point
            y = self.embed(x.long()).view(-1, 1, self.in_shape[0], self.dim)
            return y

    def neuronCount(self):
        return 0

    def showNet(self, t=""):
        print(t + "EmbeddingWithSub out=" + str(self.in_shape[0]) + " * " + str(
            self.dim))

    def printNet(self, f):
        print("EmbeddingWithSub(" + str(self.in_shape[0]) + ", " + str(self.dim) + ")")

        print(h.printNumpy(self.embed.weight), file=f)


class Linear(InferModule):
    def init(self, in_shape, out_shape, **kargs):
        self.in_neurons = h.product(in_shape)
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        self.out_neurons = h.product(out_shape)

        self.weight = torch.nn.Parameter(torch.Tensor(self.in_neurons, self.out_neurons))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_neurons))

        return out_shape

    def forward(self, x, **kargs):
        s = x.size()
        x = x.view(s[0], h.product(s[1:]))
        return (x.matmul(self.weight) + self.bias).view(s[0], *self.outShape)

    def neuronCount(self):
        return 0

    def showNet(self, t=""):
        print(t + "Linear out=" + str(self.out_neurons))

    def printNet(self, f):
        print("Linear(" + str(self.out_neurons) + ")")

        print(h.printListsNumpy(list(self.weight.transpose(1, 0).data)), file=f)
        print(h.printNumpy(self.bias), file=f)


class Activation(InferModule):
    def init(self, in_shape, global_args=None, activation="ReLU", **kargs):
        self.activation = ["ReLU", "Sigmoid", "Tanh", "Softplus", "ELU", "SELU"].index(activation)
        self.activation_name = activation
        return in_shape

    def regularize(self, p):
        return 0

    def forward(self, x, **kargs):
        return \
            [lambda x: x.relu(), lambda x: x.sigmoid(), lambda x: x.tanh(), lambda x: x.softplus(), lambda x: x.elu(),
             lambda x: x.selu()][self.activation](x)

    def neuronCount(self):
        return h.product(self.outShape)

    def depth(self):
        return 1

    def showNet(self, t=""):
        print(t + self.activation_name)

    def printNet(self, f):
        pass


class ReLU(Activation):
    pass


def activation(*args, batch_norm=False, **kargs):
    a = Activation(*args, **kargs)
    return Seq(BatchNorm(), a) if batch_norm else a


class Identity(InferModule):  # for feigning model equivelence when removing an op
    def init(self, in_shape, global_args=None, **kargs):
        return in_shape

    def forward(self, x, **kargs):
        return x

    def neuronCount(self):
        return 0

    def printNet(self, f):
        pass

    def regularize(self, p):
        return 0

    def showNet(self, *args, **kargs):
        pass


class Dropout(InferModule):
    def init(self, in_shape, p=0.5, use_2d=False, alpha_dropout=False, **kargs):
        self.p = S.Const.initConst(p)
        self.use_2d = use_2d
        self.alpha_dropout = alpha_dropout
        return in_shape

    def forward(self, x, time=0, **kargs):
        if self.training:
            with torch.no_grad():
                p = self.p.getVal(time=time)
                mask = (F.dropout2d if self.use_2d else F.dropout)(h.ones(x.size()), p=p, training=True)
            if self.alpha_dropout:
                with torch.no_grad():
                    keep_prob = 1 - p
                    alpha = -1.7580993408473766
                    a = math.pow(keep_prob + alpha * alpha * keep_prob * (1 - keep_prob), -0.5)
                    b = -a * alpha * (1 - keep_prob)
                    mask = mask * a
                return x * mask + b
            else:
                return x * mask
        else:
            return x

    def neuronCount(self):
        return 0

    def showNet(self, t=""):
        print(t + "Dropout p=" + str(self.p))

    def printNet(self, f):
        print("Dropout(" + str(self.p) + ")")


class PrintActivation(Identity):
    def init(self, in_shape, global_args=None, activation="ReLU", **kargs):
        self.activation = activation
        return in_shape

    def printNet(self, f):
        print(self.activation, file=f)


class PrintReLU(PrintActivation):
    pass


class Conv2D(InferModule):

    def init(self, in_shape, out_channels, kernel_size, stride=1, global_args=None, bias=True, padding=0,
             activation="ReLU", **kargs):
        self.prev = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_softplus = h.default(global_args, 'use_softplus', False)

        weights_shape = (self.out_channels, self.in_channels, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(torch.Tensor(*weights_shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(weights_shape[0]))
        else:
            self.bias = None  # h.zeros(weights_shape[0])

        outshape = getShapeConv(in_shape, (out_channels, kernel_size, kernel_size), stride, padding)
        return outshape

    def forward(self, input, **kargs):
        return input.conv2d(self.weight, bias=self.bias, stride=self.stride, padding=self.padding)

    def printNet(self, f):  # only complete if we've forwardt stride=1
        print("Conv2D", file=f)
        sz = list(self.prev)
        print(self.activation + ", filters={}, kernel_size={}, input_shape={}, stride={}, padding={}".format(
            self.out_channels, [self.kernel_size, self.kernel_size], list(reversed(sz)), [self.stride, self.stride],
            self.padding), file=f)
        print(h.printListsNumpy([[list(p) for p in l] for l in self.weight.permute(2, 3, 1, 0).data]), file=f)
        print(h.printNumpy(self.bias if self.bias is not None else h.dten(self.out_channels)), file=f)

    def showNet(self, t=""):
        sz = list(self.prev)
        print(t + "Conv2D, filters={}, kernel_size={}, input_shape={}, stride={}, padding={}".format(self.out_channels,
                                                                                                     [self.kernel_size,
                                                                                                      self.kernel_size],
                                                                                                     list(reversed(sz)),
                                                                                                     [self.stride,
                                                                                                      self.stride],
                                                                                                     self.padding))

    def neuronCount(self):
        return 0


class ConvTranspose2D(InferModule):

    def init(self, in_shape, out_channels, kernel_size, stride=1, global_args=None, bias=True, padding=0, out_padding=0,
             activation="ReLU", **kargs):
        self.prev = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = out_padding
        self.activation = activation
        self.use_softplus = h.default(global_args, 'use_softplus', False)

        weights_shape = (self.in_channels, self.out_channels, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(torch.Tensor(*weights_shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(weights_shape[0]))
        else:
            self.bias = None  # h.zeros(weights_shape[0])

        outshape = getShapeConvTranspose(in_shape, (out_channels, kernel_size, kernel_size), stride, padding,
                                         out_padding)
        return outshape

    def forward(self, input, **kargs):
        return input.conv_transpose2d(self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                      output_padding=self.out_padding)

    def printNet(self, f):  # only complete if we've forwardt stride=1
        print("ConvTranspose2D", file=f)
        print(self.activation + ", filters={}, kernel_size={}, input_shape={}".format(self.out_channels,
                                                                                      list(self.kernel_size),
                                                                                      list(self.prev)), file=f)
        print(h.printListsNumpy([[list(p) for p in l] for l in self.weight.permute(2, 3, 1, 0).data]), file=f)
        print(h.printNumpy(self.bias), file=f)

    def neuronCount(self):
        return 0


class MaxPool2D(InferModule):
    def init(self, in_shape, kernel_size, stride=None, **kargs):
        self.prev = in_shape
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        return getShapeConv(in_shape, (in_shape[0], kernel_size, kernel_size), stride)

    def forward(self, x, **kargs):
        return x.max_pool2d(self.kernel_size, self.stride)

    def printNet(self, f):
        print("MaxPool2D stride={}, kernel_size={}, input_shape={}".format(list(self.stride), list(self.shape[2:]),
                                                                           list(self.prev[1:] + self.prev[:1])), file=f)

    def neuronCount(self):
        return h.product(self.outShape)


class AvgPool2D(InferModule):
    def init(self, in_shape, kernel_size, stride=None, **kargs):
        self.prev = in_shape
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        out_size = getShapeConv(in_shape, (in_shape[0], kernel_size, kernel_size), self.stride, padding=1)
        return out_size

    def forward(self, x, **kargs):
        if h.product(x.size()[2:]) == 1:
            return x
        return x.avg_pool2d(kernel_size=self.kernel_size, stride=self.stride, padding=1)

    def printNet(self, f):
        print("AvgPool2D stride={}, kernel_size={}, input_shape={}".format(list(self.stride), list(self.shape[2:]),
                                                                           list(self.prev[1:] + self.prev[:1])), file=f)

    def neuronCount(self):
        return h.product(self.outShape)


class AdaptiveAvgPool2D(InferModule):
    def init(self, in_shape, out_shape, **kargs):
        self.prev = in_shape
        self.out_shape = list(out_shape)
        return [in_shape[0]] + self.out_shape

    def forward(self, x, **kargs):
        return x.adaptive_avg_pool2d(self.out_shape)

    def printNet(self, f):
        print("AdaptiveAvgPool2D out_Shape={} input_shape={}".format(list(self.out_shape),
                                                                     list(self.prev[1:] + self.prev[:1])), file=f)

    def neuronCount(self):
        return h.product(self.outShape)


class Normalize(InferModule):
    def init(self, in_shape, mean, std, **kargs):
        self.mean_v = mean
        self.std_v = std
        self.mean = h.dten(mean)
        self.std = 1 / h.dten(std)
        return in_shape

    def forward(self, x, **kargs):
        mean_ex = self.mean.view(self.mean.shape[0], 1, 1).expand(*x.size()[1:])
        std_ex = self.std.view(self.std.shape[0], 1, 1).expand(*x.size()[1:])
        return (x - mean_ex) * std_ex

    def neuronCount(self):
        return 0

    def printNet(self, f):
        print("Normalize mean={} std={}".format(self.mean_v, self.std_v), file=f)

    def showNet(self, t=""):
        print(t + "Normalize mean={} std={}".format(self.mean_v, self.std_v))


class Flatten(InferModule):
    def init(self, in_shape, **kargs):
        return h.product(in_shape)

    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], h.product(s[1:]))

    def neuronCount(self):
        return 0


class BatchNorm(InferModule):
    def init(self, in_shape, track_running_stats=True, momentum=0.1, eps=1e-5, **kargs):
        self.gamma = torch.nn.Parameter(torch.Tensor(*in_shape))
        self.beta = torch.nn.Parameter(torch.Tensor(*in_shape))
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.num_batches_tracked = 0
        return in_shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x, **kargs):
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        new_mean = x.vanillaTensorPart().detach().mean(dim=0)
        new_var = x.vanillaTensorPart().detach().var(dim=0, unbiased=False)
        if torch.isnan(new_var * 0).any():
            return x
        if self.training:
            self.running_mean = (
                                        1 - exponential_average_factor) * self.running_mean + exponential_average_factor * new_mean if self.running_mean is not None else new_mean
            if self.running_var is None:
                self.running_var = new_var
            else:
                q = (1 - exponential_average_factor) * self.running_var
                r = exponential_average_factor * new_var
                self.running_var = q + r

        if self.track_running_stats and self.running_mean is not None and self.running_var is not None:
            new_mean = self.running_mean
            new_var = self.running_var

        diver = 1 / (new_var + self.eps).sqrt()

        if torch.isnan(diver).any():
            print("Really shouldn't happen ever")
            return x
        else:
            out = (x - new_mean) * diver * self.gamma + self.beta
            return out

    def neuronCount(self):
        return 0


class Unflatten2d(InferModule):
    def init(self, in_shape, w, **kargs):
        self.w = w
        self.outChan = int(h.product(in_shape) / (w * w))

        return (self.outChan, self.w, self.w)

    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], self.outChan, self.w, self.w)

    def neuronCount(self):
        return 0


class View(InferModule):
    def init(self, in_shape, out_shape, **kargs):
        assert (h.product(in_shape) == h.product(out_shape))
        return out_shape

    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], *self.outShape)

    def neuronCount(self):
        return 0


class Seq(InferModule):
    def init(self, in_shape, *layers, **kargs):
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.prev = in_shape
        for s in layers:
            in_shape = s.infer(in_shape, **kargs).outShape
        return in_shape

    def forward(self, x, **kargs):

        for l in self.layers:
            x = l(x, **kargs)
        return x

    def clip_norm(self):
        for l in self.layers:
            l.clip_norm()

    def regularize(self, p):
        return sum(n.regularize(p) for n in self.layers)

    def remove_norm(self):
        for l in self.layers:
            l.remove_norm()

    def printNet(self, f):
        for l in self.layers:
            l.printNet(f)

    def showNet(self, *args, **kargs):
        for l in self.layers:
            l.showNet(*args, **kargs)

    def neuronCount(self):
        return sum([l.neuronCount() for l in self.layers])

    def depth(self):
        return sum([l.depth() for l in self.layers])


def FFNN(layers, last_lin=False, last_zono=False, **kargs):
    starts = layers
    ends = []
    if last_lin:
        ends = ([CorrelateAll(only_train=False)] if last_zono else []) + [PrintActivation(activation="Affine"),
                                                                          Linear(layers[-1], **kargs)]
        starts = layers[:-1]

    return Seq(*([Seq(PrintActivation(**kargs), Linear(s, **kargs), activation(**kargs)) for s in starts] + ends))


def Conv(*args, **kargs):
    return Seq(Conv2D(*args, **kargs), activation(**kargs))


def ConvTranspose(*args, **kargs):
    return Seq(ConvTranspose2D(*args, **kargs), activation(**kargs))


MP = MaxPool2D


def LeNet(conv_layers, ly=[], bias=True, normal=False, **kargs):
    def transfer(tp):
        if isinstance(tp, InferModule):
            return tp
        if isinstance(tp[0], str):
            return MaxPool2D(*tp[1:])
        return Conv(out_channels=tp[0], kernel_size=tp[1], stride=tp[-1] if len(tp) == 4 else 1, bias=bias,
                    normal=normal, **kargs)

    conv = [transfer(s) for s in conv_layers]
    return Seq(*conv, FFNN(ly, **kargs, bias=bias)) if len(ly) > 0 else Seq(*conv)


def InvLeNet(ly, w, conv_layers, bias=True, normal=False, **kargs):
    def transfer(tp):
        return ConvTranspose(out_channels=tp[0], kernel_size=tp[1], stride=tp[2], padding=tp[3], out_padding=tp[4],
                             bias=False, normal=normal)

    return Seq(FFNN(ly, bias=bias), Unflatten2d(w), *[transfer(s) for s in conv_layers])


class FromByteImg(InferModule):
    def init(self, in_shape, **kargs):
        return in_shape

    def forward(self, x, **kargs):
        return x.to_dtype() / 256.

    def neuronCount(self):
        return 0


class Skip(InferModule):
    def init(self, in_shape, net1, net2, **kargs):
        self.net1 = net1.infer(in_shape, **kargs)
        self.net2 = net2.infer(in_shape, **kargs)
        assert (net1.outShape[1:] == net2.outShape[1:])
        return [net1.outShape[0] + net2.outShape[0]] + net1.outShape[1:]

    def forward(self, x, **kargs):
        r1 = self.net1(x, **kargs)
        r2 = self.net2(x, **kargs)
        return r1.cat(r2, dim=1)

    def regularize(self, p):
        return self.net1.regularize(p) + self.net2.regularize(p)

    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def remove_norm(self):
        self.net1.remove_norm()
        self.net2.remove_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()

    def printNet(self, f):
        print("SkipNet1", file=f)
        self.net1.printNet(f)
        print("SkipNet2", file=f)
        self.net2.printNet(f)
        print("SkipCat dim=1", file=f)

    def showNet(self, t=""):
        print(t + "SkipNet1")
        self.net1.showNet("    " + t)
        print(t + "SkipNet2")
        self.net2.showNet("    " + t)
        print(t + "SkipCat dim=1")


class ParSum(InferModule):
    def init(self, in_shape, net1, net2, **kargs):
        self.net1 = net1.infer(in_shape, **kargs)
        self.net2 = net2.infer(in_shape, **kargs)
        assert (net1.outShape == net2.outShape)
        return net1.outShape

    def forward(self, x, **kargs):
        r1 = self.net1(x, **kargs)
        r2 = self.net2(x, **kargs)
        return x.addPar(r1, r2)

    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def remove_norm(self):
        self.net1.remove_norm()
        self.net2.remove_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()

    def depth(self):
        return max(self.net1.depth(), self.net2.depth())

    def printNet(self, f):
        print("ParNet1", file=f)
        self.net1.printNet(f)
        print("ParNet2", file=f)
        self.net2.printNet(f)
        print("ParCat dim=1", file=f)

    def showNet(self, t=""):
        print(t + "ParNet1")
        self.net1.showNet("    " + t)
        print(t + "ParNet2")
        self.net2.showNet("    " + t)
        print(t + "ParSum")


class ReduceToZono(InferModule):
    def init(self, in_shape, max_length, customRelu=None, only_train=False, **kargs):
        self.all_possible_sub = max_length * 2 - 2 + 1
        self.customRelu = customRelu
        self.only_train = only_train
        self.in_shape = in_shape
        return in_shape

    def forward(self, x, **kargs):
        num_e = h.product(x.size())
        view_num = self.all_possible_sub * h.product(self.in_shape)
        if num_e >= view_num and num_e % view_num == 0:  # convert to Box (HybirdZonotope)
            x = x.view(-1, self.all_possible_sub, *self.in_shape)
            lower = x.min(1)[0]
            # print(lower.size())
            upper = x.max(1)[0]
            return ai.HybridZonotope((lower + upper) / 2, (upper - lower) / 2, None)
        else:  # if it is in Point() shape
            return x

    def neuronCount(self):
        return 0

    # def abstract_forward(self, x, **kargs):
    #     return x.abstractApplyLeaf('hybrid_to_zono', customRelu=self.customRelu)

    def showNet(self, t=""):
        print(t + self.__class__.__name__ + " only_train=" + str(self.only_train))


class ToZono(Identity):
    def init(self, in_shape, customRelu=None, only_train=False, **kargs):
        self.customRelu = customRelu
        self.only_train = only_train
        return in_shape

    def forward(self, x, **kargs):
        return self.abstract_forward(x, **kargs) if self.training or not self.only_train else x

    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('hybrid_to_zono', customRelu=self.customRelu)

    def showNet(self, t=""):
        print(t + self.__class__.__name__ + " only_train=" + str(self.only_train))


class CorrelateAll(ToZono):
    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('hybrid_to_zono', correlate=True, customRelu=self.customRelu)


class ToHZono(ToZono):
    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('zono_to_hybrid', customRelu=self.customRelu)


class Concretize(ToZono):
    def init(self, in_shape, only_train=True, **kargs):
        self.only_train = only_train
        return in_shape

    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('concretize')


# stochastic correlation
class CorrRand(Concretize):
    def init(self, in_shape, num_correlate, only_train=True, **kargs):
        self.only_train = only_train
        self.num_correlate = num_correlate
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf("stochasticCorrelate", self.num_correlate)

    def showNet(self, t=""):
        print(t + self.__class__.__name__ + " only_train=" + str(self.only_train) + " num_correlate=" + str(
            self.num_correlate))


class CorrMaxK(CorrRand):
    def abstract_forward(self, x):
        return x.abstractApplyLeaf("correlateMaxK", self.num_correlate)


class CorrMaxPool2D(Concretize):
    def init(self, in_shape, kernel_size, only_train=True, max_type=ai.MaxTypes.head_beta, **kargs):
        self.only_train = only_train
        self.kernel_size = kernel_size
        self.max_type = max_type
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf("correlateMaxPool", kernel_size=self.kernel_size, stride=self.kernel_size,
                                   max_type=self.max_type)

    def showNet(self, t=""):
        print(t + self.__class__.__name__ + " only_train=" + str(self.only_train) + " kernel_size=" + str(
            self.kernel_size) + " max_type=" + str(self.max_type))


class CorrMaxPool3D(Concretize):
    def init(self, in_shape, kernel_size, only_train=True, max_type=ai.MaxTypes.only_beta, **kargs):
        self.only_train = only_train
        self.kernel_size = kernel_size
        self.max_type = max_type
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf("correlateMaxPool", kernel_size=self.kernel_size, stride=self.kernel_size,
                                   max_type=self.max_type, max_pool=F.max_pool3d)

    def showNet(self, t=""):
        print(t + self.__class__.__name__ + " only_train=" + str(self.only_train) + " kernel_size=" + str(
            self.kernel_size) + " max_type=" + self.max_type)


class CorrFix(Concretize):
    def init(self, in_shape, k, only_train=True, **kargs):
        self.k = k
        self.only_train = only_train
        return in_shape

    def abstract_forward(self, x):
        sz = x.size()
        """
        # for more control in the future
        indxs_1 = torch.arange(start = 0, end = sz[1], step = math.ceil(sz[1] / self.dims[1]) )
        indxs_2 = torch.arange(start = 0, end = sz[2], step = math.ceil(sz[2] / self.dims[2]) )
        indxs_3 = torch.arange(start = 0, end = sz[3], step = math.ceil(sz[3] / self.dims[3]) )

        indxs = torch.stack(torch.meshgrid((indxs_1,indxs_2,indxs_3)), dim=3).view(-1,3)
        """
        szm = h.product(sz[1:])
        indxs = torch.arange(start=0, end=szm, step=math.ceil(szm / self.k))
        indxs = indxs.unsqueeze(0).expand(sz[0], indxs.size()[0])

        return x.abstractApplyLeaf("correlate", indxs)

    def showNet(self, t=""):
        print(t + self.__class__.__name__ + " only_train=" + str(self.only_train) + " k=" + str(self.k))


class DecorrRand(Concretize):
    def init(self, in_shape, num_decorrelate, only_train=True, **kargs):
        self.only_train = only_train
        self.num_decorrelate = num_decorrelate
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf("stochasticDecorrelate", self.num_decorrelate)


class DecorrMin(Concretize):
    def init(self, in_shape, num_decorrelate, only_train=True, num_to_keep=False, **kargs):
        self.only_train = only_train
        self.num_decorrelate = num_decorrelate
        self.num_to_keep = num_to_keep
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf("decorrelateMin", self.num_decorrelate, num_to_keep=self.num_to_keep)

    def showNet(self, t=""):
        print(t + self.__class__.__name__ + " only_train=" + str(self.only_train) + " k=" + str(
            self.num_decorrelate) + " num_to_keep=" + str(self.num_to_keep))


class DeepLoss(ToZono):
    def init(self, in_shape, bw=0.01, act=F.relu, **kargs):  # weight must be between 0 and 1
        self.only_train = True
        self.bw = S.Const.initConst(bw)
        self.act = act
        return in_shape

    def abstract_forward(self, x, **kargs):
        if x.isPoint():
            return x
        return ai.TaggedDomain(x, self.MLoss(self, x))

    class MLoss():
        def __init__(self, obj, x):
            self.obj = obj
            self.x = x

        def loss(self, a, *args, lr=1, time=0, **kargs):
            bw = self.obj.bw.getVal(time=time)
            pre_loss = a.loss(*args, time=time, **kargs, lr=lr * (1 - bw))
            if bw <= 0.0:
                return pre_loss
            return (1 - bw) * pre_loss + bw * self.x.deep_loss(act=self.obj.act)

    def showNet(self, t=""):
        print(
            t + self.__class__.__name__ + " only_train=" + str(self.only_train) + " bw=" + str(self.bw) + " act=" + str(
                self.act))


class IdentLoss(DeepLoss):
    def abstract_forward(self, x, **kargs):
        return x


def SkipNet(net1, net2, ffnn, **kargs):
    return Seq(Skip(net1, net2), FFNN(ffnn, **kargs))


def WideBlock(out_filters, downsample=False, k=3, bias=False, **kargs):
    if not downsample:
        k_first = 3
        skip_stride = 1
        k_skip = 1
    else:
        k_first = 4
        skip_stride = 2
        k_skip = 2

    # conv2d280(input)
    blockA = Conv2D(out_filters, kernel_size=k_skip, stride=skip_stride, padding=0, bias=bias, normal=True, **kargs)

    # conv2d282(relu(conv2d278(input)))
    blockB = Seq(Conv(out_filters, kernel_size=k_first, stride=skip_stride, padding=1, bias=bias, normal=True, **kargs)
                 , Conv2D(out_filters, kernel_size=k, stride=1, padding=1, bias=bias, normal=True, **kargs))
    return Seq(ParSum(blockA, blockB), activation(**kargs))


def BasicBlock(in_planes, planes, stride=1, bias=False, skip_net=False, **kargs):
    block = Seq(Conv(planes, kernel_size=3, stride=stride, padding=1, bias=bias, normal=True, **kargs)
                , Conv2D(planes, kernel_size=3, stride=1, padding=1, bias=bias, normal=True, **kargs))

    if stride != 1 or in_planes != planes:
        block = ParSum(block, Conv2D(planes, kernel_size=1, stride=stride, bias=bias, normal=True, **kargs))
    elif not skip_net:
        block = ParSum(block, Identity())
    return Seq(block, activation(**kargs))


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
def ResNet(blocksList, extra=[], bias=False, **kargs):
    layers = []
    in_planes = 64
    planes = 64
    stride = 0
    for num_blocks in blocksList:
        if stride < 2:
            stride += 1

        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride, bias=bias, **kargs))
            in_planes = planes
        planes *= 2

    print("RESlayers: ", len(layers))
    for e, l in extra:
        layers[l] = Seq(layers[l], e)

    return Seq(Conv(64, kernel_size=3, stride=1, padding=1, bias=bias, normal=True, printShape=True),
               *layers)


def DenseNet(growthRate, depth, reduction, num_classes, bottleneck=True):
    def Bottleneck(growthRate):
        interChannels = 4 * growthRate

        n = Seq(ReLU(),
                Conv2D(interChannels, kernel_size=1, bias=True, ibp_init=True),
                ReLU(),
                Conv2D(growthRate, kernel_size=3, padding=1, bias=True, ibp_init=True)
                )

        return Skip(Identity(), n)

    def SingleLayer(growthRate):
        n = Seq(ReLU(),
                Conv2D(growthRate, kernel_size=3, padding=1, bias=True, ibp_init=True))
        return Skip(Identity(), n)

    def Transition(nOutChannels):
        return Seq(ReLU(),
                   Conv2D(nOutChannels, kernel_size=1, bias=True, ibp_init=True),
                   AvgPool2D(kernel_size=2))

    def make_dense(growthRate, nDenseBlocks, bottleneck):
        return Seq(*[Bottleneck(growthRate) if bottleneck else SingleLayer(growthRate) for i in range(nDenseBlocks)])

    nDenseBlocks = (depth - 4) // 3
    if bottleneck:
        nDenseBlocks //= 2

    nChannels = 2 * growthRate
    conv1 = Conv2D(nChannels, kernel_size=3, padding=1, bias=True, ibp_init=True)
    dense1 = make_dense(growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks * growthRate
    nOutChannels = int(math.floor(nChannels * reduction))
    trans1 = Transition(nOutChannels)

    nChannels = nOutChannels
    dense2 = make_dense(growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks * growthRate
    nOutChannels = int(math.floor(nChannels * reduction))
    trans2 = Transition(nOutChannels)

    nChannels = nOutChannels
    dense3 = make_dense(growthRate, nDenseBlocks, bottleneck)

    return Seq(conv1, dense1, trans1, dense2, trans2, dense3,
               ReLU(),
               AvgPool2D(kernel_size=8),
               CorrelateAll(only_train=False, ignore_point=True),
               Linear(num_classes, ibp_init=True))
