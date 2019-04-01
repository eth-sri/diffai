import torch
import torch.nn as nn
import math

try:
    from . import helpers as h
except:
    import helpers as h



class Const():
    def __init__(self, c):
        self.c = c if c is None else float(c)

    def getVal(self, c = None, **kargs):
        return self.c if self.c is not None else c

    def __str__(self):
        return str(self.c)

    def initConst(x):
        return x if isinstance(x, Const) else Const(x)

class Lin(Const):
    def __init__(self, start, end, steps, initial = 0, quant = False):
        self.start = float(start)
        self.end = float(end)
        self.steps = float(steps)
        self.initial = float(initial)
        self.quant = quant

    def getVal(self, time = 0, **kargs):
        if self.quant:
            time = math.floor(time)
        return (self.end - self.start) * max(0,min(1, float(time - self.initial) / self.steps)) + self.start

    def __str__(self):
        return "Lin(%s,%s,%s,%s, quant=%s)".format(str(self.start), str(self.end), str(self.steps), str(self.initial), str(self.quant))

class Until(Const):
    def __init__(self, thresh, a, b):
        self.a = Const.initConst(a)
        self.b = Const.initConst(b)
        self.thresh = thresh

    def getVal(self, *args, time = 0, **kargs):
        return self.a.getVal(*args, time = time, **kargs) if time < self.thresh else self.b.getVal(*args, time = time - self.thresh, **kargs)

    def __str__(self):
        return "Until(%s, %s, %s)" % (str(self.thresh), str(self.a), str(self.b))

class Scale(Const): # use with mix when aw = 1, and 0 <= c < 1
    def __init__(self, c):
        self.c = Const.initConst(c)

    def getVal(self, *args, **kargs):
        c = self.c.getVal(*args, **kargs)
        if c == 0:
            return 0
        assert c >= 0
        assert c < 1
        return c / (1 - c)

    def __str__(self):
        return "Scale(%s)" % str(self.c)

def MixLin(*args, **kargs):
    return Scale(Lin(*args, **kargs))

class Normal(Const):
    def __init__(self, c):
        self.c = Const.initConst(c)

    def getVal(self, *args, shape = [1], **kargs):
        c = self.c.getVal(*args, shape = shape, **kargs)
        return torch.randn(shape, device = h.device).abs() * c

    def __str__(self):
        return "Normal(%s)" % str(self.c)

class Clip(Const):
    def __init__(self, c, l, u):
        self.c = Const.initConst(c)
        self.l = Const.initConst(l)
        self.u = Const.initConst(u)

    def getVal(self, *args, **kargs):
        c = self.c.getVal(*args, **kargs)
        l = self.l.getVal(*args, **kargs)
        u = self.u.getVal(*args, **kargs)
        if isinstance(c, float):
            return min(max(c,l),u)
        else:
            return c.clamp(l,u)

    def __str__(self):
        return "Clip(%s, %s, %s)" % (str(self.c), str(self.l), str(self.u))

class Fun(Const):
    def __init__(self, foo):
        self.foo = foo
    def getVal(self, *args, **kargs):
        return self.foo(*args, **kargs)
    
    def __str__(self):
        return "Fun(...)"

class Complement(Const): # use with mix when aw = 1, and 0 <= c < 1
    def __init__(self, c):
        self.c = Const.initConst(c)

    def getVal(self, *args, **kargs):
        c = self.c.getVal(*args, **kargs)
        assert c >= 0
        assert c <= 1
        return 1 - c

    def __str__(self):
        return "Complement(%s)" % str(self.c)
