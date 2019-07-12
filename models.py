try:
    from . import components as n
    from . import ai
    from . import scheduling as S
except:
    import components as n
    import scheduling as S
    import ai

############# Previously Known Models.  Not guaranteed to have the same performance as previous papers.

def FFNN(c, **kargs):
    return n.FFNN([100, 100, 100, 100, 100,c], last_lin = True, last_zono = True, **kargs)

def ConvSmall(c, **kargs):
    return n.LeNet([ (16,4,4,2), (32,4,4,2) ], [100,c], last_lin = True, last_zono = True, **kargs)

def ConvMed(c, **kargs):
    return n.LeNet([ (16,4,4,2), (32,4,4,2) ], [100,c], padding = 1, last_lin = True, last_zono = True, **kargs)

def ConvBig(c, **kargs):
    return n.LeNet([ (32,3,3,1), (32,4,4,2) , (64,3,3,1), (64,4,4,2)], [512, 512,c], padding = 1, last_lin = True, last_zono = True, **kargs)

def ConvLargeIBP(c, **kargs):
    return n.LeNet([ (64, 3, 3, 1), (64,3,3,1), (128,3,3,2), (128,3,3,1), (128,3,3,1)], [200,c], padding=1, ibp_init = True, bias = True, last_lin = True, last_zono = True, **kargs)

def ResNetWong(c, **kargs):
    return n.Seq(n.Conv(16, 3, padding=1, bias=False), n.WideBlock(16), n.WideBlock(16), n.WideBlock(32, True), n.WideBlock(64, True), n.FFNN([1000, c], ibp_init = True, bias=True, last_lin=True, last_zono = True, **kargs))

def TruncatedVGG(c, **kargs):
    return n.LeNet([ (64, 3, 3, 1), (64,3,3,1), (128,3,3,2), (128,3,3,1)], [512,c], padding=1, ibp_init = True, bias = True, last_lin = True, last_zono = True, **kargs)


############# New Models

def ResNetTiny(c, **kargs): # resnetWide also used by mixtrain and scaling provable adversarial defenses
    def wb(c, bias = True, **kargs):
        return n.WideBlock(c, False, bias=bias, ibp_init=True, batch_norm = False, **kargs)
    return n.Seq(n.Conv(16, 3, padding=1, bias=True, ibp_init = True), 
                 wb(16), 
                 wb(32), 
                 wb(32), 
                 wb(32), 
                 wb(32), 
                 n.FFNN([500, c], bias=True, last_lin=True, ibp_init = True, last_zono = True, **kargs))

def ResNetTiny_FewCombo(c, **kargs): # resnetWide also used by mixtrain and scaling provable adversarial defenses
    def wb(c, bias = True, **kargs):
        return n.WideBlock(c, False, bias=bias, ibp_init=True, batch_norm = False, **kargs)
    dl = n.DeepLoss
    cmk = n.CorrMaxK
    cm2d = n.CorrMaxPool2D
    cm3d = n.CorrMaxPool3D
    dec = lambda x: n.DecorrMin(x, num_to_keep = True)
    return n.Seq(cmk(32), 
                 n.Conv(16, 3, padding=1, bias=True, ibp_init = True), dec(8), 
                 wb(16), dec(4), 
                 wb(32), n.Concretize(), 
                 wb(32), 
                 wb(32), 
                 wb(32), cmk(10), 
                 n.FFNN([500, c], bias=True, last_lin=True, ibp_init = True, last_zono = True, **kargs))


def ResNetTiny_ManyFixed(c, **kargs): # resnetWide also used by mixtrain and scaling provable adversarial defenses
    def wb(c, bias = True, **kargs):
        return n.WideBlock(c, False, bias=bias, ibp_init=True, batch_norm = False, **kargs)
    cmk = n.CorrFix
    dec = lambda x: n.DecorrMin(x, num_to_keep = True)
    return n.Seq(n.CorrMaxK(32), 
                 n.Conv(16, 3, padding=1, bias=True, ibp_init = True), cmk(16), dec(16), 
                 wb(16), cmk(8), dec(8), 
                 wb(32), cmk(8), dec(8), 
                 wb(32), cmk(4), dec(4), 
                 wb(32), n.Concretize(),
                 wb(32), 
                 n.FFNN([500, c], bias=True, last_lin=True, ibp_init = True, last_zono = True, **kargs))

def SkipNet18(c, **kargs):
    return n.Seq(n.ResNet([2,2,2,2], bias = True, ibp_init = True, skip_net = True), n.FFNN([512, 512, c], bias=True, last_lin=True, last_zono = True, ibp_init = True, **kargs))

def SkipNet18_Combo(c, **kargs):
    dl = n.DeepLoss
    cmk = n.CorrFix
    dec = lambda x: n.DecorrMin(x, num_to_keep = True)
    return n.Seq(n.ResNet([2,2,2,2], extra = [ (cmk(20),2),(dec(10),2)
                                              ,(cmk(10),3),(dec(5),3),(dl(S.Until(90, S.Lin(0, 0.2, 50, 40), 0)), 3)
                                              ,(cmk(5),4),(dec(2),4)], bias = True, ibp_init=True, skip_net = True), n.FFNN([512, 512, c], bias=True, last_lin=True, last_zono = True, ibp_init=True, **kargs))

def ResNet18(c, **kargs):
    return n.Seq(n.ResNet([2,2,2,2], bias = True, ibp_init = True), n.FFNN([512, 512, c], bias=True, last_lin=True, last_zono = True, ibp_init = True, **kargs))


def ResNetLarge_LargeCombo(c, **kargs): # resnetWide also used by mixtrain and scaling provable adversarial defenses
    def wb(c, bias = True, **kargs):
        return n.WideBlock(c, False, bias=bias, ibp_init=True, batch_norm = False, **kargs)
    dl = n.DeepLoss
    cmk = n.CorrMaxK
    cm2d = n.CorrMaxPool2D
    cm3d = n.CorrMaxPool3D
    dec = lambda x: n.DecorrMin(x, num_to_keep = True)
    return n.Seq(n.Conv(16, 3, padding=1, bias=True, ibp_init = True), cmk(4),
                 wb(16), cmk(4), dec(4),
                 wb(32), cmk(4), dec(4),
                 wb(32), dl(S.Until(1, 0, S.Lin(0.5, 0, 50, 3))), 
                 wb(32), cmk(4), dec(4),
                 wb(64), cmk(4), dec(2),
                 wb(64), dl(S.Until(24, S.Lin(0, 0.1, 20, 4), S.Lin(0.1, 0, 50))), 
                 wb(64), 
                 n.FFNN([1000, c], bias=True, last_lin=True, ibp_init = True, **kargs))



def ResNet34(c, **kargs):
    return n.Seq(n.ResNet([3,4,6,3], bias = True, ibp_init = True), n.FFNN([512, 512, c], bias=True, last_lin=True, last_zono = True, ibp_init = True, **kargs))


def DenseNet100(c, **kwargs):
    return n.DenseNet(growthRate=12, depth=100, reduction=0.5,
                      bottleneck=True, num_classes = c)
