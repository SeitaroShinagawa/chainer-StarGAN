
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable,cuda
import numpy as np

# differentiable backward functions

def backward_linear(x_in, x, l):
    y = F.matmul(x, l.W)
    return y

def backward_convolution(x_in, x, l):
    y = F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))
    return y

def backward_deconvolution(x_in, x, l):
    y = F.convolution_2d(x, l.W, None, l.stride, l.pad)
    return y

def backward_relu(x_in, x):
    y = (x_in.data > 0) * x
    return y

def backward_leaky_relu(x_in, x, a):
    y = (x_in.data > 0) * x + a * (x_in.data < 0) * x
    return y

def backward_sigmoid(x_in, g):
    y = F.sigmoid(x_in)
    return g * y * (1 - y)


def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

class ResBlock(chainer.Chain):
    def __init__(self, ch, bn=True, activation=F.relu):
        self.bn = bn
        self.activation = activation
        layers = {}
        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, 1)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, 1)
        if bn:
            layers['bn0'] = L.BatchNormalization(ch)
            layers['bn1'] = L.BatchNormalization(ch)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x):
        h = self.c0(x)
        if self.bn:
            h = self.bn0(h)
        h = self.activation(h)
        h = self.c1(h)
        if self.bn:
            h = self.bn1(h)
        return h + x


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False, noise=False, slope=None):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        self.slope = slope
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample=='none-9':
            layers['c'] = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w)
        elif sample=='none-7':
            layers['c'] = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w)
        elif sample=='none-5':
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
        else:
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
        if bn:
            if self.noise:
                layers['batchnorm'] = L.BatchNormalization(ch1, use_gamma=False)
            else:
                layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        if self.sample=="down" or self.sample=="none" or self.sample=='none-9' or self.sample=='none-7' or self.sample=='none-5':
            h = self.c(x)
        elif self.sample=="up":
            #h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = F.unpooling_2d(x, 4, 2, 1, cover_all=False)
            h = self.c(h)
        else:
            print("unknown sample method %s"%self.sample)
        if self.bn:
            h = self.batchnorm(h)
        if self.noise:
            h = add_noise(h, test=self.test)
        if self.dropout:
            h = F.dropout(h) #, train=not self.test)
        if not self.slope is None:
            h = F.leaky_relu(h, slope=self.slope)
        elif not self.activation is None:
            h = self.activation(h)
        return h


class Generator_ResBlock_6(chainer.Chain):
    def __init__(self):
        super(Generator_ResBlock_6, self).__init__(
            c1 = CBR(3, 32, bn=True, sample='none-7'),
            c2 = CBR(32, 64, bn=True, sample='down'),
            c3 = CBR(64, 128, bn=True, sample='down'),
            c4 = ResBlock(128, bn=True),
            c5 = ResBlock(128, bn=True),
            c6 = ResBlock(128, bn=True),
            c7 = ResBlock(128, bn=True),
            c8 = ResBlock(128, bn=True),
            c9 = ResBlock(128, bn=True),
            c10 = CBR(128, 64, bn=True, sample='up'),
            c11 = CBR(64, 32, bn=True, sample='up'),
            c12 = CBR(32, 3, bn=True, sample='none-7', activation=F.tanh)
        )

    def __call__(self, x, test=False, volatile=False):
        h = self.c1(x, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        h = self.c5(h, test=test)
        h = self.c6(h, test=test)
        h = self.c7(h, test=test)
        h = self.c8(h, test=test)
        h = self.c9(h, test=test)
        h = self.c10(h, test=test)
        h = self.c11(h, test=test)
        h = self.c12(h, test=test)
        return h

class Generator_ResBlock_9(chainer.Chain):
    def __init__(self, img_size):
        super(Generator_ResBlock_9, self).__init__(
            c1 = CBR(3, 32, bn=True, sample='none-7'),
            c2 = CBR(32+18, 64, bn=True, sample='down'),
            c3 = CBR(64, 128, bn=True, sample='down'),
            c4 = ResBlock(128, bn=True),
            c5 = ResBlock(128, bn=True),
            c6 = ResBlock(128, bn=True),
            c7 = ResBlock(128, bn=True),
            c8 = ResBlock(128, bn=True),
            c9 = ResBlock(128, bn=True),
            c10 = ResBlock(128, bn=True),
            c11 = ResBlock(128, bn=True),
            c12 = ResBlock(128, bn=True),
            c13 = CBR(128, 64, bn=True, sample='up'),
            c14 = CBR(64, 32, bn=True, sample='up'),
            c15 = CBR(32, 3, bn=True, sample='none-7', activation=F.tanh)
        )

    def __call__(self, x, att, test=False, volatile=False):
        B,ch,H,W = x.shape
        B,A = att.shape
        h = self.c1(x, test=test)
        Att = np.ones((B,A,H,W)).astype("f")
        for i in range(B):
            tmp = att[i]
            for j in range(A):
                Att[i][j] = tmp[j]*np.ones((H,W)).astype("f")#Att[i][j]
        Att = chainer.Variable(self.xp.asarray(Att),volatile=volatile)
        h = F.concat([h,Att],axis=1)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        h = self.c5(h, test=test)
        h = self.c6(h, test=test) #c6: (1, 128, 16, 16), c7: (1, 128, 16, 16)
        h = self.c7(h, test=test)
        h = self.c8(h, test=test)
        h = self.c9(h, test=test)
        h = self.c10(h, test=test)
        h = self.c11(h, test=test)
        h = self.c12(h, test=test)
        h = self.c13(h, test=test)
        h = self.c14(h, test=test)
        h = self.c15(h, test=test)
        return h


class StarGAN_Generator(chainer.Chain):
    def __init__(self, img_size, nc_size):
        self.nc_size = nc_size
        super(StarGAN_Generator, self).__init__(
            c1 = CBR(3+nc_size, 64, bn=True, sample='none-7'),
            c2 = CBR(64, 128, bn=True, sample='down'),
            c3 = CBR(128, 256, bn=True, sample='down'),
            c4 = ResBlock(256, bn=True),
            c5 = ResBlock(256, bn=True),
            c6 = ResBlock(256, bn=True),
            c7 = ResBlock(256, bn=True),
            c8 = ResBlock(256, bn=True),
            c9 = ResBlock(256, bn=True),
            c10 = CBR(256, 128, bn=True, sample='up'),
            c11 = CBR(128, 64, bn=True, sample='up'),
            c12 = CBR(64, 3, bn=True, sample='none-7', activation=F.tanh)
        )

    def __call__(self, x, att):
        B,ch,H,W = x.shape
        B,A = att.shape
        
        Attmap = self.xp.broadcast_to(att,(H,W,B,A))
        Attmap = chainer.Variable(self.xp.transpose(Attmap,(2,3,0,1)))
        
        h = F.concat([x,Attmap],axis=1)
        
        for i in range(1, 13):
            h = getattr(self, 'c'+str(i))(h)
        return h


class StarGAN_Discriminator(chainer.Chain):
    def __init__(self, in_ch=3, n_down_layers=6, n_att=5, image_size=128):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        self.n_down_layers = n_down_layers
        self.n_att = n_att

        base = 64
        layers['c0'] = CBR(in_ch, base, bn=False, sample='down', activation=F.leaky_relu)
        
        for i in range(1,n_down_layers):
            layers['c'+str(i)] = CBR(base, base*2, bn=False, sample='down', activation=F.leaky_relu, slope=0.01)
            base *= 2

        super(StarGAN_Discriminator, self).__init__(**layers,
        out = L.Convolution2D(base, 1, 3, 1, 1), #PatchGAN (n_patch = base)
        out_cls = L.Convolution2D(base, n_att, int(image_size/2**n_down_layers), 1, 0),
        )

    def __call__(self, x):

        self.h_dict = {}
        self.h_dict["h0"] = x
        
        for i in range(self.n_down_layers):
            self.h_dict["h"+str(i+1)] = getattr(self, 'c'+str(i))(self.h_dict["h"+str(i)])

        self.h_dis = getattr(self, 'out')(self.h_dict["h"+str(self.n_down_layers)]) #PatchGAN (B,1,n_patch,n_patch)
        h_cls = F.leaky_relu(self.out_cls(self.h_dict["h"+str(self.n_down_layers)]),slope=0.01) # (B,n_att,1,1) 
        h_cls = F.reshape(h_cls,(x.shape[0],self.n_att)) 

        return self.h_dis, h_cls


    def differentiable_backward(self, x):
        g = backward_convolution(self.h_dict["h"+str(self.n_down_layers)], x, self.out)
        g = backward_leaky_relu(self.h_dict["h"+str(self.n_down_layers)], g, 0.01)
        
        for i in reversed(range(1,self.n_down_layers)):
            g = backward_convolution(self.h_dict["h"+str(i)], g, getattr(self, 'c'+str(i)).c)
            g = backward_leaky_relu(self.h_dict["h"+str(i)], g, 0.01)
        g = backward_convolution(self.h_dict["h0"], g, getattr(self, 'c0').c)
        return g

