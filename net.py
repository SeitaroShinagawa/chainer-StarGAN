
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable,cuda
import numpy as np
import math
from instance_normalization import InstanceNormalization

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
    def __init__(self, ch, norm="instance", activation=F.relu, noise=False):
        self.use_norm = False if norm is None else True 
        self.activation = activation
        layers = {}
        w = chainer.initializers.Uniform(scale=math.sqrt(1/ch/3/3)) #same to pytorch conv2d initializaiton
        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True)
        if norm=="batch":
            layers['norm0'] = L.BatchNormalization(ch, use_gamma=noise, use_beta=noise)
            layers['norm1'] = L.BatchNormalization(ch, use_gamma=noise, use_beta=noise)
        elif norm=="instance":
            layers['norm0'] = InstanceNormalization(ch, use_gamma=noise, use_beta=noise)
            layers['norm1'] = InstanceNormalization(ch, use_gamma=noise, use_beta=noise)

        super(ResBlock, self).__init__(**layers)

    def __call__(self, x):
        h = self.c0(x)
        if self.use_norm:
            h = self.norm0(h)
        h = self.activation(h)
        h = self.c1(h)
        if self.use_norm:
            h = self.norm1(h)
        return h + x


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, norm='instance', sample='down', activation=F.relu, dropout=False, noise=False, slope=None):
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        self.slope = slope
        layers = {}
        #w = chainer.initializers.Normal(0.02)

        self.use_norm = False if norm is None else True

        if sample=='down':
            w = chainer.initializers.Uniform(scale=math.sqrt(1/ch0/4/4)) #same to pytorch conv2d initialization
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w, nobias=True)
        elif sample=='none-9':
            w = chainer.initializers.Uniform(scale=math.sqrt(1/ch0/9/9))
            layers['c'] = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w, nobias=True)
        elif sample=='none-7':
            w = chainer.initializers.Uniform(scale=math.sqrt(1/ch0/7/7))
            layers['c'] = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w, nobias=True)
        elif sample=='none-5':
            w = chainer.initializers.Uniform(scale=math.sqrt(1/ch0/5/5))
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w, nobias=True)
        else:
            w = chainer.initializers.Uniform(scale=math.sqrt(1/ch0/3/3))
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w, nobias=True)
        if norm=="batch":
            if self.noise:
                layers['norm'] = L.BatchNormalization(ch1, use_gamma=True, use_beta=True)
            else:
                layers['norm'] = L.BatchNormalization(ch1, use_gamma=False, use_beta=False)
        elif norm=="instance":
            if self.noise:
                layers['norm'] = InstanceNormalization(ch1, use_gamma=True, use_beta=True)
            else:
                layers['norm'] = InstanceNormalization(ch1, use_gamma=False, use_beta=False)


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
        if self.use_norm:
            h = self.norm(h)
        #if self.noise:
        #    h = add_noise(h, test=self.test)
        if self.dropout:
            h = F.dropout(h) #, train=not self.test)
        if not self.slope is None:
            h = F.leaky_relu(h, slope=self.slope)
        elif not self.activation is None:
            h = self.activation(h)
        return h


class StarGAN_Generator(chainer.Chain):
    def __init__(self, img_size, nc_size):
        self.nc_size = nc_size
        super(StarGAN_Generator, self).__init__(
            c1 = CBR(3+nc_size, 64, norm="instance", sample='none-7',noise=True),
            c2 = CBR(64, 128, norm="instance", sample='down',noise=True),
            c3 = CBR(128, 256, norm="instance", sample='down',noise=True),
            c4 = ResBlock(256, norm="instance",noise=True),
            c5 = ResBlock(256, norm="instance",noise=True),
            c6 = ResBlock(256, norm="instance",noise=True),
            c7 = ResBlock(256, norm="instance",noise=True),
            c8 = ResBlock(256, norm="instance",noise=True),
            c9 = ResBlock(256, norm="instance",noise=True),
            c10 = CBR(256, 128, norm="instance", sample='up',noise=True),
            c11 = CBR(128, 64, norm="instance", sample='up', noise=True),
            c12 = CBR(64, 3, norm=None, sample='none-7', activation=F.tanh)
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
    def __init__(self, in_ch=3, n_down_layers=6, n_att=5):
        layers = {}
        self.n_down_layers = n_down_layers
        self.image_size = 2**(n_down_layers+1)
        self.n_att = n_att

        base = 64
        layers['c0'] = CBR(in_ch, base, norm=None, sample='down', activation=F.leaky_relu, slope=0.01)
        
        for i in range(1,n_down_layers):
            layers['c'+str(i)] = CBR(base, base*2, norm=None, sample='down', activation=F.leaky_relu, slope=0.01)
            base *= 2

        cls_ksize = int(self.image_size/2**n_down_layers) 
        w_out = chainer.initializers.Uniform(scale=math.sqrt(1/base/3**2))
        w_cls = chainer.initializers.Uniform(scale=math.sqrt(1/base/cls_ksize**2))
        super(StarGAN_Discriminator, self).__init__(**layers,
        out = L.Convolution2D(base, 1, 3, 1, 1, nobias=True, initialW=w_out), #PatchGAN (n_patch = base)
        out_cls = L.Convolution2D(base, n_att, cls_ksize, 1, 0, nobias=True, initialW=w_cls),
        )

    def __call__(self, x):

        self.h_dict = {}
        self.h_dict["h0"] = x
        
        for i in range(self.n_down_layers):
            self.h_dict["h"+str(i+1)] = getattr(self, 'c'+str(i))(self.h_dict["h"+str(i)])

        self.h_dis = getattr(self, 'out')(self.h_dict["h"+str(self.n_down_layers)]) #PatchGAN (B,1,n_patch,n_patch)
        h_cls = self.out_cls(self.h_dict["h"+str(self.n_down_layers)]) # (B,n_att,1,1) 
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

