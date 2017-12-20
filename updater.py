import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os
import random
from chainer import cuda, optimizers, serializers, Variable
from chainer import training

from PIL import Image

class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        params = kwargs.pop('params')
        self._lambda_adv = params['lambda_adv'] 
        self._lambda_cls = params['lambda_cls']
        self._lambda_rec = params['lambda_rec']
        self._lambda_gp = params['lambda_gp'] #gradient penalty of WGAN
        self._n_dis = params['n_dis']
        self._learning_rate_anneal_start = params['learning_rate_anneal_start']
        self._learning_rate_anneal = params['learning_rate_anneal']
        self._image_size = params['image_size']
        self._eval_foler = params['eval_folder']
        self._dataset = params['dataset']
        self._iter = 0
        self._max_buffer_size = 50
        self.filter_size = params['nc_size']
        xp = self.gen.xp
        self._buffer_y = xp.zeros((self._max_buffer_size , 3, self._image_size, self._image_size)).astype("f")
        self._buffer_yatt = xp.zeros((self._max_buffer_size , self.filter_size)).astype("i")
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.gen.xp
        opt_gen = self.get_optimizer('opt_gen')
        opt_dis = self.get_optimizer('opt_dis')
        w_in = self._image_size
        self._iter += 1
        
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        
        #learning rate annealing () 
        if self._learning_rate_anneal_start < self._iter:
            if opt_gen.alpha > 0:
                opt_gen.alpha -= self._learning_rate_anneal
            if opt_dis.alpha > 0:
                opt_dis.alpha -= self._learning_rate_anneal
        chainer.reporter.report({'lr_g': opt_gen.alpha})
        chainer.reporter.report({'lr_d': opt_dis.alpha})
        
        x = xp.zeros((batchsize, 3, w_in, w_in)).astype("f")
        x_att = xp.zeros((batchsize, self.filter_size)).astype("i")
        y_att = xp.zeros((batchsize, self.filter_size)).astype("i")
        for j in range(batchsize):
            x[j, :] = xp.array(batch[j][0])
            x_att[j] = xp.array(batch[j][1])
            y_att[j] = xp.array(batch[j][2])
        
        x_att_in = x_att.astype("f")
        y_att_in = y_att.astype("f")
        
        #real
        x_real = Variable(x)

        #generate fake
        y_fake = self.gen(x_real,y_att_in) 
        x_fake = self.gen(y_fake,x_att_in)

        #pass Discriminator
        out_real, out_real_att = self.dis(x_real)
        out_fake, out_fake_att = self.dis(y_fake)
        
        #Generator learning
        if self._iter % self._n_dis == 0:
            loss_gen_adv = F.sum(-out_fake) / batchsize  #adversarial loss
            loss_gen_cls = F.sigmoid_cross_entropy(out_fake_att,Variable(y_att)) #classification loss, CelebA requires sigmoid_cross_entropy
            loss_gen_rec = F.mean_absolute_error(x_real,x_fake) #reconstruction loss
            
            loss_gen = self._lambda_adv * loss_gen_adv + self._lambda_cls * loss_gen_cls + self._lambda_rec * loss_gen_rec
            
            self.gen.cleargrads()
            loss_gen.backward()
            opt_gen.update()
            chainer.reporter.report({'loss_gen_adv': loss_gen_adv})
            chainer.reporter.report({'loss_gen_cls': loss_gen_cls})
            chainer.reporter.report({'loss_gen_rec': loss_gen_rec})

        y_fake.unchain_backward()
        x_fake.unchain_backward()

        #Discriminator learning
        loss_dis_adv =  F.sum(-out_real) / batchsize #adversarial loss 
        loss_dis_adv += F.sum(out_fake) / batchsize
        loss_dis_cls = F.sigmoid_cross_entropy(out_real_att,Variable(x_att)) #classification loss, CelebA requires
        
        loss_dis = self._lambda_adv * loss_dis_adv + self._lambda_cls * loss_dis_cls 

        self.dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        #wassersteinGAN
        eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
        x_mid = eps * x_real + (1.0 - eps) * y_fake
        x_mid_v = Variable(x_mid.data)
        y_mid, _ = self.dis(x_mid_v)
        y_mid = F.sum(y_mid)
        #dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))
        dydx, = chainer.grad([y_mid],[x_mid_v], enable_double_backprop=True)
        dydx = F.sqrt(F.sum(dydx * dydx, axis=(1, 2, 3)))
        loss_gp = F.mean_squared_error(dydx, xp.ones_like(dydx.data)) 

        loss_dis = self._lambda_gp * loss_gp

        self.dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        chainer.reporter.report({'loss_dis_adv': loss_dis_adv})
        chainer.reporter.report({'loss_dis_cls': loss_dis_cls})
        chainer.reporter.report({'loss_gp': loss_gp})
