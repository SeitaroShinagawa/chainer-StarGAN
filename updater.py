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
        self.filter_size = 5
        xp = self.gen.xp
        self._buffer_y = xp.zeros((self._max_buffer_size , 3, self._image_size, self._image_size)).astype("f")
        self._buffer_yatt = xp.zeros((self._max_buffer_size , self.filter_size)).astype("i")
        super(Updater, self).__init__(*args, **kwargs)

    def getAndUpdateBufferY(self, data, att): #optional 
        if  self._iter < self._max_buffer_size:
            self._buffer_y[self._iter, :] = data[0]
            self._buffer_yatt[self._iter, :] = att[0]
            return data, att

        self._buffer_y[0:self._max_buffer_size-2, :] = self._buffer_y[1:self._max_buffer_size-1, :]
        self._buffer_yatt[0:self._max_buffer_size-2, :] = self._buffer_yatt[1:self._max_buffer_size-1, :]
        self._buffer_y[self._max_buffer_size-1, : ]=data[0]
        self._buffer_yatt[self._max_buffer_size-1, : ]=att[0]

        if np.random.rand() < 0.5:
            return data, att
        id = np.random.randint(0, self._max_buffer_size)
        return self._buffer_y[id, :].reshape((1, 3, self._image_size, self._image_size)), self._buffer_yatt[id, :].reshape((1, self.filter_size))


    def update_core(self):
        xp = self.gen.xp
        opt_gen = self.get_optimizer('opt_gen')
        opt_dis = self.get_optimizer('opt_dis')
        w_in = self._image_size

        for i in range(self._n_dis):
            self._iter += 1
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            
            #learning rate annealing () 
            if self._learning_rate_anneal_start > self._iter:
                if opt_gen.alpha > 0:
                    opt_gen.alpha -= self._learning_rate_anneal
                if opt_dis.alpha > 0:
                    opt_dis.alpha -= self._learning_rate_anneal

            #data arangement
            x = xp.zeros((batchsize, 3, w_in, w_in)).astype("f")
            #y = xp.zeros((batchsize, 3, w_in, w_in)).astype("f")
            x_att = xp.zeros((batchsize, self.filter_size)).astype("i")
            y_att = xp.zeros((batchsize, self.filter_size)).astype("i")
            for i in range(batchsize):
                x[i, :] = xp.array(batch[i][0])
                x_att[i] = xp.array(batch[i][1])
                y_att[i] = xp.array(batch[i][2])

            x_att_in = x_att.astype("f")
            y_att_in = y_att.astype("f")
            #real
            x_real = Variable(x)
        
            #generate fake
            y_fake = self.gen(x_real,y_att_in) 
            x_fake = self.gen(y_fake,x_att_in)
            y_fake_copy, y_att_copy = self.getAndUpdateBufferY(y_fake.data,y_att)
            
            #Discriminator
            out_real, out_real_att = self.dis(x_real)
            out_fake, out_fake_att = self.dis(y_fake_copy) # (B,1,P,P) P: patch_size of PatchGAN
            patch_size = y_fake.data.shape[-1]

            if i == 0:
                loss_gen_adv = F.sum(-out_fake) / batchsize / patch_size**2
                loss_gen_cls = F.sigmoid_cross_entropy(out_fake_att,Variable(y_att_copy)) #CelebA requires
                loss_gen_rec = F.mean_absolute_error(x_fake,x_real)
                loss_gen = loss_gen_adv + self._lambda_cls * loss_gen_cls + self._lambda_rec * loss_gen_rec
                self.gen.cleargrads()
                loss_gen.backward()
                opt_gen.update()
                chainer.reporter.report({'loss_gen_adv': loss_gen_adv})
                chainer.reporter.report({'loss_gen_cls': loss_gen_cls})
                chainer.reporter.report({'loss_gen_rec': loss_gen_rec})

            x_fake.unchain_backward()

            eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * y_fake_copy
            x_mid_v = Variable(x_mid.data)
            y_mid, _ = self.dis(x_mid_v)
            dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))
            dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
            loss_gp = F.mean_squared_error(dydx, xp.ones_like(dydx.data))

            loss_dis_adv = F.sum(-out_real) / batchsize / patch_size**2
            loss_dis_adv += F.sum(out_fake) / batchsize / patch_size**2
            loss_dis_adv += self._lambda_gp * loss_gp

            loss_dis_cls = F.sigmoid_cross_entropy(out_real_att,Variable(x_att)) #CelebA requires
            loss_dis = loss_dis_adv + self._lambda_cls * loss_dis_cls 

            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            opt_dis.update()

            chainer.reporter.report({'loss_dis_adv': loss_dis_adv})
            chainer.reporter.report({'loss_dis_cls': loss_dis_cls})
            chainer.reporter.report({'loss_gp': loss_gp})
