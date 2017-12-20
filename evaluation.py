
import chainer
from chainer.training import extension
from chainer import Variable
import numpy as np
from PIL import Image

def batch_postprocess_images(img, batch_w, batch_h):
    b, ch, w, h = img.shape
    img = img.reshape((batch_w, batch_h, ch, w, h))
    img = img.transpose(0,1,3,4,2)
    img = (img + 1) *127.5
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img.reshape((batch_w, batch_h, w, h, ch)).transpose(0,2,1,3,4).reshape((w*batch_w, h*batch_h, ch))[:,:,::-1]
    return img


def evaluation(gen, test_image_folder, image_size=128, side=2):
    @chainer.training.make_extension()
    def _eval(trainer, it):
        xp = gen.xp
        batch = it.next()
        batchsize = len(batch)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            x_origin = xp.zeros((batchsize, 3, image_size, image_size)).astype("f")
            att = np.array([[int(k) for k in j] for j in [bin(i)[2:].zfill(gen.nc_size) for i in range(2**gen.nc_size)]]).astype("f") #make all cases of 2bit attributes
            for i in range(batchsize):
                x_origin[i, :] = xp.array(batch[i][0])
       
            x = Variable(x_origin)

            for i in range(2**gen.nc_size):
                att_part = xp.array(np.broadcast_to(att[i],(batchsize, gen.nc_size)))
            
                result = gen(x,att_part)
                img = result.data.get()
                img = batch_postprocess_images(img, 1, batchsize)
                Image.fromarray(img).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_"+str(i+1).zfill(len(str(2**gen.nc_size)))+".jpg")

        x_origin = x_origin.get()
        img_origin = batch_postprocess_images(x_origin, 1, batchsize)
        Image.fromarray(img_origin).save(test_image_folder+"/iter_"+str(trainer.updater.iteration)+"_"+str(0).zfill(len(str(2**gen.nc_size)))+".jpg")

    def evaluation(trainer):
        it = trainer.updater.get_iterator('test')
        _eval(trainer, it)

    return evaluation
