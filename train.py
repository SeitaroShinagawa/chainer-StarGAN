#!/usr/bin/env python
import argparse
import os
import chainer
from chainer import training
from chainer import cuda, serializers
from chainer.training import extension
from chainer.training import extensions
import sys
import net
from common import celebA
from updater import *
from evaluation import *
from common.record import record_setting
#from chainerui.extensions import CommandsExtension

def main():
    parser = argparse.ArgumentParser(
        description='Train StarGAN')
    parser.add_argument('--source_path', default="source/celebA/",help="data resource Directory")
    parser.add_argument('--att_list_path', default="att_list.txt", help="attribute list")
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--max_iter', '-m', type=int, default=200000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_folder', '-e', default='test',
                        help='Directory to output the evaluation result')

    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Interval of evaluating generator')

    parser.add_argument("--learning_rate_g", type=float, default=0.0001, help="Learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=0.0001, help="Learning rate for discriminator")
    parser.add_argument("--load_gen_model", default='', help='load generator model')
    parser.add_argument("--load_dis_model", default='', help='load discriminator model')

    parser.add_argument('--gen_class', default='StarGAN_Generator', help='Default generator class')
    parser.add_argument('--dis_class', default='StarGAN_Discriminator', help='Default discriminator class')

    parser.add_argument("--n_dis", type=int, default=6, help='The number of loop of WGAN Discriminator')
    parser.add_argument("--lambda_gp", type=float, default=10.0, help='lambda for gradient penalty of WGAN')
    parser.add_argument("--lambda_adv", type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument("--lambda_cls", type=float, default=1.0, help='lambda for classification loss')
    parser.add_argument("--lambda_rec", type=float, default=10.0, help='lambda for reconstruction loss')

    parser.add_argument("--flip", type=int, default=1, help='flip images for data augmentation')
    parser.add_argument("--resize_to", type=int, default=128, help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=178, help='crop the resized image to')
    parser.add_argument("--load_dataset", default='celebA_train', help='load dataset')
    parser.add_argument("--discriminator_layer_n", type=int, default=6, help='number of discriminator layers')

    parser.add_argument("--learning_rate_anneal", type=float, default=10e-8, help='anneal the learning rate')
    parser.add_argument("--learning_rate_anneal_start", type=int, default=100000, help='time to anneal the learning')
    
    args = parser.parse_args()
    print(args)
    record_setting(args.out)
    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    with open(args.att_list_path,"r") as f:
        att_list = []
        att_name = []
        for line in f:
            line = line.strip().split(" ")
            if len(line)==3:
                att_list.append(int(line[0])) #attID
                att_name.append(line[1]) #attname
    print("attribute list:",",".join(att_name))

    #load dataset
    train_dataset = getattr(celebA, args.load_dataset)(args.source_path, att_name, flip=args.flip, resize_to=args.resize_to, crop_to=args.crop_to)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, args.batch_size, n_processes=4)

    #test_dataset = getattr(celebA, args.load_dataset)(root_celebA, flip=args.flip, resize_to=args.resize_to, crop_to=args.crop_to)
    test_batchsize = 8
    test_iter = chainer.iterators.SerialIterator(train_dataset, test_batchsize) 

    #set generator and discriminator 
    nc_size = len(att_list) #num of attribute
    gen = getattr(net, args.gen_class)(args.resize_to, nc_size)
    dis = getattr(net, args.dis_class)(n_down_layers=args.discriminator_layer_n)

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator model loaded")

    if args.load_dis_model != '':
        serializers.load_npz(args.load_dis_model, dis)
        print("Discriminator model loaded")

    if not os.path.exists(args.eval_folder):
         os.makedirs(args.eval_folder)

    # select GPU
    if args.gpu >= 0:
        gen.to_gpu()
        dis.to_gpu()
        print("use gpu {}".format(args.gpu))

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0001, beta1=0.5, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer

    opt_gen=make_optimizer(gen, alpha=args.learning_rate_g)
    opt_dis=make_optimizer(dis, alpha=args.learning_rate_d)

    # Set up a trainer
    updater = Updater(
        models=(gen, dis),
        iterator={
            'main': train_iter,
            'test': test_iter
            },
        optimizer={
            'opt_gen': opt_gen,
            'opt_dis': opt_dis,
            },
        device=args.gpu,
        params={
            'n_dis': args.n_dis,
            'lambda_adv': args.lambda_adv,
            'lambda_cls': args.lambda_cls,
            'lambda_rec': args.lambda_rec,
            'lambda_gp': args.lambda_gp,
            'image_size' : args.resize_to,
            'eval_folder' : args.eval_folder,
            'nc_size': nc_size, 
            'learning_rate_anneal' : args.learning_rate_anneal,
            'learning_rate_anneal_start' : args.learning_rate_anneal_start,
            'dataset' : train_dataset
        })

    model_save_interval = (4000, 'iteration')
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=args.out)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis{.updater.iteration}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'lr_g', 'lr_d', 'loss_dis_adv', 'loss_gen_adv', 'loss_dis_cls', 'loss_gen_cls','loss_gen_rec', 'loss_gp']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(20, 'iteration')))
    trainer.extend(extensions.PrintReport(log_keys), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=50))

    trainer.extend(
        evaluation(gen, args.eval_folder, image_size=args.resize_to
        ), trigger=(args.eval_interval ,'iteration')
    )
   
    #trainer.extend(CommandsExtension())
    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
