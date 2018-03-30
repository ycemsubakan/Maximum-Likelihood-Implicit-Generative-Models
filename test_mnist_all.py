import numpy as np
import torch
from algorithms import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, adversarial_wasserstein_trainer
import pdb
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils as ut
import os
from drawnow import drawnow, figure
import visdom 
import torch.optim.lr_scheduler as tol
import torch.nn.functional as F
import itertools as it
import torch.nn as nn
import sklearn.mixture as mix
import pickle
import argparse

# now get (generate) some data and fit 
np.random.seed(2)
torch.manual_seed(9)

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model (NF/VAE/GAN/WGAN)', default='NF')
argparser.add_argument('--use_vis', type=int, help='use visdom to visualize or not', default=1)
argparser.add_argument('--vis_server', type=str, help='use visdom to visualize or not', default='')
argparser.add_argument('--vis_port', type=int, help='visdom port', default=1)

arguments = argparser.parse_args()

if arguments.use_vis:
    vis = visdom.Visdom(port=arguments.vis_port, server=arguments.vis_server, env='dev')
    assert vis.check_connection()
else:
    vis = None


arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 3000
arguments.data = 'mnist'
arguments.input_type = 'autoenc'

train_loader, test_loader = ut.get_loaders(arguments.batch_size, c=0, 
                                           arguments=arguments)

if arguments.data == 'mnist':
    L1 = L2 = 784
    M = N = 28

results = []
results_impl = []
Kss = [[K, 600] for K in range(20, 21, 20)]  #[[20, 600], [40, 600]]
model = arguments.model
if model == 'NF':
    NF_choice = NF_changedim
    nlulayers = 0
    nfulllayers = 0
    nlulayers = 0 
    sig_obs = 1e-2
    EP = 250
    base_inits = 20

    N, M = 28, 28

    # now fit
    base_dist = 'mixture_full_gauss'
    Kdict = 30 

    for config_num, Ks in enumerate(Kss):
        print(Ks)
        NF = NF_choice(N=N, M=M, cuda=arguments.cuda, arguments=arguments, Kdict=Kdict, 
                       base_dist=base_dist, sig_obs=sig_obs, base_inits=base_inits, Ks=Ks)
        if arguments.cuda:
            NF.cuda()

        path = 'models/NF_{}_K_{}.t'.format(arguments.data, NF.Ks)
        if os.path.exists(path):
            NF.load_state_dict(torch.load(path))
            # add another if else here
            if 1 & os.path.exists(path + '.gmm'):
                GMM = pickle.load(open(path + '.gmm', 'rb'))
                NF.GMM = GMM
            else:
                NF.train_base(train_loader)
                pickle.dump(NF.GMM, open(path + '.gmm', 'wb'))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            NF.NF_trainer(train_loader, vis, EP=EP, mode='NF_base', config_num=config_num)
            torch.save(NF.state_dict(), path)
            NF.train_base(train_loader)
            pickle.dump(NF.GMM, open(path + '.gmm', 'wb'))

        av_impl_lls = NF.compute_implicit_density_test(test_loader, arguments.cuda)
        av_lls, im_gen, im_test = compute_nparam_density(test_loader, NF, 0.2, arguments.cuda, num_samples=2, task='mnist')
        results.append((av_lls, Ks))
        results_impl.append((av_impl_lls, Ks))

elif model == 'VAE': 
    EP = 250 
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        mdl = VAE(L1, L2, Ks, M=M) 
       
        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            mdl.VAE_trainer(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=config_num)
            torch.save(mdl.state_dict(), path)

        av_lls, im_gen, im_test = compute_nparam_density(test_loader, mdl, 0.2, arguments.cuda, num_samples=2, task='mnist')
        results.append((av_lls, Ks))

elif model == 'GAN':
    EP = 250

    #Kss = [[40, 600]]
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        generator = netG(L2, Ks)
        discriminator = netD(L2, 200, was=False) 

        if arguments.cuda:
            generator.cuda()
            discriminator.cuda()

        path = 'models/GAN_{}_K_{}.t'.format(arguments.data, Ks)
        if os.path.exists(path):
            generator.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            adversarial_trainer(EP, train_loader, generator, discriminator, arguments, config_num, vis=vis)
            torch.save(generator.state_dict(), path)
        
        av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=2, task='mnist')
        results.append((av_lls, Ks))
elif model == 'GAN_W':
    EP = 250

    #Kss = [[40, 600]]
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        generator = netG(L2, Ks)
        discriminator = netD(L2, 200, was=True) 

        if arguments.cuda:
            generator.cuda()
            discriminator.cuda()

        path = 'models/GAN_W_{}_K_{}.t'.format(arguments.data, Ks)
        if os.path.exists(path):
            generator.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            adversarial_wasserstein_trainer(train_loader, generator, discriminator, EP, arguments=arguments, config_num=config_num, vis=vis)
            torch.save(generator.state_dict(), path)
        
        av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=2, task='mnist')
        results.append((av_lls, Ks))


for dt,_ in it.islice(train_loader, 0, 1, 1):
    pass

N = 64
if model == 'NF':
    gen_data, _ = NF.generate_data(N=N, base_dist=base_dist)
elif model == 'VAE':
    gen_data, _ = mdl.generate_data(N=N)
elif model in ['GAN', 'GAN_W']:
    gen_data, _ = generator.generate_data(N)
gen_randoms = ut.collate_images(gen_data, N=N, ncols=8)

N = 64
sz = 800
opts={'width':sz, 'height':sz}
opts['title'] = 'Generated Images {}'.format(model)
vis.heatmap(gen_randoms, opts=opts, win='generated {}'.format(model))

#opts['title'] = 'Test Images {}'.format(model)
#vis.heatmap(im_test, opts=opts, win='test_images {}'.format(model))

