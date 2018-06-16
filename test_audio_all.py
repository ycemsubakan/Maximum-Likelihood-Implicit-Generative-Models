import numpy as np
import torch
from algorithms_v2 import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, conv_autoenc, CharRNN, audionet
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
from torchvision import datasets, transforms
import argparse
import string
import wavenet_things.audio_data as ad
import librosa as lr

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev')
assert vis.check_connection()

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
arguments = argparser.parse_args()

arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 200

np.random.seed(2)
torch.manual_seed(9)
arguments.cuda = torch.cuda.is_available()

arguments.data = 'real_violin' 

train_loader, wf = ut.preprocess_audio_files(arguments, overlap=True)

model = 'NF'
results = []
if model == 'NF':
    EP = 200
    base_inits = 20

    # now fit
    base_dist = 'HMM'
    Kdict = 370
    L = 800

    Kss = [[80]]
    for config_num, Ks in enumerate(Kss):
        mdl = audionet(L, Ks[0], 1, 1, 1, base_inits=base_inits, Kdict=Kdict, base_dist=base_dist, 
                       num_gpus=arguments.num_gpus, usecuda=arguments.cuda)

        if arguments.cuda:
            mdl.cuda()
        
        if base_dist == 'HMM':
            bd = mdl.HMM
            ext = '.hmm'
        elif base_dist == 'GMM':
            bd = mdl.GMM
            ext = '.gmm'

        path = 'models/charnn_{}_K_{}.t'.format(arguments.data, Ks)
        if 1 and os.path.exists(path):
            
            mdl.load_state_dict(torch.load(path))
            #mdl.trainer(train_loader, vis, EP, arguments.cuda, config_num) 
            #torch.save(mdl.state_dict(), path)
            if 0:
                mdl.base_dist_trainer(train_loader, arguments.cuda, vis, path)  
                pickle.dump(bd, open(path + ext, 'wb'))
            else:
                if base_dist == 'HMM':
                    mdl.HMM = pickle.load(open(path + ext, 'rb'))
                else:
                    mdl.GMM = pickle.load(open(path + ext, 'rb'))

        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            mdl.trainer(train_loader, vis, EP, arguments.cuda, config_num) 
            torch.save(mdl.state_dict(), path)
            
            mdl.base_dist_trainer(train_loader, arguments.cuda, vis=vis, path=path)  
            if base_dist == 'mixture_full_gauss':
                pickle.dump(mdl.GMM, open(path + '.gmm', 'wb'))
            #elif base_dist == 'HMM':
            #    pickle.dump(mdl.HMM, open(path + '.hmm', 'wb'))

        #av_lls, im_gen, im_test = compute_nparam_density(test_loader, NF, 0.2, arguments.cuda, num_samples=2)
        #results.append((av_lls, Ks))
        gen_data, seed = mdl.generate_data(4000, arguments)
        opts = {'title' : 'generated data {}'.format(model)}
        vis.line(gen_data.squeeze()[:3].t().data.cpu(), win='generated_{}'.format(model), opts=opts)
        
        gen_data_concat = ut.pt_to_audio_overlap(gen_data) 

        vis.line(gen_data_concat[:2000], win='generated_concat')
        vis.heatmap(seed.data.cpu()[:200].squeeze().t(), win='hhat-gen', opts={'title': 'Generated hhat'})
        
        lr.output.write_wav('sample_{}.wav'.format(arguments.data), gen_data_concat, 8000, norm=True)

# do some plotting here
imagepath = os.path.expanduser('~')
imagepath = os.path.join(imagepath, 'Dropbox', 'GANs', 'UAI2018', 'paper', 'figures')
if not os.path.exists(imagepath):
    os.mkdir(imagepath)

Tstart = 120000
Tmax = 120000*2
pw = .3
plt.subplot(2, 1, 1)
spec = np.abs(lr.stft(wf[Tstart:Tmax]))**pw
lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
plt.title('Original Data')

plt.subplot(2, 1, 2)
spec = np.abs(lr.stft(gen_data_concat[Tstart:Tmax]))**pw
lr.display.specshow(spec, sr=8000, y_axis='log', x_axis='time')
plt.title('Generated Data')

plt.savefig(imagepath + '/{}.eps'.format(arguments.data), format='eps')
