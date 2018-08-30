import numpy as np
import torch
from algorithms_v2 import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, adversarial_wasserstein_trainer, get_embeddings, get_scores
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
import torchvision
from sklearn.decomposition import NMF

vis = visdom.Visdom(port=5800, server='your server', env='your env')
assert vis.check_connection()

# now get (generate) some data and fit 
np.random.seed(2)
torch.manual_seed(9)

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model', default='NF')
arguments = argparser.parse_args()

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
num_samples = 10
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
        av_lls, im_gen, im_test = compute_nparam_density(test_loader, NF, 0.2, arguments.cuda, num_samples=num_samples, task='mnist')
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


            # train the VAE
            mdl.VAE_trainer(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=config_num)

            torch.save(mdl.state_dict(), path)

            # get the embeddings 
            #all_hhats = get_embeddings(mdl, train_loader)

            # fit the GMM 
            #GMM = mix.GaussianMixture(n_components=20, covariance_type='diag')
            #GMM.fit(all_hhats.data.cpu().numpy())
        

        av_lls, im_gen, im_test = compute_nparam_density(test_loader, mdl, 0.2, arguments.cuda, num_samples=num_samples, task='mnist')
        results.append((av_lls, Ks))

elif model == 'VaDE':

    EP = 100 
    num_samples = 1
    
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        mdl = VAE(L1, L2, Ks, M=M) 
       
        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if 1 & os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            pass
        all_hhats = get_embeddings(mdl, train_loader)

        if 1:
            scores_vae, vae_samples = get_scores(test_loader, mdl, arguments.cuda, 
                                       num_samples=num_samples,
                                       task='mnist', base_dist='fixed_iso_gauss')
            #pickle.dump(scores_vae, 
            #            open('/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/vae_mnist.pk', 'wb'))
            torchvision.utils.save_image(vae_samples[:64].reshape(-1, 1, 28, 28),  '/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/samples_vae_mnist.png')
            
        for Kcomps in range(50, 51, 5):
            print('Number of GMM components {}'.format(Kcomps))
            path2 = 'models/VaDEGMM_{}_K_{}_Kmog_{}.t'.format(arguments.data, Ks, Kcomps)
            
            if 0 & os.path.exists(path2 + 'Kmog{}.gmm'.format(Kcomps)):
                GMM = pickle.load(open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'rb'))
            else:
                GMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=10, max_iter=200, covariance_type='diag', warm_start=True)
                GMM.fit(all_hhats.data.cpu().numpy())
                pickle.dump(GMM, open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'wb'))
            mdl.initialize_GMMparams(GMM)
            if arguments.cuda:
                mdl.cuda()
           
            scores_iml, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                              num_samples=num_samples,
                                              task='mnist', base_dist='mog')
            
            torchvision.utils.save_image(gen_data[:64].reshape(-1, 1, 28, 28),  '/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/samples_iml_mnist_{}.png'.format(Kcomps))

            #gen_images = ut.collate_images(gen_data, 64)
            #opts = {}
            #opts['title'] = 'IML Generated Images'
            #vis.heatmap(gen_images, opts=opts, win='iml_gen_data')

            if 1 & os.path.exists(path2):
                mdl.load_state_dict(torch.load(path2))
            else:
                mdl.VAE_trainer_mog(arguments.cuda, train_loader, vis=vis, 
                                    EP=EP, config_num=config_num)

                torch.save(mdl.state_dict(), path2)

            scores_vade, gen_data_vade = get_scores(test_loader, mdl, arguments.cuda, 
                                              num_samples=num_samples,
                                              task='mnist', base_dist='mog')

            torchvision.utils.save_image(gen_data_vade[:64].reshape(-1, 1, 28, 28),  '/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/samples_vade_mnist_{}.png'.format(Kcomps))

            #gen_images = ut.collate_images(gen_data_vade, 64)
            #opts = {}
            #opts['title'] = 'VADE Generated Images'
            #vis.heatmap(gen_images, opts=opts, win='vade_gen_data')

            results.append({'Kcomps':Kcomps, 
                            'scores_iml': scores_iml, 
                            'scores_vade': scores_vade})

            #pickle.dump(results, open('/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/imlvsjoint_mnist.pk', 'wb'))
            #results.append((av_lls, Ks))

        pdb.set_trace()

elif model == 'NMFGMM':
    for tar, _ in it.islice(train_loader, 0, 1, 1):
        pass
    tar = tar.reshape(-1, 784).numpy()

    nmf_model = NMF(n_components=50, init='nndsvd', solver='mu', beta_loss='kullback-leibler', verbose=True, max_iter=300)
    H = nmf_model.fit_transform(tar)
    W = nmf_model.components_

    recons = np.dot(H, W)

    GMM = mix.GaussianMixture(n_components=50, verbose=1, n_init=5, max_iter=200, covariance_type='full', warm_start=True)
    GMM.fit(H)

    h_samples = GMM.sample(64)[0]
    gen_data = np.dot(h_samples, W)

    rec_images = ut.collate_images(torch.from_numpy(recons).float().cuda(), N=64)
    opts = {}
    opts['title'] = 'NMF Reconstructions'
    vis.heatmap(rec_images, opts=opts, win='nmf_reconstructions')

    gen_images = ut.collate_images(torch.from_numpy(gen_data).float().cuda(), N=64)
    opts = {}
    opts['title'] = 'NMF Samples'
    vis.heatmap(gen_images, opts=opts, win='nmf_samples')

elif model == 'PCAGMM':
    for tar, _ in it.islice(train_loader, 0, 1, 1):
        pass
    if arguments.cuda:
        tar = tar.cuda()
    tar = tar.reshape(-1, 784)
    tar_mean = tar.mean(0, keepdim=True)
    tar_centered = tar - tar_mean

    U, S, V = torch.svd(tar_centered.t())
    Uthin = U[:, :100]

    zs = torch.matmul(Uthin.t(), tar_centered.t()).t()
    recons = (torch.matmul(Uthin, zs.t()) + tar_mean.t()).t()

    GMM = mix.GaussianMixture(n_components=30, verbose=1, n_init=5, max_iter=200, covariance_type='full', warm_start=True)
    GMM.fit(zs.data.cpu().numpy())

    z_samples = torch.from_numpy(GMM.sample(64)[0]).float()
    if arguments.cuda:
        z_samples = z_samples.cuda()

    gen_data = (torch.matmul(Uthin, z_samples.t()) + tar_mean.t()).t()

    gen_images = ut.collate_images(gen_data, N=64)
    opts = {}
    opts['title'] = 'PCA Samples'
    vis.heatmap(gen_images, opts=opts, win='pca_samples')

    recons_images = ut.collate_images(recons, N=64)
    opts = {}
    opts['title'] = 'PCA Reconstructions'
    vis.heatmap(recons_images, opts=opts, win='pca_recons')

    pdb.set_trace()


elif model == 'VAEpGMM':
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

        all_hhats = get_embeddings(mdl, train_loader)

        GMM = mix.GaussianMixture(n_components=30, verbose=1, n_init=5, max_iter=200, covariance_type='full', warm_start=True)
        GMM.fit(all_hhats.data.cpu().numpy())

        seed = GMM.sample(100)[0]
        seed = torch.from_numpy(seed).float()

        seed = seed.cuda()
        gen_data = mdl.decode(seed)
        gen_data = gen_data.view(-1, 1, 28, 28)

        torchvision.utils.save_image(gen_data, 'vaepgmm_images.png', nrow=8)

        pdb.set_trace()

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
        
        av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=num_samples, task='mnist')
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
        
        av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=num_samples, task='mnist')
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

opts['title'] = 'Test Images {}'.format(model)
vis.heatmap(im_test, opts=opts, win='test_images {}'.format(model))

pdb.set_trace()
imagepath = os.path.expanduser('~')
imagepath = os.path.join(imagepath, 'Dropbox', 'GANs', 'UAI2018', 'paper', 'figures')
if not os.path.exists(imagepath):
    os.mkdir(imagepath)

print(results)

# save them results here
if 0:
    pickle.dump(results, open(imagepath + '/Kvskde_thesis_am_{}.pickle'.format(model), 'wb')) 

    if model in ['NF']:
        print(results_impl)
        pickle.dump(results_impl, open(imagepath + '/Kvsimpl_thesis_am_{}.pickle'.format(model), 'wb')) 

# save them images here 
if 0:
    if model == 'NF':
        pickle.dump(im_test.numpy(), open(imagepath + '/test_images_{}.pickle'.format(arguments.data), 'wb')) 
    pickle.dump(im_gen.numpy(), open(imagepath + '/kde_images_{}.pickle'.format(model), 'wb'))
    pickle.dump(gen_randoms.numpy(), open(imagepath + '/mnist_random_{}.pickle'.format(model), 'wb'))




