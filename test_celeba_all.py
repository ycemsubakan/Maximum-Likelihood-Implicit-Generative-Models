import numpy as np
import torch
from algorithms_v2 import netD, netG, adversarial_trainer, VAE, NF_changedim, compute_nparam_density, conv_autoenc, netg_dcgan, netd_dcgan, conv_VAE, adversarial_wasserstein_trainer, netd_dcgan_par, netg_dcgan_par, weights_init_seq, get_embeddings, get_scores
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
from WassersteinGAN.models.dcgan import DCGAN_G
import gmms.gmm_learn as cgmm
import torchvision

vis = visdom.Visdom(port=5800, server='your server', env='your env')
assert vis.check_connection()

# now get (generate) some data and fit 
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='choose your model', default='NF')
arguments = argparser.parse_args()

np.random.seed(2)
torch.manual_seed(9)
arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 128
arguments.data = 'celeba'
arguments.input_type = 'autoenc'

isCrop = 0
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
train_data_dir = '/media/data2/crop_celeba_train/'          # this path depends on your directories
test_data_dir = '/media/data2/crop_celeba_test/'

dset_train = datasets.ImageFolder(train_data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=128, shuffle=True,
                                           pin_memory=True, num_workers=arguments.num_gpus)

dset_test = datasets.ImageFolder(test_data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=24, shuffle=False)

for dt in it.islice(train_loader, 0, 1, 1):
    vis.images(0.5+(dt[0]*0.5), nrow=8, win='celebafaces')
     
    #pdb.set_trace()

#h = torch.randn(100, 100, 1, 1)
#out = netG.forward(Variable(h))

compute_kdes = 1
results = []
mmds = []
fids = []
model = arguments.model
if model == 'NF':
    EP = 25
    base_inits = 10

    N, M = 28, 28

    # now fit
    base_dist = 'mixture_full_gauss'
    Kdict = 200

    Kss = [[100]]
    for config_num, Ks in enumerate(Kss):
        mdl = conv_autoenc(base_inits=base_inits, K=Ks[0], Kdict=Kdict,
                           num_gpus=arguments.num_gpus)
        if arguments.cuda:
            mdl.cuda()

        #path = 'models/convauto_nobatch_reluft_{}_K_{}.t'.format(arguments.data, Ks)
        path = 'models/convauto_{}_K_{}.t'.format(arguments.data, Ks)

        if 1 and os.path.exists(path):
            
            mdl.load_state_dict(torch.load(path))
            if 0 and os.path.exists(path + '.gmm'):
                mdl.GMM = pickle.load(open(path + '.gmm', 'rb'))
            else:
                mdl.gmm_trainer(train_loader, arguments.cuda, vis=vis)  
                pickle.dump(mdl.GMM, open(path + '.gmm', 'wb'))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            mdl.trainer(train_loader, vis, EP, arguments.cuda, config_num) 
            torch.save(mdl.state_dict(), path)
            mdl.gmm_trainer(train_loader, arguments.cuda, vis=vis)  
            pickle.dump(mdl.GMM, open(path + '.gmm', 'wb'))
        gen_data, seed = mdl.generate_data(100)
        opts = {'title':'NF generated data config {}'.format(config_num)}
        vis.images(0.5 + 0.5*gen_data.data.cpu(), opts=opts, win='NF_config_{}'.format(config_num))

        if compute_kdes:
            num_samples = 1
            #av_lls, im_gen, im_test = compute_nparam_density(test_loader, mdl, 0.2,
            #                   arguments.cuda, num_samples=num_samples)
            #av_mmds = get_mmds(test_loader, mdl, 1, arguments.cuda, 
            #                   num_samples=num_samples)

            #results.append((av_lls, Ks))
            #mmds.append((av_mmds, Ks))

            fid = get_mmds(test_loader, mdl, 1, arguments.cuda, num_samples, metric='FID')
            fids.append((np.mean(fid), np.std(fid), Ks))

            #vis.image(im_gen*0.5 + 0.5, win='NF genim') 
            #vis.image(im_test*0.5 + 0.5, win='NF testim') 


elif model == 'VAE': 
    EP = 25
    Kss = [[100, 100]]
    L = 64*64*3
    for config_num, Ks in enumerate(Kss):
        
        mdl = conv_VAE(L, L, Ks, M=64, num_gpus=arguments.num_gpus) 
       
        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if 1 & os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            mdl.VAE_trainer(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=config_num)
            torch.save(mdl.state_dict(), path)

        #gen_data, seed = mdl.generate_data(100)
        #opts = {'title':'VAE generated data config {}'.format(config_num)}
        #vis.images(0.5 + 0.5*gen_data.data.cpu(), opts=opts, win='VAE_config_{}'.format(config_num))
        
        if compute_kdes:
            num_samples = 1

            scores_iml, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                              num_samples=num_samples,
                                              task='celeba', base_dist='fixed_iso_gauss')
            vis.images(gen_data.data*0.5 + 0.5, win='VAE genim')
            pdb.set_trace() 
            #vis.image(im_gen*0.5 + 0.5, win='VAE genim') 
            #vis.image(im_test*0.5 + 0.5, win='VAE testim') 


elif model == 'VaDE':

    EP = 5
    num_samples = 1
    Kss = [[100, 100]]
    L = 64*64*3
    
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        mdl = conv_VAE(L, L, Ks, M=64, num_gpus=arguments.num_gpus) 

        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if 1 & os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            pass
        all_hhats = get_embeddings(mdl, train_loader)
        all_hhats_chunks = torch.chunk(all_hhats, chunks=35, dim=0)

        if 1:
            scores_vae, samples_vae = get_scores(test_loader, mdl, arguments.cuda, 
                                       num_samples=num_samples,
                                       task='celeba', base_dist='fixed_iso_gauss')
            #pickle.dump(scores_vae, 
            #            open('/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/vae_celeba.pk', 'wb'))
            torchvision.utils.save_image(0.5*samples_vae[:64] + 0.5,  '/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/samples_vae_celeba.png')

        use_gmms = [1, 1, 0]
        scores_iml_all = []
        for Kcomps in range(25, 26, 5):
            print('Number of GMM components {}'.format(Kcomps))
            path2 = 'models/VaDEGMM_{}_K_{}_Kmog_{}.t'.format(arguments.data, Ks, Kcomps)
            
            mdl = conv_VAE(L, L, Ks, M=64, num_gpus=arguments.num_gpus) 
            if arguments.cuda:
                mdl.cuda()
            mdl.load_state_dict(torch.load(path))

            if 0 & os.path.exists(path2 + 'Kmog{}.gmm'.format(Kcomps)):
                GMM = pickle.load(open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'rb'))
            else:
                if use_gmms[0]:
                    GMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=10, max_iter=200, covariance_type='diag')
                    GMM.fit(all_hhats.data.cpu().numpy())
                    pickle.dump(GMM, open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'wb'))
                    mdl.initialize_GMMparams(GMM)

                if use_gmms[1]:
                    BGMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=10, max_iter=200, covariance_type='full')
                    BGMM.fit(all_hhats.data.cpu().numpy())
                    pickle.dump(BGMM, open(path2 + 'Kmog{}.bgmm'.format(Kcomps), 'wb'))
                    mdl.GMM = BGMM

                if use_gmms[2]:
                    cudagmm = cgmm.gmm(num_components=Kcomps, L=Ks[0], cuda=arguments.cuda)
                    cudagmm.kmeanspp(all_hhats)
                    cudagmm.kmeans(all_hhats_chunks, vis)
                    cudagmm.em(all_hhats_chunks, em_iters=15)
                    mdl.GMM = cudagmm


            if arguments.cuda:
                mdl.cuda()
           
            if 1:
                if use_gmms[0]:
                    scores_iml, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog')
                else: 
                    scores_iml = []

                if use_gmms[1]:
                    scores_iml_bgmm, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog_skt')

                    torchvision.utils.save_image(0.5*gen_data[:64] + 0.5,  '/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/samples_iml_celeba_{}.png'.format(Kcomps))
                else:
                    scores_iml_bgmm = []

                if use_gmms[2]:
                    scores_iml_cuda, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog_cuda')
                else:
                    scores_iml_cuda = []


                scores_iml_all.append({'Kcomps':Kcomps, 
                                       'scores_iml': scores_iml, 
                                       'scores_iml_bgmm': scores_iml_bgmm,
                                       'score_iml_cuda': scores_iml_cuda})

                #pickle.dump(scores_iml_all, 
                #        open('/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/iml_celeba_{}_Kmog{}.pk'.format(use_gmms, Kcomps), 'wb'))

            
            if 1:
                if 1 & os.path.exists(path2):
                    mdl.load_state_dict(torch.load(path2))
                    
                    scores_vade, gen_data_vade = get_scores(test_loader, mdl, arguments.cuda, 
                                                      num_samples=num_samples,
                                                      task='celeba', base_dist='mog')

                    torchvision.utils.save_image(0.5 + 0.5*gen_data_vade[:64],  '/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/samples_vade_celeba_{}.png'.format(Kcomps))


                else:
                    mdl.VAE_trainer_mog(arguments.cuda, train_loader, vis=vis, 
                                        EP=EP, config_num=config_num, data='celeba')

                    torch.save(mdl.state_dict(), path2)

                    scores_vade, gen_data_vade = get_scores(test_loader, mdl, arguments.cuda, 
                                                      num_samples=num_samples,
                                                      task='celeba', base_dist='mog')

                    torchvision.utils.save_image(gen_data_vade[:64],  '/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/samples_vade_celeba_{}.png'.format(Kcomps))

                    results.append({'Kcomps':Kcomps, 
                                    'scores_iml': scores_iml, 
                                    'scores_iml_bgmm': scores_iml_bgmm,
                                    'scores_vade': scores_vade})

                    #pickle.dump(results, open('/home/cem/Dropbox/GANs/UAI2018/mtl_ai_pres/imlvsjoint_celeba_{}.pk'.format(Kcomps), 'wb'))

        pdb.set_trace()



elif model == 'GAN':
    EP = 25

    Kss = [[100]]
    for config_num, Ks in enumerate(Kss):
        generator = netg_dcgan(Ks[0])
        discriminator = netd_dcgan()

        generator.weight_init(mean=0.0, std=0.02)
        discriminator.weight_init(mean=0.0, std=0.02)

        if arguments.cuda:
            generator.cuda()
            discriminator.cuda()

        path = 'models/GAN_{}_K_{}.t'.format(arguments.data, Ks)
        if os.path.exists(path):
            generator.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            adversarial__trainer(EP, train_loader, generator, discriminator, arguments, config_num, vis=vis)
            torch.save(generator.state_dict(), path)
        
        if compute_kdes:
            num_samples=1
            #av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=num_samples)
            #av_mmds = get_mmds(test_loader, generator, 10, arguments.cuda, 
            #                   num_samples=num_samples)

            #results.append((av_lls, Ks))
            #mmds.append((av_mmds, Ks))
            fid = get_mmds(test_loader, generator, 1, arguments.cuda, num_samples, metric='FID')
            fids.append((np.mean(fid), np.std(fid), Ks))

elif model == 'GAN_W':
    EP = 250

    Kss = [[100]]
    for config_num, Ks in enumerate(Kss):
        generator = DCGAN_G(64, 100, 3, 128, ngpu=arguments.num_gpus, 
                            n_extra_layers=0)
        netg_dcgan_par(Ks[0])
        #discriminator = netd_dcgan_par()

        #generator.apply(weights_init_seq)
        #discriminator.apply(weights_init_seq)

        if arguments.cuda:
            generator.cuda()
            #discriminator.cuda()

        path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GANs', 'implicit_models',
                            'WassersteinGAN', 'GANW_128_netG_celeba_epoch_24.t')
        if 1: #& os.path.exists(path):
            generator.load_state_dict(torch.load(path))
            pdb.set_trace()
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            adversarial_wasserstein_trainer(train_loader, generator, discriminator, arguments=arguments, config_num=config_num, vis=vis, EP = EP)
            torch.save(generator.state_dict(), path)
        
        if compute_kdes:
            num_samples = 10
            av_lls, im_gen, im_test = compute_nparam_density(test_loader, generator, 0.2, arguments.cuda, num_samples=num_samples)
            av_mmds = get_mmds(test_loader, generator, 10, arguments.cuda, 
                               num_samples=num_samples)

            results.append((av_lls, Ks))
            mmds.append((av_mmds, Ks))



imagepath = os.path.expanduser('~')
imagepath = os.path.join(imagepath, 'Dropbox', 'GANs', 'UAI2018', 'paper', 'figures')
if not os.path.exists(imagepath):
    os.mkdir(imagepath)

if 0:
    
    N = 64
    if model == 'NF':
        gen_data, _ = mdl.generate_data(N=N, base_dist=base_dist)
    elif model == 'VAE':
        gen_data, _ = mdl.generate_data(N)
    elif model in ['GAN', 'GAN_W']:
        gen_data, _ = generator.generate_data(N=N)
    gen_randoms = ut.collate_images_color(gen_data, N=N, ncols=8)

    if 1:
        plt.imshow(0.5*gen_randoms.permute(1, 2, 0).numpy()+0.5, interpolation='None')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(imagepath + '/more_randoms_celeba_{}.eps'.format(model), format='eps')

    if compute_kdes:
        print(results)
        if 1:
            if model == 'NF':
                pickle.dump(im_test.numpy(), open(imagepath + '/test_images_{}.pickle'.format(arguments.data), 'wb')) 
            pickle.dump(im_gen.numpy(), open(imagepath + '/celeba_kde_images_{}.pickle'.format(model), 'wb'))
            pickle.dump(gen_randoms.numpy(), open(imagepath + '/celeba_random_{}.pickle'.format(model), 'wb'))
            pickle.dump(results, open(imagepath + '/celeba_Kvskde_{}.pickle'.format(model), 'wb')) 

print(results)
print(mmds)
print(fids)
if 0:    
    pickle.dump(results, open(imagepath + '/celeba_results_thesis_am_{}.pickle'.format(model), 'wb')) 
    pickle.dump(mmds, open(imagepath + '/celeba_mmds_thesis_am_{}.pickle'.format(model), 'wb')) 




