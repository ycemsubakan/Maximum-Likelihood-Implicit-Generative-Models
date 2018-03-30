import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from drawnow import drawnow, figure
import torch.utils.data as data_utils
import torch.nn.utils.rnn as rnn_utils
from itertools import islice
import os
import torch.nn.init as torchinit
import librosa as lr
import librosa.display as lrd
import utils as ut
import itertools as it
import copy
import visdom
#import pytorch_fft.fft.autograd as pfft
import scipy.linalg as spl
import math
import sklearn.mixture as mix
import string as st
from hmmlearn import hmm

class Normalizing_Flow_utils(nn.Module):
    def __init__(self, c, useu=False, cuda=False):
        super(Normalizing_Flow_utils, self).__init__()
        self.c = c
        self.usecuda = cuda
        self.useu = useu
    
    def compute_implicit_density_test(self, test_loader, cuda):
        log_densities = []
        gmm_scores = []
        for test_data, _ in test_loader:
            test_data = test_data.view(-1, self.L2)
            if cuda: 
                test_data = test_data.cuda()
            test_data = Variable(test_data)
            log_density, _ , _ = self.compute_density_normal(invf=self.inverse_forward, x=test_data)
            log_densities.append(log_density[0] + log_density[1])
            gmm_scores.append(log_density[0])
        log_density_all = torch.cat(log_densities, dim=0)
        gmm_scores_all = torch.cat(gmm_scores, dim=0)
        mean_log_density = log_density_all.mean(0)
        mean_gmm_density = gmm_scores_all.mean(0)
        return mean_log_density[0], mean_gmm_density[0]


    def tanh_linear(self, x, nl):
        c = self.c[nl]
        mask = torch.le(torch.abs(x), 1).float()

        mask2 = torch.gt(x, 1).float() 
        mask3 = torch.lt(x, -1).float()

        y = mask*x + mask2*((c*x).add(1-c)) + mask3*((c*x).add(-1+c)) 
        return y

    def tanh_linear_inverse(self, x, nl):
        c = self.c[nl]
        mask = torch.le(torch.abs(x), 1).float()

        mask2 = torch.gt(x, 1).float() 
        mask3 = torch.lt(x, -1).float()

        y = mask*x + mask2*(x.add(c-1)/c) + mask3*(x.add(-c+1)/c) 
        
        ones = torch.ones(x.size())
        if self.usecuda:
            ones = ones.cuda()
        ones = Variable(ones)

        yprime = mask*ones + (1-mask)*ones*(1/c)
        return y, yprime

    def compute_sigm_limits(self, eps=0.01):
        xlims = float(np.log(1-eps) - np.log(eps))
        c = float(np.exp(-xlims)/( (1+np.exp(-xlims))**2 ))
        bu = 1 - eps - c*xlims
        bl = eps + c*xlims

        self.es = eps
        self.xlims_sig = xlims
        self.sigc = c
        self.bu = bu
        self.bl = bl

    def sigmoid_si(self, x):
        '''
        This function requires calling compute_sigm_limits() before using it.
        '''
        mask1 = torch.gt(x, self.xlims_sig).float() 
        mask2 = torch.lt(x, -self.xlims_sig).float()
        mask = 1 - mask1 - mask2

        c = self.sigc
        y = mask*F.sigmoid(x) + mask1*((c*x).add(self.bu)) + mask2*((c*x).add(self.bl)) 
        return y

    def sigmoid_si_inverse(self, x):
        '''
        This function requires calling compute_sigm_limits() before using it.
        '''
        
        mask1 = torch.gt(x, 1-self.es).float() 
        mask2 = torch.lt(x, self.es).float()
        mask = 1 - mask1 - mask2

        sigmoid_inverse = lambda y: y.log() - (1-y).log()
        sigmoid_inverse_prime = lambda y : (1/y) + (1/(1-y))

        xmasked = x*mask
        mask_aux = torch.lt(xmasked.abs(), 1e-10).float()
        xmasked = (mask_aux)*0.5 + xmasked
        #xmasked[xmasked.abs() < 1e-10] = 0.5 # this is to avoid nans
        y = mask*sigmoid_inverse(xmasked) + mask1*(x-self.bu)/self.sigc + mask2*(x-self.bl)/self.sigc
        yprime = mask*sigmoid_inverse_prime(x) + (mask1+mask2)*(1/self.sigc)
        return y, yprime


    def compute_tanh_limits(self, ymax=0.95):
        xmax = 0.5*(np.log(1+ymax) - np.log(1-ymax))    
        clims = 4*np.exp(-2*xmax)/((1+np.exp(-2*xmax))**2)
        b = ymax - clims*xmax

        self.tanh_ymax = ymax
        self.tanh_xmax = xmax
        self.tanh_clims = clims
        self.tanh_b = b

    def tanh_si(self, x, nl=None):
        '''
        This function requires calling compute_tanh_limits() before using it.
        '''
        ylim = self.tanh_ymax 
        xlim = self.tanh_xmax
        c = float(self.tanh_clims)
        b = float(self.tanh_b)

        mask = torch.le(torch.abs(x), xlim).float()
        mask2 = torch.gt(x, xlim).float() 
        mask3 = torch.lt(x, -xlim).float()

        y = mask*F.tanh(x) + mask2*((c*x).add(b)) + mask3*((c*x).add(-b)) 
        return y


    def tanh_si_inverse(self, x, nl=None):
        '''
        This function requires calling compute_tanh_limits() before using it.
        '''
        ylim = float(self.tanh_ymax) 
        xlim = float(self.tanh_xmax)
        c = float(self.tanh_clims)
        b = float(self.tanh_b)

        y = torch.zeros(x.size())
        yprime = torch.zeros(x.size())
        ones = torch.ones(x.size())
        if self.usecuda:
            ones = ones.cuda()
            y = y.cuda()
            yprime = yprime.cuda()
        ones = Variable(ones)
        y, yprime = Variable(y), Variable(yprime)

        mask = torch.le(torch.abs(x), ylim).float()
        mask2 = torch.gt(x, ylim).float()
        mask3 = torch.lt(x, -ylim).float()

        eps = 1e-20
        tanh_inverse = lambda y: 0.5*((1+y).log() - (1-y).log())
        tanh_inverse_prime = lambda y: 0.5*(1/(1+y).add(eps) + 1/(1-y).add(eps))

        y1 = tanh_inverse(x*mask)
        y2 = x.add(-b)/c 
        y3 = x.add(b)/c 
        y = mask*y1 + mask2*y2 + mask3*y3

        if np.isnan(y1.cpu().data.numpy().sum()):
            pdb.set_trace()
       
        yprime = mask*tanh_inverse_prime(x) + (mask2+mask3)*ones*(1/c)
        return y, yprime


    def leaky_relu(self, x, c):
        mask = torch.gt(x, 0).float()

        y = mask*x + (1-mask)*(c*x)
        return y
    
    def leaky_relu_inverse(self, x, c):
        mask = torch.gt(x,0).float()

        y = mask*x + (1-mask)*(x/c)

        ones = torch.ones(x.size())
        if self.usecuda:
            ones = ones.cuda()
        ones = Variable(ones)

        yprime = mask*ones + (1-mask)*ones*(1/c)
        return y, yprime

    def x3(self, x, nl):
        if nl+1 == self.nlayers:
            return x
        else:
            sign = torch.sign(x)
            return torch.exp((1/3)*x.abs().log())*sign

    def x3_inverse(self, x, nl):
        ones = torch.ones(x.size())
        if self.usecuda:
            ones = ones.cuda()
        ones = Variable(ones)

        if nl+1 == self.nlayers:
            return x, ones
        else:
            eps = 1e-20
            y = x**3
            yprime = 3*(x**2)
            return y, yprime

    def relu_si(self, x, c=3):
        '''
        Smooth & Invertible relu
        '''
        mask = torch.gt(x, 0).float()
        y = mask*x + (1-mask)*(- c*torch.exp((1/c)*(x-1).abs().log()) + c)
        return y
    
    def relu_si_inverse(self, x, c=3):
        '''
        Smooth & Invertible relu inverse
        '''
        mask = torch.gt(x, 0).float() 
        y = mask*x + (1-mask)*(((x - c)/c)**c + 1)

        ones = torch.ones(x.size())
        if self.usecuda:
            ones = ones.cuda()
        ones = Variable(ones)

        yprime = mask*ones + (1-mask)*((x-c)/c)**(c-1)

        return y, yprime


    def compute_density_normal(self, invf, x):
        inv_f = invf(x)
        h = inv_f[0]
        detterm = inv_f[1]
        eps = self.sig_obs

        if self.base_dist == 'factor':
            D = self.D
            mu = torch.matmul(D, self.mu)
            sz = self.sz

            h = h.view(-1, sz)
            eyeK = torch.eye(self.Kdict)
            eyesz = torch.eye(sz)
            if self.cuda:
                eyeK = eyeK.cuda()
                eyesz = eyesz.cuda()
            eyeK, eyesz = Variable(eyeK), Variable(eyesz)

            invmat = torch.inverse(eyeK + torch.matmul(D.t(), D)/(eps**2))
            gausdetterm = -0.5*torch.potrf(invmat).diag().abs().log().sum()

            precmat = (eyesz/(eps**2)) - torch.matmul(torch.matmul(D, invmat), D.t())/(eps**4)

            log_density = -0.5*(torch.matmul((h - mu), precmat)*(h-mu)).sum(1)
            log_density = log_density + detterm + gausdetterm
    
        elif self.base_dist in ['diag_gauss', 'fixed_iso_gauss']:
            sigma = self.sig
            h = h.contiguous().view(-1, self.sz)

            gausdetterm = -sigma.abs().log().sum()

            log_density =  - (((h - self.mu)/(2*sigma.add(eps)**2))*(h - self.mu)).sum(1) + detterm + gausdetterm
        
        elif self.base_dist == 'mixture_diag_gauss':
            mus = self.mus.view(1, self.sz, self.Kdict)
            sigs = self.sigs.view(1, self.sz, self.Kdict)
            h = h.contiguous().view(-1, self.sz, 1)

            gausdetterm = -sigs.abs().log().sum(1)
            gausdetterm = gausdetterm.view(1, self.Kdict)

            log_density = self.logsumexp( - (((h - mus)/(2*sigs.add(eps)**2))*(h - mus)).sum(1) + gausdetterm, dim=1).sum(1) + detterm
        elif self.base_dist == 'mixture_full_gauss':
            h = h.contiguous().view(-1, self.sz).data.cpu().numpy() 
            gmmscores = torch.from_numpy(self.GMM.score_samples(h)).float()
            log_density = [gmmscores, detterm.data.cpu()]

        return log_density, None, inv_f

    def logsumexp(self, h, dim=1):
         m, _ = torch.max(h, dim=dim, keepdim=True)
         first_term = (h - m).exp().sum(dim, keepdim=True).log() 

         return first_term + m 

    def forward_layer_changedim(self, h):
        h = torch.matmul(self.tfmat, h.t())
        h = h + self.tf_b.unsqueeze(1)
        return h.t()

    def inverse_layer_changedim(self, h):
        mat = (self.tfmat)
        grammat = torch.matmul(mat.t(), mat)
        inv_grammat = torch.inverse(grammat)
        pinvmat = torch.matmul(inv_grammat, mat.t())

        temp = h.t() - self.tf_b.unsqueeze(1)
        temp = torch.matmul(pinvmat, temp)

        detterm = -2*torch.potrf(grammat).diag().abs().log().sum()
        return temp.t(), detterm


    def forward_layer_tri_nonlinear(self, h, nl, mats, tritype='lower',
                                    useu='False'):
        if tritype == 'upper':
            mat = torch.triu(mats[nl])
        else:
            mat = torch.tril(mats[nl])

        temp, _ = torch.gesv(h.t(), mat) 
        temp = temp + self.bs[nl].unsqueeze(1)
        if self.useu:
            temp = (temp*self.us[nl].unsqueeze(1)) #+ self.b2s[nl].unsqueeze(1)
        h = temp.t()
        return h

    def inverse_layer_tri_nonlinear(self, h, nl, mats, tritype='lower'):
        
        if tritype == 'upper':
            mat = torch.triu(mats[nl])
        else:
            mat = torch.tril(mats[nl])

        eps = 1e-20
        if self.useu:
            temp = (h)/self.us[nl].add(eps)
        else:
            temp = h
        temp = temp - self.bs[nl].unsqueeze(0)
        temp = torch.matmul(mat, temp.t())
        h = temp.t()

        Wdet = (torch.diag(mat)).abs().log().sum()
        if self.useu:
            udet =  - (self.us[nl].add(eps)).abs().log()
            udet = torch.sum(udet, 1)
            logdet = udet + Wdet
        else:
            logdet = Wdet

        return h, logdet

    def forward_layer_tri_linear(self, h, nl, mats):
        mat = torch.triu(mats[nl]) 
        temp, _ = torch.gesv(h.t(), mat) 

        h = temp.t()
        return h
    
    def inverse_layer_tri_linear(self, h, nl, mats):
        mat = torch.triu(mats[nl])
        h = torch.matmul(mat, h.t()).t()

        Wdet = torch.diag(mat).abs().log().sum()
        return h, Wdet

    def conv_2d(self, H, nl, mode='forward'): 
        '''
        2d convolution via FFT
        '''
        W = self.Ws[nl] 
       
        N, M = H.size(1), H.size(2)

        K = W.size(0)
        
        fft = pfft.Fft2d()
        ifft = pfft.Ifft2d()

        # fft of the filter
        W_rsh = W.unsqueeze(0)
        if (self.forward_pad) | (self.inverse_pad):  
            W_pad = F.pad(W_rsh, (0, M-1, 0, N-1, 0, 0), 'constant', 0)
        else:
            W_pad = F.pad(W_rsh, (0, M-K, 0, N-K, 0, 0), 'constant', 0)

        W_im = torch.zeros(W_pad.size()).cuda()
        if self.usecuda:
            W_im = W_im.cuda()
        W_im = Variable(W_im)

        Wdft_real, Wdft_im = self.dft2([W_pad, W_im], mode='forward')
        #Wfft_real, Wfft_im = fft(W_pad, W_im)
        #W_real, W_i = self.dft2([Wdft_real, Wdft_im], mode='inverse')

        #sm = Wfft_real.sum()
        #sm.backward()
        #print(W.grad)

        # zero-padding, fft of the image, convolution, and cropping
        if mode == 'forward':
            if self.forward_pad:
                H_pad = self.pad_zeros_conv(H)
            else:
                H_pad = H
            H_im = Variable(torch.zeros(H_pad.size()).cuda())
            
            #Hfft_real, Hfft_im  = fft(H_pad, H_im)
            Hdft_real, Hdft_im  = self.dft2([H_pad, H_im])
            
            Ydft_real, Ydft_im = self.complex_mult([Hdft_real, Hdft_im], [Wdft_real, Wdft_im])
            
            #Yreal, Yim  = ifft(Yfft_real, Yfft_im)  
            Yreal, Yim = self.dft2([Ydft_real, Ydft_im], mode='inverse')  

            if not self.forward_pad:
                Yreal = self.crop_images_conv(Yreal)
        elif mode == 'inverse':
            if self.inverse_pad:
                H_pad = self.pad_zeros_deconv(H) 
            else:
                H_pad = H
            H_im = Variable(torch.zeros(H_pad.size()).cuda())

            #Hfft_real, Hfft_im  = fft(H_pad, H_im)
            Hdft_real, Hdft_im  = self.dft2([H_pad, H_im], mode='forward')
            
            #Yfft_real, Yfft_im = self.complex_div([Hfft_real, Hfft_im], [Wfft_real, Wfft_im])
            Ydft_real, Ydft_im = self.complex_div([Hdft_real, Hdft_im], [Wdft_real, Wdft_im])

            #Yreal, Yim  = ifft(Yfft_real, Yfft_im)  
            Yreal, Yim = self.dft2([Ydft_real, Ydft_im], mode='inverse')  

            #Yreal = H_pad * W_pad
            #cropped_image = self.crop_images_deconv(Yreal)

        log_det = - 0.5*(Wdft_real**2 + Wdft_im**2).log().sum()

        return Yreal, log_det

    def pad_zeros_conv(self, H):
        N = self.N
        M = self.M
        K = self.Ws[0].size(0)

        H_rsh = H.contiguous().view(-1, N, M)
        H_pad = F.pad(H_rsh.cuda(), (0, K-1, 0, K-1, 0, 0), 'constant', 0)
        return H_pad

    def complex_matmul(self, A, B):
        Ar, Ai = A
        Br, Bi = B

        mat1 = torch.matmul(Ar, Br)
        mat2 = torch.matmul(Ai, Bi) 
        mat3 = torch.matmul(Ai, Br)
        mat4 = torch.matmul(Ar, Bi)

        return mat1-mat2, mat3+mat4

    def get_dft_mat(self, sz):
        DFT_mat = spl.dft(sz, scale=None)
        DFT_r = torch.from_numpy(np.real(DFT_mat)).float()
        DFT_im = torch.from_numpy(np.imag(DFT_mat)).float()
        if self.usecuda:
            DFT_r = DFT_r.cuda()
            DFT_im = DFT_im.cuda()

        return Variable(DFT_r.unsqueeze(0)), Variable(DFT_im.unsqueeze(0))

    def get_idft_mat(self, sz):
        IDFT_mat = np.conj(spl.dft(sz, scale='n'))
        IDFT_r = torch.from_numpy(np.real(IDFT_mat)).float()
        IDFT_im = torch.from_numpy(np.imag(IDFT_mat)).float()
        if self.usecuda:
            IDFT_r = IDFT_r.cuda()
            IDFT_im = IDFT_im.cuda()

        return Variable(IDFT_r.unsqueeze(0)), Variable(IDFT_im.unsqueeze(0))

    def dft2(self, H, mode='forward'):
        '''
        2d dft, and dft
        '''
        Hr, Hi = H
        
        sz = Hr.size(1) 
        if mode == 'forward':
            Fr, Fi = self.get_dft_mat(sz)
        elif mode == 'inverse':
            Fr, Fi = self.get_idft_mat(sz)
        else:
            raise ValueError('No such mode in this function yo')

        FHlr, FHli = self.complex_matmul([Fr, Fi], [Hr, Hi])
        FHr, FHl = self.complex_matmul([FHlr, FHli], [Fr, Fi]) 
        return FHr, FHl

    
    def pad_zeros_deconv(self, H):
        K = self.K
        pd = int(np.floor(K/2))

        H_pad = F.pad(H, (pd, pd, pd, pd, 0, 0), 'constant', 0)
        return H_pad



    def crop_images_conv(self, H):
        K = self.K 
        pd = int(np.floor(K/2))
        return H[:, pd:-pd, pd:-pd]

    def crop_images_deconv(self, H):
        N = self.N
        M = self.M
        return H[:, :N, :M]


    def complex_mult(self, H, W):
        H_real, H_im = H
        W_real, W_im = W

        Y_real = H_real*W_real - H_im*W_im
        Y_im = H_im*W_real + H_real*W_im

        return Y_real, Y_im

    def complex_div(self, Y, W):
        Y_real, Y_im = Y
        W_real, W_im = W

        eps = 1e-15
        denom = ( (W_real**2) + (W_im**2) ) 
        H_real = (Y_real*W_real + Y_im*W_im)/denom
        H_im = (Y_im*W_real - Y_real*W_im)/denom

        return H_real, H_im

    def get_dirac_filter(self, K):
        '''
        get a dirac-delta filter
        '''
        fltr = torch.zeros(K, K)
        cntr = int(np.floor(K/2))
        fltr[cntr,cntr] = 1
        return fltr

    def generate_data(self, N, base_dist='factor', return_inds=False):
        if base_dist == 'factor':
            L = self.M*self.N
            noise0 = torch.randn(N, self.Kdict)
            noise1 = self.sig_obs*torch.randn(N, L)
            if self.usecuda:
                noise0 = noise0.cuda()
                noise1 = noise1.cuda()
            noise0 = Variable(noise0)
            noise1 = Variable(noise1)

            seed0 = noise0 + self.mu
            seed = torch.matmul(self.D, seed0.t()).t() + noise1
        elif base_dist == 'fixed_iso_gauss':
            try:
                noise = self.sig.data.cpu()*torch.randn(N, self.L1)
                seed = self.mu + Variable(self.tocuda(noise))
            except:
                noise = torch.randn(N, self.L1)
                seed = Variable(self.tocuda(noise))

        elif base_dist == 'diag_gauss': 
            L = self.M

            noise = self.sig.add(self.sig_obs).data.cpu()*torch.randn(N, self.L1)
            if self.usecuda:
                noise = noise.cuda()
            seed = self.mu + Variable(noise)
        elif base_dist == 'mixture_full_gauss':
            seed, inds = self.GMM.sample(N)
            seed = torch.from_numpy(seed).float()
            seed = Variable(self.tocuda(seed))

        elif base_dist == 'mixture_diag_gauss':

            mixind = torch.from_numpy(np.random.randint(0, self.Kdict, N))
            if self.cuda:
                mixind = mixind.cuda()
            
            mumat = self.mus[:, mixind].t()
            sigmat = self.sigs[:, mixind].t()
            noise = sigmat.add(self.sig_obs).data.cpu()*torch.randn(N, self.L1)
            if self.usecuda:
                noise = noise.cuda()
            seed = mumat + Variable(noise)
        elif base_dist == 'full_gauss':
            noise = torch.randn(N, self.sz)
            if self.usecuda:
                noise = noise.cuda()
            noise = Variable(noise)

            noise = torch.matmul(self.W, noise.t()).t()
            seed = noise + self.mu

        if return_inds == True:
            return self.forward(seed), seed, inds
        else:
            return self.forward(seed), seed
    

    def build_dict(self):
        L = self.M
        
        dct = torch.zeros(self.Kdict, L, L) 
        for k in range(self.Kdict):
            roc = np.random.rand() < 0.5 
            num = int(3 + 5*np.random.rand())
            if roc == 0:
                dct[k, 3:8, num] = 1
            else:
                dct[k, num, 3:8] = 1
        self.D = nn.Parameter(dct.view(self.Kdict, -1).t())

    def gaussian_pdf(self, x, mu, sig, mode='iso'):
        ndim = mu.size(-1)
        pi = math.pi
        logdet = - math.log(2*pi)*(ndim/2) - math.log(abs(sig))*ndim
        logpdf = - (((x-mu)*(x-mu))/(2*(sig**2)) ).sum(-1) + (logdet)
        return logpdf

    def klqp(self, xtrain, xgen, pdf_computer=None):

        ndim = self.M**2
        if pdf_computer == None:
            kdes = self.compute_kde(xgen, Variable(xtrain), 0.1)
            logpdf = -self.logsumexp(kdes, dim=0).add(-math.log(ndim))
            outs = None
        else:
            outs = pdf_computer(xgen.view(-1, ndim))
            logpdf = pdf_computer.criterion(outs[0], xgen.view(-1, ndim), outs[1], outs[2])

        logdens, _, _ = self.compute_density_normal(invf=self.inverse_forward, x=xgen)

        logdens2, _, inv_f = self.compute_density_normal(invf=self.inverse_forward, x=Variable(xtrain))

        cost = 0*logpdf.mean() + 0*logdens.mean() - logdens2.mean()
        return cost, inv_f, outs

    def jsdiv(self, xtrain, xgen): 
        ndim = xtrain.size(0)
        sig = 0.1 
        
        kdes_p = self.compute_kde(Variable(xtrain), Variable(xtrain), sig)
        kdes_p = self.logsumexp(kdes_p, dim=0).add(-math.log(ndim))
        log_dens_p, _, _ = self.compute_density_normal(invf=self.inverse_forward, x=Variable(xtrain.cuda()))

        catmat = torch.cat([kdes_p.t(), log_dens_p.unsqueeze(1)], dim=1)
        term1 = - self.logsumexp(catmat, dim=1).mean()

        kdes_q = self.compute_kde(xgen, Variable(xtrain), sig) 
        kdes_q = self.logsumexp(kdes_q, dim=0).add(-math.log(ndim))

        log_dens_q, _, _ = self.compute_density_normal(invf=self.inverse_forward, x=xgen)
        catmat2 = torch.cat([kdes_q.t(), log_dens_q.unsqueeze(1)], dim=1)

        term2 = log_dens_q.mean()
        term3 = -self.logsumexp(catmat2, dim=1).mean()

        cost = term1 + term2 + term3
        return cost


        

    def compute_kde(self, x, xtrain, sig):

        N = x.size(0)
        ndim = x.size(1)
        x = x.view(x.size(0), -1)
        xtrain = xtrain.view(xtrain.size(0), -1)

        #test = self.gaussian_pdf(x[0], xtrain[0], sig=sig)

        x = x.unsqueeze(0)
        xtrain = xtrain.unsqueeze(1)
        kdes = self.gaussian_pdf(x, mu=(xtrain), sig=sig)
        return kdes

    def tocuda(self, var):
        if self.usecuda:
            var = var.cuda()
        return var

    def NF_trainer(self, train_loader, vis, EP=150, mode='NF_base', config_num=1, arguments=None):
    
        lr = 1e-3
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, threshold=1e-4, patience=100, factor=0.8)

        all_costs = torch.zeros(EP)
        nbatches = 100
        for ep in range(EP):
            for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                
                data = data.squeeze()
                self.zero_grad()
                
                if mode == 'NF_base':
                    inv_f = self.inverse_forward(Variable(self.tocuda(data)))
                    # logdens, _, inv_f = self.compute_density_normal(invf=self.inverse_forward, x=Variable(data.cuda()))
                    cost = ((inv_f[-1] - Variable(self.tocuda(data)))**2).mean()
                elif mode == 'NF_ML':
                    logdens, _, inv_f = NF.compute_density_normal(invf=NF.inverse_forward, x=Variable(data))
                    cost = (torch.abs(inv_f[-1] - Variable(data))).sum()  - logdens.mean() 
                                    
                cost.backward()

                opt.step()

                scheduler.step(cost.data[0], epoch=ep)

                print('EP [{}/{}], batch [{}/{}],  Cost is {}, Cost ML is {}, Learning rate is {}, Config num {}'.format(ep+1, EP, i+1, len(train_loader), cost.data[0], 0, opt.param_groups[0]['lr'], config_num))
            all_costs[ep] = np.abs(cost.data[0])
            if ep % 5 == 0 and vis is not None:
                vis_plot_results(self, inv_f, data, vis, all_costs, ep, arguments)

        
    def train_base(self, train_loader):
        # get hhats for all batches
        all_hhats = []
        nbatches = 100
        for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
            data = data.squeeze()
            inv_f = self.inverse_forward(Variable(self.tocuda(data)))
            all_hhats.append(inv_f[0].contiguous().view(-1, self.Ks[0]))
        all_hhats = torch.cat(all_hhats, dim=0)

        # train base dist
        if self.base_dist == 'full_gauss': 
            mean = all_hhats.mean(dim=0, keepdim=True)
            centered_hhats = all_hhats - mean
            cov = torch.matmul(centered_hhats.t(), centered_hhats)/(centered_hhats.size(0) - 1)
            W = torch.potrf(cov)
            
            self.mu = mean
            self.W = W
        elif self.base_dist == 'fixed_iso_gauss':
            self.mu = all_hhats.mean(0)
            self.sig = all_hhats.std(0)
            
        elif self.base_dist == 'mixture_full_gauss':
            data = all_hhats.data.cpu().numpy()[:3000]
            self.GMM.fit(data)



def adversarial_trainer_step(ep, data, generator, discriminator, opts, arguments):
   
    imgs = True if isinstance(generator, netg_dcgan) else False
    ft, tar = data
    
    if not imgs:
        L2 = generator.L2
        tar = tar.contiguous().view(-1, L2)
    ft, tar = Variable(ft), Variable(tar)

    #use random noise
    if not imgs:
        L1 = generator.Ks[0]
        ft = torch.randn(tar.size(0), L1)
    else:
        L1 = 100
        ft = torch.randn(tar.size(0), L1).view(-1, L1, 1, 1)
    if arguments.cuda:
        ft = ft.cuda()
    ft = Variable(ft)
    

    false, true = 0, 1
    criterion = nn.BCELoss()
    optimizerD, optimizerG = opts

    if arguments.cuda:
        ft = ft.cuda()
        tar = tar.cuda()
        
    if ep < 20 or ep % 500 == 0:
        Diters = 5
    else:
        Diters = 1

    for disc_ep in range(Diters):
        # discriminator gradient
        discriminator.zero_grad()
        out_d = discriminator.forward(tar)
        sz = out_d.size(0)
        labels = Variable(torch.ones(sz)).squeeze().float()
        if arguments.cuda:
            labels = labels.cuda()
        err_D_real = criterion(out_d.squeeze(), labels*true)
        err_D_real.backward()

        out_g = generator.forward(ft)
        if not imgs:
            out_g = out_g.contiguous().view(-1, L2) 
        out_d_g = discriminator.forward(out_g.detach())
        err_D_fake = criterion(out_d_g.squeeze(), labels*false) 
        err_D_fake.backward()

        err_D = err_D_real + err_D_fake
        optimizerD.step()

    for gent_ep in range(1):
        # generator gradient
        generator.zero_grad()
        out_d_g = discriminator.forward(out_g)
        err_G = criterion(out_d_g.squeeze(), labels*true)
        err_G.backward(retain_variables=True)

        optimizerG.step()

    return out_g, out_d_g.mean(0)

def adversarial_trainer(EP, train_loader, generator, discriminator, arguments, config_num=0, vis=None):
    '''
    Wrapper function for adversarial_trainer_step
    '''
    lr = 1e-4
    optG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    nbatches = 1200
    all_costs = np.zeros(EP)
    for ep in range(EP):
        for i, (dt, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
            gen_data, cost = adversarial_trainer_step(ep, [None, dt], 
                                                      generator, discriminator, 
                                                      [optD, optG], arguments)
            
            print('EP [{}/{}], batch [{}/{}],  Cost is {}, Learning rate is {}, config num is {}'.format(ep+1, EP, i+1, 
                                                                                                         len(train_loader), 
                                                                                                         cost.data.cpu().numpy()[0], 
                                                                                                         optG.param_groups[0]['lr'],
                                                                                                         config_num))
        all_costs[i] = cost.data.cpu()[0]
        

        imgs = isinstance(generator, netg_dcgan)
        if not imgs:
            if vis is not None and ep % 5 == 0:
                if arguments.data is not 'toy_example': 
                    N = 64
                    M = generator.M 
                    images = ut.collate_images(gen_data, N=N, L=M)
                    sz = 800
                    opts={'width':sz, 'height':sz}
                    opts['title'] = 'Generated Images'
                    hm1 = vis.heatmap(images, opts=opts, win='generated')
                else: 
                    opts = {}
                    opts['title'] = 'Training Data'
                    vis.scatter(dt.cpu(), win='x', opts=opts, name='x')

                    opts['title'] = 'Generated Data'
                    vis.scatter(gen_data.data.cpu(), win='xgen', opts=opts)

                    #opts['title'] = 'Training Data Histogram'
                    #vis.mesh(data.cpu(), win='x_heat', opts=opts)

                    inv_f = generator.inverse_forward(Variable(generator.tocuda(dt)))
                    opts['title'] = 'Mappings to Base Distribution Space'
                    vis.scatter(inv_f[0].data.cpu(), win='hhat', opts=opts)

                    opts['title'] = 'Mapping Back to observation space'
                    vis.scatter(inv_f[-1].data.cpu(), win='xhat', opts=opts)

                    opts['title'] = 'Epoch vs Cost'
                    opts['xtype'], opts['ytype'] = 'linear', 'log'
                    vis.line(Y=all_costs[:ep+1], X=torch.arange(1,ep+2), opts=opts, win='cost')

                    
            
        else:
            opts = {'title':'GAN generated data config {}'.format(config_num)}
            vis.images(0.5 + 0.5*gen_data.data.cpu(), opts=opts, win='NF_config_{}'.format(config_num))

    
class Normalizing_Flow_Conv(Normalizing_Flow_utils):
    def __init__(self, N, M, L1=100, K=5, c=0.01, cp=1, out='relu', cuda=False, check_fit=True, easy_init=False, forward_pad=False, inverse_pad=True, base_dist='factor', Kdict=50, sig_obs=1e-2, generate_mode=False, **kwargs):
        super(Normalizing_Flow_Conv, self).__init__(c, useu=False, cuda=cuda)

        self.nlayers = kwargs['nlayers']
        self.nfulllayers = kwargs['nfulllayers']
        self.nlulayers = kwargs['nlulayers']
        self.arguments = kwargs['arguments']
        self.out = out
        self.check_fit = True
        self.c = c
        self.nonlin = self.tanh_si
        self.nonlininv = self.tanh_si_inverse
        self.forward_pad = forward_pad
        self.inverse_pad = inverse_pad
        self.base_dist = base_dist
        self.Kdict = Kdict 
        self.sig_obs = sig_obs
        self.generate_mode = generate_mode

        self.compute_tanh_limits(ymax=0.99)
        self.compute_sigm_limits(eps=1e-4)
        self.K = K 
        self.N = N
        self.M = M 
        pd = (K-1)*self.nlayers
        self.pd = pd
        self.sz = sz = self.L1 = L1 
        self.L2 = self.M**2

        if self.nlulayers > 0:
            ceye=1
            cp = 0
            Us =[nn.Parameter(ceye*torch.eye(sz) + cp*torch.triu(torch.randn(sz, sz))) for nl in range(self.nlulayers)]
            self.Us = nn.ParameterList(Us) 

            Ls =[nn.Parameter(ceye*torch.eye(sz) + cp*torch.tril(torch.randn(sz, sz))) for nl in range(self.nlulayers)]
            self.Ls = nn.ParameterList(Ls) 
   
            bs = [nn.Parameter((cp)*torch.randn(sz)) for nl in range(self.nlulayers)]
            self.bs = nn.ParameterList(bs)

        if self.nfulllayers > 0:
            ceye=1
            cp = 0
            Ws =[nn.Parameter(ceye*torch.eye(sz) + cp*torch.randn(sz, sz)) for nl in range(self.nfulllayers)]
            self.Ws = nn.ParameterList(Ws) 

            bs = [nn.Parameter((cp)*torch.randn(sz)) for nl in range(self.nfulllayers)]
            self.bs = nn.ParameterList(bs)


        cp = 0.1
        self.tfmat = nn.Parameter(cp*torch.randn(self.L2, self.sz))
        self.tf_b = nn.Parameter(0*torch.randn(self.L2))

        if self.base_dist == 'factor':
            if generate_mode:
                self.build_dict()
            else:
                self.D = nn.Parameter(torch.randn(sz, Kdict))
            self.mu = nn.Parameter(torch.zeros(Kdict))
        elif self.base_dist == 'diag_gauss':
            self.mu = nn.Parameter(torch.randn(sz))
            self.sig = nn.Parameter(0.1 + torch.rand(sz))
        elif self.base_dist == 'mixture_diag_gauss':
            masks = torch.bernoulli(0.01*torch.ones(sz,Kdict))
            self.mus = nn.Parameter(torch.rand(sz, Kdict)*masks)
            self.sigs = nn.Parameter(0.1*torch.rand(sz, Kdict)) 
        elif self.base_dist == 'fixed_iso_gauss':
            mu = torch.zeros(sz)
            sig = torch.ones(sz)
            mu, sig = mu.cuda(), sig.cuda()
            self.mu, self.sig = Variable(mu), Variable(sig)
 


       

    def pretrain_basedist(self, loader, vis): 
        '''
        pretrain the dictionary model for the base distribution
        '''
        if self.cuda:
            identity = lambda x: (x, Variable(torch.ones(x.size(0)).cuda()))
        else:
            identity = lambda x:x, Variable(torch.ones(x.size(0)))

        opt = torch.optim.Adam([self.D, self.mu], lr=1e-3)
        EP = 500
        for ep in range(EP):
            for i, (data, _) in enumerate(it.islice(loader, 0, 1, 1)):
                
                data = data.squeeze()
                if self.cuda:
                    data = data.cuda()
                data = F.pad(data, (0, self.pd, 0, self.pd, 0, 0), 'constant', 0)
                # cost 1 
                self.zero_grad()
                logdens, _, inv_f = self.compute_density_normal(invf=identity, 
                                            x=data)
                cost = - logdens.mean()

                cost.backward()
                opt.step()

                print('Pretraining Base Distibution: EP [{}/{}], batch [{}/{}],  Cost is {}, Learning rate is {}'.format(ep+1, EP, i+1, len(loader), cost.data[0], opt.param_groups[0]['lr']))

            if ep % 50 == 0: 
                _ , seed = self.generate_data(100, base_dist='factor') 

            N = 64
            M = self.M
            if len(seed.size()) == 2:
                seed = seed.view(-1, M, M)
            seed_images = ut.collate_images(seed, N=N, L=M)

            sz = 800
            opts={'width':sz, 'height':sz}
            opts['title'] = 'Draws from the base distribution'
            hm3 = vis.heatmap(seed_images, opts=opts, win='shat_pret')

    
    def inverse_forward(self, H):
        x_org = H
        logdetsum = torch.zeros(H.size(0))
        if self.usecuda:
            logdetsum = logdetsum.cuda()
        logdetsum = Variable(logdetsum)

        H, _ = self.sigmoid_si_inverse(H)
        
        h = H.view(-1, (self.M + self.pd)**2)

        h, det_cd = self.inverse_layer_changedim(h)
        logdetsum = det_cd + logdetsum
        for nl in range(self.nfulllayers-1, -1, -1):
            if 1:#nl + 1 < self.nfulllayers + self.nlayers:
                h, logdet_t = self.nonlininv(h) 
                logdet_t = logdet_t.abs().log().sum(1)
            else:
                logdet_t = torch.zeros(h.size(0))
                if self.usecuda:
                    logdet_t = Variable(logdet_t.cuda())
                else: 
                    logdet_t = Variable(logdet_t)

            if self.nlulayers > 0:
                h, logdet_L = self.inverse_layer_tri_nonlinear(h, nl, mats=self.Ls)
                h, logdet_U = self.inverse_layer_tri_linear(h, nl, mats=self.Us)
                logdetsum = logdetsum + logdet_U + logdet_L + logdet_t
            else:
                h, logdet_W = self.inverse_layer_full(h, nl)
                logdetsum = logdetsum + logdet_W + logdet_t

        if self.check_fit:
            xhat= self.forward(h)
            mismatch = (x_org - xhat).abs().sum()
            print(mismatch.data[0])
        return h, logdetsum, xhat

    

    def forward(self, h):
        #h = h.view(-1, self.sz)
        if self.nlulayers > 0:
            for nl in range(0, self.nlulayers):
                h = self.forward_layer_tri_linear(h, nl, mats=self.Us)
                h = self.forward_layer_tri_nonlinear(h, nl, mats=self.Ls)
                h = self.nonlin(h) 
        else: 
            for nl in range(0, self.nfulllayers):
                h = self.forward_layer_full(h, nl)
                h = self.nonlin(h) 


        h = self.forward_layer_changedim(h)

        #H = self.leaky_relu(H, 0.01)
        H = self.sigmoid_si(h)

        H = H.contiguous().view(-1, self.M, self.M)
        return H

    
class netD(nn.Module):
    def __init__(self, L, K, was=False, **kwargs):
        super(netD, self).__init__()
        self.L = L
        self.K = K
        self.was = was

        self.l1 = nn.Linear(self.L, self.K, bias=True)
        self.l2 = nn.Linear(self.K, 1, bias=True)

    def forward(self, inp):

        h1 = F.leaky_relu((self.l1(inp)))
        if self.was:
            output = self.l2(h1)
        else:
            output = F.sigmoid(self.l2(h1))

        return output

class netG(nn.Module):
    def __init__(self, L2, Ks, M=28):
        super(netG, self).__init__()
        self.L2 = L2
        self.M = M
        self.Ks = Ks
        self.base_dist = 'iso_fixed_gauss'

        self.l1 = nn.Linear(self.Ks[0], self.Ks[1], bias=True)
        c = 0.1
        nn.init.uniform(self.l1.weight, a=-c, b=c)
        nn.init.uniform(self.l1.bias, a=-c, b=c)


        self.l2 = nn.Linear(self.Ks[1], self.L2, bias=True)
        nn.init.uniform(self.l2.weight, a=-c, b=c)
        nn.init.uniform(self.l2.bias, a=-c, b=c)



    def forward(self, inp): 
        inp = inp.view(-1, self.Ks[0])

        h1 = F.tanh(self.l1(inp))
        output = F.sigmoid(self.l2(h1))

        return output
    
    def generate_data(self, N, base_dist='fixed_iso_gauss'):
        seed = torch.randn(N, self.Ks[0]) 
        if self is self.cuda(): 
            seed = seed.cuda()
        seed = Variable(seed)
        gen_data = self.forward(seed)
        return gen_data, seed


class NF_changedim(Normalizing_Flow_utils):
    def __init__(self, N, M, c=0.01, cp=1, cuda=False, check_fit=True, forward_pad=False, inverse_pad=True, base_dist='factor', out='sigmsi', 
                 Kdict=30, Ks=[40, 600], sig_obs=1e-2, base_inits=10, **kwargs):
        super(NF_changedim, self).__init__(c, useu=False, cuda=cuda)

        self.arguments = kwargs['arguments']
        self.check_fit = True
        self.c = c
        self.nonlin = self.tanh_si
        self.nonlininv = self.tanh_si_inverse
        self.forward_pad = forward_pad
        self.inverse_pad = inverse_pad
        self.base_dist = base_dist
        self.Kdict = Kdict 
        self.sig_obs = sig_obs

        self.compute_tanh_limits(ymax=0.99)
        self.compute_sigm_limits(eps=1e-4)
        self.Ks = Ks
        self.nlayers = len(self.Ks)
        self.out = out

        self.N = N
        self.M = M 
        self.sz = sz = self.L1 = self.Ks[0]
        if 'L2' in kwargs.keys():
            self.L2 = kwargs['L2']
        else:
            self.L2 = M**2

        cp = 1
        tfmats = [nn.Parameter(cp*torch.randn(K2, K1)) for K1, K2 in zip(self.Ks[:-1], self.Ks[1:])]
        tfmats.append(nn.Parameter(cp*torch.bernoulli(1*torch.ones(self.L2, self.Ks[-1]))*torch.randn(self.L2, self.Ks[-1])))
        self.tfmats = nn.ParameterList(tfmats)
        
        bias_sizes = self.Ks
        bias_sizes.append(self.L2)
        biases = [nn.Parameter(0*torch.randn(bsz)) for bsz in bias_sizes[1:]]
        self.biases = nn.ParameterList(biases)

        if self.base_dist == 'fixed_iso_gauss':
            mu = torch.zeros(sz)
            sig = torch.ones(sz)
            if cuda:
                mu, sig = mu.cuda(), sig.cuda()
            self.mu, self.sig = Variable(mu), Variable(sig)
        elif self.base_dist == 'diag_gauss':
            self.mu = nn.Parameter(0*torch.randn(sz))
            self.sig = nn.Parameter(0.1 + 1*torch.rand(sz))
        elif self.base_dist == 'full_gauss':
            self.mu = Variable(torch.randn(sz))
            self.W = Variable(torch.randn(sz,sz))
        elif self.base_dist == 'mixture_diag_gauss':
            masks = torch.bernoulli(0.01*torch.ones(sz,Kdict))
            self.mus = nn.Parameter(torch.rand(sz, Kdict))
            self.sigs = nn.Parameter(10*torch.ones(sz, Kdict)) 
        elif self.base_dist == 'mixture_full_gauss':
            self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, n_init=base_inits, max_iter = 200)
        else:
            raise(ValueError, 'there is no such base distribution in my book')


    def forward_layer_changedim(self, h, mat, bias):
        h = torch.matmul(mat, h.t())
        h = h + bias.unsqueeze(1)
        return h.t()

    def inverse_layer_changedim(self, h, mat, bias):
        grammat = torch.matmul(mat.t(), mat)
        inv_grammat = torch.inverse(grammat)
        pinvmat = torch.matmul(inv_grammat, mat.t())

        temp = h - bias.unsqueeze(0)
        temp = torch.matmul(pinvmat, temp.t())

        detterm = -1*torch.potrf(grammat).diag().abs().log().sum() # coef was one
        return temp.t(), detterm

    def inverse_forward(self, H):
        x_org = H
        logdetsum = torch.zeros(H.size(0))
        if self.usecuda:
            logdetsum = logdetsum.cuda()
        logdetsum = Variable(logdetsum)

        if self.out == 'sigmsi':
            H, _ = self.sigmoid_si_inverse(H)

        h = H.view(-1, self.L2)

        for nl in range(self.nlayers-1, -1, -1):
            if nl + 1 < self.nlayers:
                h, logdet_t = self.nonlininv(h) 
                logdet_t = logdet_t.abs().log().sum(1)
            else:
                logdet_t = torch.zeros(h.size(0))
                if self.usecuda:
                    logdet_t = Variable(logdet_t.cuda())
                else: 
                    logdet_t = Variable(logdet_t)

            h, logdet_c = self.inverse_layer_changedim(h, mat=self.tfmats[nl],
                                                       bias=self.biases[nl])
            logdetsum = logdetsum + logdet_c + logdet_t

        x_org_shape = x_org.size()

        if self.check_fit:
            xhat= self.forward(h)
            xhat = xhat.contiguous().view(x_org_shape) 
            mismatch = (x_org - xhat).abs().sum()
            print(mismatch.data[0])
        return h, logdetsum, xhat
    

    def forward(self, h):
        #h = h.view(-1, self.sz)
        for nl in range(0, self.nlayers):
            h = self.forward_layer_changedim(h, mat=self.tfmats[nl],
                                             bias=self.biases[nl])
            if nl + 1 < self.nlayers: 
                h = self.nonlin(h) 

        if self.out == 'sigmsi':
            H = self.sigmoid_si(h)
        else:
            H = h

        #H = H.contiguous().view(-1, self.M, self.M)
        return H


class netG_LU(Normalizing_Flow_utils):

    def __init__(self, N, M, L1=100, K=5, c=0.01, cp=1, out='relu', cuda=False, check_fit=True, easy_init=False, forward_pad=False, inverse_pad=True, base_dist='factor', Kdict=50, sig_obs=1e-2, generate_mode=False, **kwargs):
        super(netG_LU, self).__init__(c, useu=False, cuda=cuda)

        self.nlayers = kwargs['nlayers']
        self.nfulllayers = kwargs['nfulllayers']
        self.nlulayers = kwargs['nlulayers']
        self.arguments = kwargs['arguments']
        self.out = out
        self.check_fit = True
        self.c = c
        self.nonlin = self.tanh_si
        self.nonlininv = self.tanh_si_inverse
        self.forward_pad = forward_pad
        self.inverse_pad = inverse_pad
        self.base_dist = base_dist
        self.Kdict = Kdict 
        self.sig_obs = sig_obs
        self.generate_mode = generate_mode

        self.compute_tanh_limits(ymax=0.99)
        self.compute_sigm_limits(eps=1e-4)
        self.K = K 
        self.N = N
        self.M = M 
        pd = (K-1)*self.nlayers
        self.pd = pd
        self.sz = sz = self.L1 = L1 
        self.L2 = self.M**2

        if self.nlulayers > 0:
            ceye=1
            cp = 0
            Us =[nn.Parameter(ceye*torch.eye(sz) + cp*torch.triu(torch.randn(sz, sz))) for nl in range(self.nlulayers)]
            self.Us = nn.ParameterList(Us) 

            Ls =[nn.Parameter(ceye*torch.eye(sz) + cp*torch.tril(torch.randn(sz, sz))) for nl in range(self.nlulayers)]
            self.Ls = nn.ParameterList(Ls) 
   
            bs = [nn.Parameter((cp)*torch.randn(sz)) for nl in range(self.nlulayers)]
            self.bs = nn.ParameterList(bs)

        if self.nfulllayers > 0:
            ceye=1
            cp = 0
            Ws =[nn.Parameter(ceye*torch.eye(sz) + cp*torch.randn(sz, sz)) for nl in range(self.nfulllayers)]
            self.Ws = nn.ParameterList(Ws) 

            bs = [nn.Parameter((cp)*torch.randn(sz)) for nl in range(self.nfulllayers)]
            self.bs = nn.ParameterList(bs)



        cp = 0.1
        self.tfmat = nn.Parameter(cp*torch.randn(self.L2, self.sz))
        self.tf_b = nn.Parameter(0*torch.randn(self.L2))

        if self.base_dist == 'fixed_iso_gauss':
            mu = torch.zeros(sz)
            sig = torch.ones(sz)
            mu, sig = mu.cuda(), sig.cuda()
            self.mu, self.sig = Variable(mu), Variable(sig)

    def forward_layer_full(self, h, nl):
        h = torch.matmul(self.Ws[nl], h.t())
        h = h + self.bs[nl].unsqueeze(1) 
        return h.t()

    def inverse_layer_full(self, h, nl):
        h = h - self.bs[nl].unsqueeze(0)
        h, _ = torch.gesv(h.t(), self.Ws[nl])

        detterm = 0 #-torch.potrf(self.Ws[nl]).diag().abs().log().sum()
        return h.t(), detterm

    def forward_layer_tri_nonlinear(self, h, nl, mats, tritype='lower',
                                    useu='False'):
        if tritype == 'upper':
            mat = torch.triu(mats[nl])
        else:
            mat = torch.tril(mats[nl])

        temp = torch.matmul(mat, h.t())
        temp = temp + self.bs[nl].unsqueeze(1)
        h = temp.t()
        return h

    def forward_layer_tri_linear(self, h, nl, mats):
        mat = torch.triu(mats[nl]) 
        temp = torch.matmul(mat, h.t())

        h = temp.t()
        return h



    def inverse_layer_tri_nonlinear(self, h, nl, mats, tritype='lower'):
        
        if tritype == 'upper':
            mat = torch.triu(mats[nl])
        else:
            mat = torch.tril(mats[nl])

        temp = h
        temp = temp - self.bs[nl].unsqueeze(0)
        temp, _ = torch.gesv(temp.t(), mat)
        h = temp.t()

        Wdet = -(torch.diag(mat)).abs().log().sum()
        logdet = Wdet

        return h, logdet

    def inverse_layer_tri_linear(self, h, nl, mats):
        mat = torch.triu(mats[nl])
        temp, _ = torch.gesv(h.t(), mat)
        h = temp.t()

        Wdet = -torch.diag(mat).abs().log().sum()
        return h, Wdet

    def inverse_forward(self, H):
        x_org = H
        logdetsum = torch.zeros(H.size(0))
        if self.usecuda:
            logdetsum = logdetsum.cuda()
        logdetsum = Variable(logdetsum)

        H, _ = self.sigmoid_si_inverse(H)
        
        h = H.view(-1, (self.M + self.pd)**2)

        h, det_cd = self.inverse_layer_changedim(h)
        logdetsum = det_cd + logdetsum
        for nl in range(self.nfulllayers-1, -1, -1):
            if 1:#nl + 1 < self.nfulllayers + self.nlayers:
                h, logdet_t = self.nonlininv(h) 
                logdet_t = logdet_t.abs().log().sum(1)
            else:
                logdet_t = torch.zeros(h.size(0))
                if self.usecuda:
                    logdet_t = Variable(logdet_t.cuda())
                else: 
                    logdet_t = Variable(logdet_t)

            if self.nlulayers > 0:
                h, logdet_L = self.inverse_layer_tri_nonlinear(h, nl, mats=self.Ls)
                h, logdet_U = self.inverse_layer_tri_linear(h, nl, mats=self.Us)
                logdetsum = logdetsum + logdet_U + logdet_L + logdet_t
            else:
                h, logdet_W = self.inverse_layer_full(h, nl)
                logdetsum = logdetsum + logdet_W + logdet_t

        if self.check_fit:
            xhat= self.forward(h)
            mismatch = (x_org - xhat).abs().sum()
            print(mismatch.data[0])
        return h, logdetsum, xhat

    


    def forward(self, h):
        #h = h.view(-1, self.sz)
        if self.nlulayers > 0:
            for nl in range(0, self.nlulayers):
                h = self.forward_layer_tri_linear(h, nl, mats=self.Us)
                h = self.forward_layer_tri_nonlinear(h, nl, mats=self.Ls)
                h = self.nonlin(h) 
        else: 
            for nl in range(0, self.nfulllayers):
                h = self.forward_layer_full(h, nl)
                h = self.nonlin(h) 


        h = self.forward_layer_changedim(h)

        #H = self.leaky_relu(H, 0.01)
        H = self.sigmoid_si(h)

        H = H.contiguous().view(-1, self.M, self.M)
        return H

class VAE(nn.Module):
    def __init__(self, L1, L2, Ks, M):
        super(VAE, self).__init__()

        self.L1 = L1
        self.L2 = L2
        self.Ks = Ks
        self.M = M
        self.base_dist = 'fixed_iso_gauss'

        self.fc1 = nn.Linear(self.L1, self.Ks[1])
        #initializationhelper(self.fc1, 'relu')

        self.fc21 = nn.Linear(self.Ks[1], self.Ks[0])
        #initializationhelper(self.fc21, 'relu')

        self.fc22 = nn.Linear(self.Ks[1], self.Ks[0])
        #initializationhelper(self.fc22, 'relu')

        self.fc3 = nn.Linear(self.Ks[0], self.Ks[1])
        #initializationhelper(self.fc3, 'relu')

        self.fc4 = nn.Linear(self.Ks[1], self.L2)
        #initializationhelper(self.fc4, 'relu')



    def encode(self, x):
        h1 = F.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        z1 = F.tanh(self.fc3(z))
        return F.sigmoid(self.fc4(z1))

    def forward(self, inp):
        #if not (type(inp) == Variable):
        #    inp = Variable(inp[0])

        mu, logvar = self.encode(inp)
        h = self.reparameterize(mu, logvar)

        print('mean of mu {} variance of mu {}'.format(torch.mean(h).data[0], torch.var(h).data[0]))
        return self.decode(h), mu, logvar, h

    def criterion(self, recon_x, x, mu, logvar):
        eps = 1e-20
        #criterion = lambda lam, tar: torch.mean(-tar*torch.log(lam+eps) + lam)
        #crt = lambda xhat, tar: torch.sum(((xhat - tar)**2 ), 1)
        mask = torch.ge(recon_x, 1).float()
        mask2 = torch.le(recon_x, 0).float()
        recon_x = mask*(1-eps) + (1-mask)*recon_x
        recon_x = mask2*eps + (1-mask)*recon_x
        crt = lambda xhat, tar: -torch.sum(tar*torch.log(xhat+eps) + (1-tar)*torch.log(1-xhat+eps), 1)

        BCE = crt(recon_x, x)
        v = 1
        KLD = -0.5 * torch.sum(1 + logvar - ((mu.pow(2) + logvar.exp())/v), 1)
        # Normalise by same number of elements as in reconstruction
        #KLD = KLD /(x.size(0) * x.size(1))
        return BCE + KLD

    def generate_data(self, N, base_dist='fixed_iso_gauss'):
        seed = torch.randn(N, self.Ks[0]) 
        if self is self.cuda(): 
            seed = seed.cuda()
        seed = Variable(seed)
        gen_data = self.decode(seed)
        return gen_data, seed

    def VAE_trainer(self, cuda, train_loader, 
                    EP = 400,
                    vis = None, config_num=0, **kwargs):

        if hasattr(kwargs, 'optimizer'):
            optimizer = kwargs['optimizer']
        else:
            optimizer = 'Adam'

        self.train(mode=True)

        L1 = self.L1
        L2 = self.L2
        Ks = self.Ks

        lr = 5e-4
        if optimizer == 'Adam':
            optimizerG = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        elif optimizer == 'RMSprop':
            optimizerG = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            optimizerG = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'LBFGS':
            optimizerG = optim.LBFGS(self.parameters(), lr=lr)

        nbatches = 1300
        for ep in range(EP):
            for i, (tar, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                if cuda:
                    tar = tar.cuda()

                tar = tar.view(-1, L2)
                tar = Variable(tar)

                # generator gradient
                self.zero_grad()
                out_g, mu, logvar, h = self.forward(tar)
                err_G = self.criterion(out_g, tar, mu, logvar)
                err_G = err_G.mean(0)

                err_G.backward()

                # step 
                optimizerG.step()
                print('EP [{}/{}], error = {}, batch = [{}/{}], config num {}'.format(ep+1, EP, err_G.data[0], i+1, len(train_loader), config_num))
                                                                       
            # visdom plots
            # generate samples 
            gen_data, seed = self.generate_data(100, cuda)

            if ep % 1 == 0:
                if 0:
                    
                    N = 64
                    sz = 800
                    opts={'width':sz, 'height':sz, 'xmax':1.5}
                    gen_images = ut.collate_images(out_g, N=N)
                    opts['title'] = 'VAE Approximations'
                    vis.heatmap(gen_images, opts=opts, win='vae_approximations')
                    
                    input_images = ut.collate_images(tar, N) 
                    opts['title'] = 'VAE Input images'
                    vis.heatmap(input_images, opts=opts, win='vae_x')

                    gen_images = ut.collate_images(gen_data, N)
                    opts['title'] = 'VAE Generated Images'
                    vis.heatmap(gen_images, opts=opts, win='vae_gen_data')
                else: 
                    N = 64
                    sz = 800
                    tar = tar.view(-1, 3, 64, 64)

                    opts={'width':sz, 'height':sz, 'xmax':1.5}
                    opts['title'] = 'VAE Approximations'
                    vis.images(0.5*out_g.data.cpu() + 0.5, opts=opts, win='vae_approximations')
                    opts['title'] = 'VAE Input images'
                    vis.images(tar.data.cpu()*0.5 + 0.5, opts=opts, win='vae_x')

                    opts['title'] = 'VAE Generated Images'
                    vis.images(gen_data.data.cpu()*0.5 + 0.5, opts=opts, win='vae_gen_data')



class conv_VAE(VAE):
    def __init__(self, L1, L2, Ks, M, num_gpus=4):
        super(conv_VAE, self).__init__(L1, L2, Ks, M)

        self.fc1 = None
        self.fc21 = None
        self.fc22 = None
        self.fc3 = None
        self.fc4 = None

        self.num_gpus = num_gpus
        K = self.K = Ks[0]

        d = 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(K, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(d*2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.ConvTranspose2d(d, 3, 4, 2, 1),
            nn.Tanh()
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(d*4, d*8, 4, 2, 1),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),

            nn.Conv2d(d*8, 2*K, 4, 1, 0),
        )

    def encode(self, x):
        x = x.view(-1, 3, 64, 64)
        #nn.parallel.data_parallel(self.enc, inp, range(self.num_gpus))
        h = nn.parallel.data_parallel(self.encoder, x, range(self.num_gpus))
        return (h[:, :self.K, 0, 0]), h[:, self.K:, 0, 0]

    def decode(self, z):
        z = z.contiguous().view(-1, self.K, 1, 1)
        z1 = nn.parallel.data_parallel(self.decoder, z, range(self.num_gpus))
        return z1

    def criterion(self, recon_x, x, mu, logvar):
        eps = 1e-20
        recon_x = recon_x.view(-1, self.L2)
        x = x.view(-1, self.L2)
        crt = lambda xhat, tar: torch.sum(((xhat - tar)**2 ), 1)

        BCE = crt(recon_x, x)
        v = 1
        KLD = -0.5 * torch.sum(1 + logvar - ((mu.pow(2) + logvar.exp())/v), 1)
        # Normalise by same number of elements as in reconstruction
        # KLD = KLD /(x.size(0) * x.size(1))
        return BCE + KLD

    def generate_data(self, N, base_dist='fixed_iso_gauss'):
        #self.train(mode=False)
        #self.eval()
        seed = torch.randn(N, self.Ks[0]) 
        if self is self.cuda(): 
            seed = seed.cuda()
        seed = Variable(seed)
        gen_data = self.decode(seed)
        return gen_data, seed

def vis_plot_results(model, inv_f, data, vis, all_costs, ep, arguments):
    #if model.base_dist == 'mixture_full_gauss':
    #    gen_data = Variable(torch.randn(200, model.L2))
    #    seed = Variable(torch.randn(200, model.Ks[0]))
    #else:
    gen_data, seed = model.generate_data(200, base_dist='fixed_iso_gauss')
    if arguments == None:
        datatype = 'images'
    else:
        datatype = arguments.data

    if datatype == 'images':
        N = 64
        M = model.M 
        images = ut.collate_images(gen_data, N=N, L=M)
        sz = 800
        opts={'width':sz, 'height':sz}
        opts['title'] = 'Generated Images'
        hm1 = vis.heatmap(images, opts=opts, win='generated')

        M=data.size(1)
        data_images = ut.collate_images(Variable(data.cpu()), N=N, L=M)
        opts['title'] = 'Training Data'
        hm = vis.heatmap(data_images, opts=opts, win='tr_data')
       
        M=inv_f[0].size(1)
        opts['title'] = 'Mapping to Base Distribution Space'
        if M == model.L2:
            hhat_images = ut.collate_images(inv_f[0], N=N, L=M)
            opts['xmax'] = 2
            opts['xmin'] = -0.2
            hm2 = vis.heatmap(hhat_images, opts=opts, win='hhat')
        elif M == 2:
            hhat = inv_f[0].data.cpu()
            opts['markercolor'] = np.ones(hhat.size(0))*0
            vis.scatter(hhat, opts=opts, win='hhat')
            opts['markercolor'] = np.ones(seed.size(0))*100
            vis.scatter(seed.data.cpu(), opts=opts, win='hhat', update='append')

        M=inv_f[-1].size(1)
        xhat_images = ut.collate_images(inv_f[-1], N=N, L=M)
        opts['title'] = 'Remappings to Observation Space'
        hm3 = vis.heatmap(xhat_images, opts=opts, win='xhat')

        opts['title'] = 'Epoch vs Cost'
        opts['xtype'], opts['ytype'] = 'linear', 'log'
        vis.line(Y=all_costs[:ep+1], X=torch.arange(1,ep+2), opts=opts, win='cost')
    
    elif datatype == 'toy_example':

        opts = {}
        opts['title'] = 'Training Data'
        vis.scatter(data.cpu(), win='x', opts=opts, name='x')

        opts['title'] = 'Generated Data'
        vis.scatter(gen_data.data.cpu(), win='xgen', opts=opts)

        #opts['title'] = 'Training Data Histogram'
        #vis.mesh(data.cpu(), win='x_heat', opts=opts)

        opts['title'] = 'Mappings to Base Distribution Space'
        vis.scatter(inv_f[0].data.cpu(), win='hhat', opts=opts)

        opts['title'] = 'Mapping Back to observation space'
        vis.scatter(inv_f[-1].data.cpu(), win='xhat', opts=opts)

        opts['title'] = 'Epoch vs Cost'
        opts['xtype'], opts['ytype'] = 'linear', 'log'
        vis.line(Y=all_costs[:ep+1], X=torch.arange(1,ep+2), opts=opts, win='cost')

def compute_nparam_density(test_loader, model, sig, cuda, num_samples=1, task='celeba'):

    utils = Normalizing_Flow_utils(0.1, cuda=cuda) 

    all_averages = []
    for n in range(num_samples):
        gen_data, _ = model.generate_data(1000, base_dist=model.base_dist) 
        gen_data = gen_data.view(-1, model.L2).data

        all_kdes = []
        for i, (test_data, _) in enumerate(it.islice(test_loader,0,5000,1)):
            print('batch {}'.format(i))
            test_data = test_data.view(-1, model.L2)
            kdes = utils.compute_kde(utils.tocuda(test_data), (gen_data), sig)
            _, kde_inds = torch.max(kdes, dim=0)
            kdes = utils.logsumexp(kdes, dim=0)
            all_kdes.append(kdes.squeeze()) 
        all_averages.append(torch.cat(all_kdes, dim=0).mean(0)[0])

        # again for below
        for i, (test_data, _) in enumerate(it.islice(test_loader,0, 2, 1)):
            print('batch {}'.format(i))
            test_data = test_data.view(-1, model.L2)
            kdes = utils.compute_kde(utils.tocuda(test_data), (gen_data), sig)
            _, kde_inds = torch.max(kdes, dim=0)
            kdes = utils.logsumexp(kdes, dim=0)
            all_kdes.append(kdes.squeeze()) 

    n_images = 24
    ncols = n_images

    if task == 'mnist':
        M = model.M
        im_test = ut.collate_images(Variable(test_data.view(-1, M, M)), n_images, ncols)

        im_gen = gen_data[kde_inds[:n_images], :].view(-1, M, M) 
        im_gen = ut.collate_images(Variable(im_gen).view(-1, M, M), n_images, ncols)
    else: 
        M = 64
        im_test = ut.collate_images_color(Variable(test_data.view(-1, 3, M, M)), n_images, ncols)

        im_gen = gen_data[kde_inds[:n_images], :].view(-1, 3, M, M) 
        im_gen = ut.collate_images_color(Variable(im_gen).view(-1, 3, M, M), n_images, ncols)

    return all_averages, im_gen, im_test




def adversarial_wasserstein_trainer(train_loader, 
                                    generator, discriminator, EP=250,
                                    **kwargs):
    arguments = kwargs['arguments']
    vis = kwargs['vis']
    config_num = kwargs['config_num']
    cuda = arguments.cuda
    optimizer = 'Adam'
    lr = 1e-4
    clamp_lower = -0.01
    clamp_upper = 0.01

    if optimizer == 'Adam':
        optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
        optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    elif optimizer == 'RMSprop':
        optimizerD = optim.RMSprop(discriminator.parameters(), lr=lr)
        optimizerG = optim.RMSprop(generator.parameters(), lr=lr)
    else:
        raise ValueError('Whaaaat?')

    one = torch.FloatTensor([1])
    mone = one * -1
    
    for ep in range(EP):
        nbatches = 10 
        for i, (tar, _) in enumerate(it.islice(train_loader,0 ,nbatches, 1)):
            if arguments.cuda:
                tar = tar.cuda()
                one, mone = one.cuda(), mone.cuda()
                #mix = mix.cuda()

            
            for p in discriminator.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            if ep < 2 and i < 400:
                Diters = 100
            else:
                Diters = 5

            # sort the tensors within batch
            #tar = tar.contiguous().view(-1, generator.L2)
            tar = Variable(tar)

            for disc_ep in range(Diters):
                
                for p in discriminator.parameters():
                    p.data.clamp_(clamp_lower, 
                                  clamp_upper)

                # discriminator gradient with real data
                discriminator.zero_grad()
                out_d = discriminator.forward(tar)
                err_D_real = out_d.mean()
                err_D_real.backward(one)

                ft = torch.randn(arguments.batch_size, generator.Ks[0])
                if cuda: 
                    ft = ft.cuda()
                # discriminator gradient with generated data
                out_g = generator.forward(Variable(ft, volatile=True))
                out_d_g = discriminator.forward(Variable(out_g.data))
                err_D_fake = out_d_g.mean()
                err_D_fake.backward(mone)

                err_D = err_D_real - err_D_fake
                optimizerD.step()


            #generator_params = list(generator.parameters())
            #print(generator_params[0].data.sum())

            for p in discriminator.parameters():
                p.requires_grad = False # to avoid computation

            # generator gradient

            ft = torch.randn(arguments.batch_size, generator.Ks[0])
            if cuda: 
                ft = ft.cuda()
            generator.zero_grad()
            out_g = generator.forward(Variable(ft))
            out_d_g = discriminator.forward(out_g)
            err_G = out_d_g.mean()
            err_G.backward(one)

            optimizerG.step()
            print('batch {}'.format(i) )
            print('[%d/%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f config num %d, learning rate is %f\r'%
                  (ep, EP, err_D.data[0], err_G.data[0], err_D_real.data[0], 
                  err_D_fake.data[0], config_num, optimizerG.param_groups[0]['lr']))

        if ep % 1 == 0:
            N = 64
            if arguments.data == 'mnist':
                M = 28
                images = ut.collate_images(out_g, N=N, L=M)
                sz = 800
                opts={'width':sz, 'height':sz}
                opts['title'] = 'Generated Images Wasserstein'
                hm1 = vis.heatmap(images, opts=opts, win='generated was')
            else:
                opts = {}
                opts['title'] = 'VAE Generated Images'
                vis.images(out_g.data.cpu()*0.5 + 0.5, opts=opts, win='vae_gen_data')



class conv_autoenc(nn.Module):
    # initializers
    def __init__(self, d=128, K=100, Kdict=30, base_inits=20, num_gpus=1):
        super(conv_autoenc, self).__init__()

        self.usesequential = True if num_gpus > 1 else False
        self.num_gpus = num_gpus
        if self.usesequential:
            self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(K, d*8, 4, 1, 0),
                    nn.BatchNorm2d(8*d),
                    nn.ReLU(True),
                    
                    nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
                    nn.BatchNorm2d(d*4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
                    nn.BatchNorm2d(d*2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(d*2, d, 4, 2, 1),
                    nn.BatchNorm2d(d),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(d, 3, 4, 2, 1),
                    nn.Tanh()

                )

        else:
            self.deconv1 = nn.ConvTranspose2d(K, d*8, 4, 1, 0)
            self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
            self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),

            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),

            nn.Conv2d(d*4, d*8, 4, 2, 1),
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.Conv2d(d*8, K, 4, 1, 0),
        )
        self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, n_init=base_inits, max_iter = 200, covariance_type='full')
        self.base_dist = 'mixture_full_gauss'
        self.L2 = 3*64*64


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # decoder
    
    def decode(self, inp):
        if self.usesequential:
            x = nn.parallel.data_parallel(self.decoder, inp, range(self.num_gpus))
        else:
            x = F.relu((self.deconv1(inp)))
            x = F.relu((self.deconv2(x)))
            x = F.relu((self.deconv3(x)))
            x = F.relu((self.deconv4(x)))
            x = F.tanh(self.deconv5(x))
        return x

    def encode(self, x):
        h = nn.parallel.data_parallel(self.encoder, x, range(self.num_gpus))
        return h

    def trainer(self, train_loader, vis, EP, cuda, config_num=0):
        opt = torch.optim.Adam(self.parameters(), lr=2e-5, betas=(0.5, 0.999)) 

        nbatches = 25000 
        for ep in range(EP):
            for i, (dt, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                self.zero_grad()
                if cuda:
                    dt = dt.cuda()
                h = self.encode(Variable(dt))
                xhat = self.decode(h)
                cost = ((Variable(dt) - xhat)**2).mean()
            
                cost.backward()

                opt.step()

                print('EP [{}/{}], batch [{}/{}],  Cost is {}, Learning rate is {}, Config num {}'.format(ep+1, EP, i+1, 
                                                                                                          len(train_loader), cost.data[0], 
                                                                                                          opt.param_groups[0]['lr'], config_num))

                if i % 50 == 0:
                    vis.images(0.5 + (dt.cpu()*0.5), win='x')
                    vis.images(0.5 + (xhat.data.cpu()*0.5), win='xhat')
                 
    def gmm_trainer(self, train_loader, cuda=True, vis=None):

        # get hhats for all batches
        nbatches = 2
        all_hhats = []
        for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
            if cuda:
                data = data.cuda()
            hhat = self.encoder(Variable(data))
            all_hhats.append(hhat.data.squeeze())
            print(i)
        all_hhats = torch.cat(all_hhats, dim=0)

        if vis is not None:
            vis.heatmap(all_hhats.cpu()[:200].t(), win='hhat', opts = {'title':'hhats for reconstructions'})


        # train base dist
        data = all_hhats.cpu().numpy()
        self.GMM.fit(data)
    
    def generate_data(self, N, base_dist='mixture_full_gauss'):
        seed = self.GMM.sample(N)[0]
        seed = torch.from_numpy(seed).float()
        seed = seed.view(N, -1, 1, 1)
        if self.cuda:
            seed = seed.cuda()
        seed = Variable(seed)
        return self.decode(seed), seed

class netg_dcgan_par(nn.Module):
    def __init__(self, K, d=128, num_gpus=4):
        super(netg_dcgan_par, self).__init__()
        self.K = K
        self.L2 = 3*64*64
        self.base_dist = 'fixed_iso_gauss'
        self.num_gpus = num_gpus 
        self.Ks = [100]
        self.M = 64

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(K, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            nn.ConvTranspose2d(d, 3, 4, 2, 1),
            nn.Tanh()
        )

    # forward method
    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1,1)
        x = nn.parallel.data_parallel(self.dec, input, range(self.num_gpus))
        return x

    def generate_data(self, N, base_dist='fixed_iso_gauss'):
        seed = torch.randn(N, self.K, 1, 1).cuda()
        seed = Variable(seed)
        return self.forward(seed), seed



class netg_dcgan(nn.Module):
    # initializers
    def __init__(self, K, d=128, num_gpus=4):
        super(netg_dcgan, self).__init__()
        self.K = K
        self.L2 = 3*64*64
        self.base_dist = 'fixed_iso_gauss'
        self.num_gpus = num_gpus 
        self.Ks = [100]
        self.M = 64

        self.deconv1 = nn.ConvTranspose2d(K, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        #self.dec = nn.Sequential(
        #    nn.ConvTranspose2d(K, d*8, 4, 1, 0),
        #    nn.BatchNorm2d(d*8),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
        #    nn.BatchNorm2d(d*4),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
        #    nn.BatchNorm2d(d*2),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d*2, d, 4, 2, 1),
        #    nn.BatchNorm2d(d),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d, 3, 4, 2, 1),
        #    nn.Tanh()
        #)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        input = input.view(input.size(0), input.size(1), 1,1)
        #x = nn.parallel.data_parallel(self.dec, input, range(self.num_gpus))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

    def generate_data(self, N, base_dist='fixed_iso_gauss'):
        seed = torch.randn(N, self.K, 1, 1).cuda()
        seed = Variable(seed)
        return self.forward(seed), seed

class netd_dcgan_par(nn.Module):
    # initializers
    def __init__(self, d=128, num_gpus=4):
        super(netd_dcgan_par, self).__init__()

        self.num_gpus = num_gpus

        c = 0.1 
        self.disc = nn.Sequential(
                nn.Conv2d(3, d, 4, 2, 1),
                nn.LeakyReLU(c),
                nn.Conv2d(d, d*2, 4, 2, 1),
                nn.BatchNorm2d(d*2),
                nn.LeakyReLU(c),
                nn.Conv2d(d*2, d*4, 4, 2, 1),
                nn.BatchNorm2d(d*4),
                nn.LeakyReLU(c),
                nn.Conv2d(d*4, d*8, 4, 2, 1),
                nn.BatchNorm2d(d*8),
                nn.LeakyReLU(c),
                nn.Conv2d(d*8, 1, 4, 1, 0),
        )


    # forward method
    def forward(self, input):
       
        #input = input.view(-1, 3, 64, 64)
        x = nn.parallel.data_parallel(self.disc, input, range(self.num_gpus))
        
        return x


class netd_dcgan(nn.Module):
    # initializers
    def __init__(self, d=128, num_gpus=4):
        super(netd_dcgan, self).__init__()

        self.num_gpus = num_gpus

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        
        self.conv3 =  nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)

        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)


        #self.disc = nn.Sequential(
        #        nn.Conv2d(3, d, 4, 2, 1),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d, d*2, 4, 2, 1),
        #        nn.BatchNorm2d(d*2),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d*2, d*4, 4, 2, 1),
        #        nn.BatchNorm2d(d*4),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d*4, d*8, 4, 2, 1),
        #        nn.BatchNorm2d(d*8),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d*8, 1, 4, 1, 0),
        #        nn.Sigmoid()
        #)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
       
        #input = input.view(-1, 3, 64, 64)
        #x = nn.parallel.data_parallel(self.disc, input, range(self.num_gpus))
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = (self.conv5(x))

        return x


def weights_init_seq(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if 0: #isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

    

class CharRNN(nn.Module):
    def __init__(self, L, K1, K2, output_size, n_layers, Kdict=30, base_inits = 1, base_dist='GMM', usecuda=True):
        super(CharRNN, self).__init__()
        self.L = L
        self.K1 = K1
        self.K2 = K2
        self.n_layers = n_layers
        #self.encoder = Variable(torch.eye(input_size).cuda())
        #self.encoder = nn.Embedding(L, K1)
        #self.rnn = ModRNNBase(self.model, input_size, hidden_size, n_layers)
        self.rnn = nn.LSTM(1, K2, n_layers, batch_first=True)
        self.decrnn = nn.LSTMCell(K2, K2)
        #self.decrnn_out = nn.Linear(K2, K2)
        self.declin = nn.Linear(K2, 1)

        self.base_dist = base_dist
        if base_dist == 'mixture_full_gauss':
            self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, tol=1e-5, n_init=base_inits, max_iter = 200)
        elif base_dist == 'HMM':
            self.HMM = hmm.GaussianHMM(n_components=Kdict, n_iter=1000,
                                       covariance_type='diag', tol=1e-7, verbose='True')

        
    def decode(self, code, ln):
        inp = (code)        
        zerosmat = torch.zeros(code.size())
        if next(self.parameters()).is_cuda:
            zerosmat = zerosmat.cuda()
        zerosmat = Variable(zerosmat)
        h = (zerosmat, zerosmat)
        outs = []
        for l in range(ln):
            h = self.decrnn(inp, h)
            inp = (h[0])
            outs.append(h[0].unsqueeze(1))
        outs = torch.cat(outs, dim=1)
        #outs = outs.permute([1, 0, 2])
        outs = outs.contiguous().view(-1, self.K2)

        outs = self.declin(outs)
        return outs


    def forward(self, inp):
        batch_size = inp.size(0)
        #encoded = torch.index_select(self.encoder, 0, inpt)
        #encoded = self.encoder(inp)
        inp = inp.unsqueeze(-1).float()
        output, _ = self.rnn(inp)
        code = output.sum(1)

        output = self.decode(code, inp.size(1))
        return output, code

    def trainer(self, train_loader, vis, EP, cuda, config_num=0, path=None):
        opt = torch.optim.Adam(self.parameters(), lr=5e-5) 

        nbatches = 10
        for ep in range(EP):
            for i, (dt, tar, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                self.zero_grad()

                if cuda:
                    dt = dt.cuda()
                    tar = tar.cuda()

                tar_rsh = tar.unsqueeze(1).float()

                xhat, code = self.forward(Variable(tar_rsh))
                
                #tar_rsh = tar.contiguous().view(-1).float()
                #cost = F.cross_entropy(xhat, Variable(tar_rsh))

                cost = ((Variable(tar_rsh) - xhat)**2).mean()

                cost.backward()

                opt.step()

                print('EP [{}/{}], batch [{}/{}], \
                       Cost is {}, Learning rate is {}, \
                       Config num {}, path {}'.format(ep+1, EP, i+1, 
                                             len(train_loader), 
                                             cost.data[0], 
                                             opt.param_groups[0]['lr'], 
                                             config_num, path))
                
            N = 200
            opts = {'title' : 'x'}
            vis.line(tar_rsh.cpu()[0].squeeze(), opts=opts, win='x')

            opts = {'title' : 'xhat'}
            vis.line(xhat.squeeze().data.cpu()[0].squeeze(), opts=opts, win='xhat')

            #recons_x = self.join_chars(dt_rsh) 
            #recons_xhat = self.recons_xhat(xhat)
            #opts = {'title': 'xhat'}
            #vis.text(recons_xhat, win='xhat', opts=opts)
            #opts = {'title': 'x'}
            #vis.text(recons_x, win='x', opts=opts)

    def base_dist_trainer(self, train_loader, cuda=True, vis=None, path=None):
        # get hhats for all batches
        nbatches = 2 
        all_hhats = []
        for i, (data, _, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
            if cuda:
                data = data.cuda()
            data = data.unsqueeze(1)
            xhat, hhat = self.forward(Variable(data.float()))
            all_hhats.append(hhat.squeeze())
            print(i)
        all_hhats = torch.cat(all_hhats, dim=0)
        
        # reconstruct
        xhat_concat = ut.pt_to_audio_overlap(xhat)
        lr.output.write_wav('reconstructions.wav', xhat_concat, 8000, norm=True)
        # see the embeddings
        
        vis.heatmap(all_hhats.data.cpu()[:100].t(), win='hhat', opts = {'title':'hhats for reconstructions'})
        

        #lr.output.write_wav('original_data.wav', data.cpu().numpy().reshape(-1)
        #                    , 8000, norm=True)

        # train base dist
        if self.base_dist == 'mixture_full_gauss':
            data = all_hhats.data.cpu().numpy()
            self.GMM.fit(data)
        elif self.base_dist == 'HMM':
            data = all_hhats.data.cpu().numpy()
            self.HMM.fit(data)
        elif self.base_dist == 'RNN':
            # get the loader here
            inputs = all_hhats[:-1, :].data.cpu()
            targets = all_hhats[1:, :].data.cpu()

            def form_rnn_dataset(all_hhats):
                split_size = 100
                splits = torch.split(all_hhats, split_size=100, dim=0)
                splits = [splt.unsqueeze(0) for splt in splits if splt.size(0) == 100]  # filter out different size sequences
                concat = torch.cat(splits, dim=0)
                return concat 
            inputs_splt = form_rnn_dataset(inputs).contiguous()
            targets_splt = form_rnn_dataset(targets).contiguous()

            train_dataset = data_utils.TensorDataset(data_tensor=inputs_splt,
                                                     target_tensor=targets_splt)
            loader = data_utils.DataLoader(train_dataset, batch_size=100, shuffle=False, 
                                           pin_memory=True, num_workers=1)

            self.RNN = GaussianRNN(L=self.K1, K=30, usecuda=self.usecuda,
                                   usevar=False)
            if self.usecuda:
                self.RNN = self.RNN.cuda()

            path = path + '.rnn'
            if 1 & os.path.exists(path):
               self.RNN.load_state_dict(torch.load(path)) 
            else:
                muhat = self.RNN.trainer(loader)
                torch.save(self.RNN.state_dict(), path)

                xhat_frames = self.dec(muhat.unsqueeze(-1))
                xhat_concat = ut.pt_to_audio_overlap(xhat_frames)

                lr.output.write_wav('rnn_reconstructions.wav', xhat_concat, 8000, norm=True)


    def generate_data(self, N, arguments):
        if self.base_dist == 'mixture_full_gauss':
            seed = self.GMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()
        elif self.base_dist == 'HMM':
            seed = self.HMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()
        elif self.base_dist == 'RNN':
            self.training = False
            bs = 2
            #zerosmat = torch.zeros(  ,bs, self.RNN.K)
            #if self.usecuda: #next(self.parameters()).is_cuda:
            #    zerosmat = zerosmat.cuda()
            #zerosmat = Variable(zerosmat)
            #h = (zerosmat, zerosmat)
            inp = torch.zeros(bs, 1, self.RNN.L)
            if self.usecuda:
                inp = inp.cuda()
            inp = Variable(inp)
            outs = []
            for l in range(N):
                if l == 0:
                    h = self.RNN.rnn(inp)
                else: 
                    h = self.RNN.rnn(inp, h[1])

                U = torch.rand(h[0].squeeze().size())
                if self.usecuda:
                    U = U.cuda()
                U = Variable(U)
                lam = 0.001
                sms = self.RNN.sm(h[0].squeeze())
                
                samples = torch.multinomial(F.softmax(sms, dim=1), 1) 

                eye = torch.eye(self.RNN.K)
                if self.usecuda:
                    eye = eye.cuda()
                eye = Variable(eye)

                smh = eye[samples.squeeze()]
                
                #G = - torch.log(-torch.log(U))
                #smh = F.softmax( (sms + G  )/lam, dim=1)

                mu = torch.matmul(self.RNN.out_mean, smh.t()).t()
                if self.RNN.usevar:
                    std = torch.matmul(self.RNN.out_logvar, smh.t()).t()
                else:
                    std = 0.2
                z = torch.randn(bs, self.RNN.L)
                if self.usecuda: 
                    z = z.cuda()
                inp = (Variable(z)*std + mu).unsqueeze(1)
                outs.append(inp) 
            outs = torch.cat(outs, dim=1) 
            seed = outs.view(-1, outs.size(-1), 1)

        if self.base_dist is not 'RNN':
            seed = seed.view(N, -1, 1)
            if self.cuda:
                seed = seed.cuda()
            seed = Variable(seed)
        
        decoded = self.dec(seed)
        return decoded, seed


    def join_chars(self, chnk):
        all_characters = st.printable
        n_characters = len(all_characters)

        recons = ''.join([all_characters[ch] for ch in list(chnk.cpu().numpy())])
        return recons



    def init_hidden(self, batch_size):
        if 'lstm' in self.model:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


class audionet(CharRNN):
    def __init__(self, L, K1, K2, output_size, n_layers, Kdict=30, base_inits = 1, base_dist='GMM', num_gpus=1, usecuda=True):
        super(audionet, self).__init__(L, K1, K2, output_size, 
                                       n_layers, Kdict, 
                                       base_inits, base_dist, usecuda=usecuda)
        
        self.rnn = None
        self.decrnn = None
        self.declin = None
        self.num_gpus = num_gpus
        self.usecuda = usecuda

        d = 64
        kernel_size = math.floor(L/4)
        self.kernel_size = kernel_size

        self.enc = nn.Sequential(
            nn.Conv1d(1, d, kernel_size, 2, int((kernel_size-1)/2)),
            nn.Tanh(),
            nn.Conv1d(d, 2*d, kernel_size, 2, int((kernel_size-1)/2)),
            nn.Tanh(),
            nn.Conv1d(2*d, K1, kernel_size, 1, 0),
            #nn.Tanh(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose1d(K1, d, kernel_size, 1, 0),
            nn.Tanh(),
            nn.ConvTranspose1d(d, 2*d, kernel_size, 2, int((kernel_size-1)/2)),
            nn.Tanh(),
            nn.ConvTranspose1d(2*d, 1, kernel_size, 2, int((kernel_size-1)/2)),
            #nn.Tanh()
        )
    
    def forwardpass(self, inp):
        code = self.enc(inp)
        xhat = self.dec(code)
        return xhat, code

    def forward(self, inp):
        if self.num_gpus > 1: 
            code = nn.parallel.data_parallel(self.enc, inp, range(self.num_gpus))
            xhat = nn.parallel.data_parallel(self.dec, code, range(self.num_gpus))
            outs = [xhat, code]
        else: 
            outs = self.forwardpass(inp)
        return outs

class GaussianRNN(nn.Module):
    def __init__(self, L, K=200, usecuda=False, usevar=False):
        super(GaussianRNN, self).__init__()
        self.L = L
        self.K = K
        self.usecuda = usecuda
        self.num_layers = 1
        self.usevar = usevar

        self.rnn = nn.LSTM(input_size=L, hidden_size=K, num_layers=self.num_layers, batch_first=True, dropout=0.2)
        #self.out_mean = nn.Linear(K, L)
        #torchinit.uniform(self.out_mean.weight, a=-0.01, b=0.01) 
        #torchinit.uniform(self.out_mean.bias, a=-0.1, b=0.1) 
        c = 0.1
        self.sm = nn.Linear(K, K)
        self.out_mean = nn.Parameter(c*torch.randn(L, K))

        if usevar:
            self.out_logvar = nn.Parameter(c*torch.randn(L, K))
        #torchinit.uniform(self.out_logvar.weight, a=-0.01, b=0.01) 
        #torchinit.uniform(self.out_logvar.bias, a=-2, b=0) 

    def forward(self, inp, lam):
        h = self.rnn(inp)[0]
        h_flat = h.contiguous().view(-1, h.size(-1))



        U = torch.rand(h_flat.size())
        if self.usecuda:
            U = U.cuda()
        G = Variable(-torch.log(-torch.log(U)))
        sms = F.softmax((self.sm(h_flat))/lam, dim=1)

        mu = torch.matmul(self.out_mean, sms.t()).t()
        if self.usevar:
            logsig = torch.matmul(self.out_logvar, sms.t()).t()
            return mu, logsig
        else:
            return mu, None

    def trainer(self, train_loader): 
        opt = torch.optim.RMSprop(self.parameters(), lr=1e-3) 
        
        # get a loader here
        EP = 3000
        lam = 1
        for ep in range(EP):
            for i, (dt, tar) in enumerate(train_loader):
                self.zero_grad()

                if self.usecuda:
                    dt = dt.cuda()
                    tar = tar.cuda()
                tar_rsh = tar.view(-1, self.L)
                
                if ep > 0 and ep % 1000 == 0:
                    if self.usevar:
                        lam = lam*0.5
                        print(lam)
                    else:
                        lam = lam*1
                muhat, logsighat = self.forward(Variable(dt), lam)
                
                #logsighat = logsighat.unsqueeze(-1)
                #tar_rsh = tar.contiguous().view(-1).float()
                #cost = F.cross_entropy(xhat, Variable(tar_rsh))

                if self.usevar:
                    eps = 1e-20
                    cost = ( (((Variable(tar_rsh) - muhat)**2)/ (eps + 2*(2*logsighat).exp())).sum(1) + logsighat.sum(1) ).mean()
                    print(logsighat.exp().data.max()) 
                else: 
                    cost = ((Variable(tar_rsh) - muhat)**2).mean()

                #cost = (((self.out_mean.unsqueeze(0) - Variable(tar_rsh).unsqueeze(-1))**2) / ((2*self.out_logvar).exp()*2)).sum(1) + self.out_logvar.sum(0).unsqueeze(0)
                #cost = (cost*sms).sum(1).mean()
                
                #
                cost.backward()

                opt.step()

                print('EP [{}/{}], batch [{}/{}], \
                       Cost is {}, Learning rate is {}'.format(ep+1, EP, i+1, 
                                                         len(train_loader), 
                                                         cost.data[0], 
                                                         opt.param_groups[0]['lr'])) 
        #muhat = torch.matmul(self.out_mean, sms.t()).t()
        return muhat
