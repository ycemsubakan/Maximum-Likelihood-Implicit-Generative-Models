import torch
import numpy as np
import pdb
import torch.nn.utils.rnn as rnn_utils 
import librosa as lr
import torch.utils.data as data_utils
import os
import torch.nn.init as torchinit
import mir_eval.separation as mevalsep 
import pandas as pd
from torchvision import datasets, transforms
import scipy as sp
import sklearn as skt
import itertools as it
import torch
import string
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib as mpl

def collate_images(data, N, ncols = 8, L=28): 
    nrows = int(np.ceil(N/ncols))

    data = data.data.cpu()[:N].view(N, L, L) 
    
    rows = torch.chunk(data, nrows, dim=0) 
    collated_rows = []
    for i, row in enumerate(rows):
        row = torch.chunk(row, ncols, dim=0)
        row = [im.squeeze() for im in row]
        #if i == nrows-1: 
        base = torch.zeros(L, L*(ncols))
        base[:, :(len(row))*L] = torch.cat(row, dim=1)
        collated_rows.append(base)
       # else:
       #     collated_rows.append(torch.cat(row, dim=1))

    collated_im = torch.cat(collated_rows, dim=0)

    inds = torch.arange(L*nrows-1, -1, -1).long()
    return torch.index_select(collated_im, dim=0, index=inds)


def collate_images_rectangular(data, N, ncols = 8, L1=28, L2=28): 
    nrows = int(np.ceil(N/ncols))

    data = data.cpu()[:N].view(N, L1, L2) 
    
    rows = torch.chunk(data, nrows, dim=0) 
    collated_rows = []
    for i, row in enumerate(rows):
        row = torch.chunk(row, ncols, dim=0)
        row = [im.squeeze() for im in row]
        if i == nrows-1: 
            base = torch.zeros(L1, L2*ncols)
            base[:, :len(row)*L2] = torch.cat(row, dim=1)
            collated_rows.append(base)
        else:
            collated_rows.append(torch.cat(row, dim=1))

    collated_im = torch.cat(collated_rows, dim=0)

    inds = torch.arange(L1*nrows-1, -1, -1).long()
    return torch.index_select(collated_im, dim=0, index=inds)

def collate_images_rectangular_color(data, N, ncols = 8, L1=28, L2=28): 
    nrows = int(np.ceil(N/ncols))

    data = data.cpu()[:N].view(N, 3, L1, L2) 
    
    rows = torch.chunk(data, nrows, dim=0) 
    collated_rows = []
    for i, row in enumerate(rows):
        row = torch.chunk(row, ncols, dim=0)
        row = [im.squeeze() for im in row]
        if i == nrows-1: 
            base = torch.zeros(3, L1, L2*ncols)
            base[:, :, :len(row)*L2] = torch.cat(row, dim=2)
            collated_rows.append(base)
        else:
            collated_rows.append(torch.cat(row, dim=2))

    collated_im = torch.cat(collated_rows, dim=1)

    collated_im = collated_im.permute(1, 2, 0)

    inds = torch.arange(L1*nrows-1, -1, -1).long()
    return collated_im #torch.index_select(collated_im, dim=0, index=inds)




def collate_images_color(data, N, ncols = 8, L=64): 
    nrows = int(np.ceil(N/ncols))

    data = data.data.cpu()[:N].view(N, 3, L, L) 
    
    rows = torch.chunk(data, nrows, dim=0) 
    collated_rows = []
    for i, row in enumerate(rows):
        row = torch.chunk(row, ncols, dim=0)
        row = [im.squeeze() for im in row]
        if i == nrows-1: 
            base = torch.zeros(3, L, L*ncols)
            base[:, :, :len(row)*L] = torch.cat(row, dim=2)
            collated_rows.append(base)
        else:
            collated_rows.append(torch.cat(row, dim=2))

    collated_im = torch.cat(collated_rows, dim=1)

    inds = torch.arange(L*nrows-1, -1, -1).long()
    return collated_im #torch.index_select(collated_im, dim=1, index=inds)


def append_dirs(directories):
    all_dirs = [ ''.join(dr) for dr in directories]
    all_dirs_str = '_' + ''.join(all_dirs) + '_'
    return all_dirs_str 

def list_timit_dirs():
    home = os.path.expanduser('~')
    p = os.path.join(home, 'Dropbox', 'RNNs', 'timit', 'timit-wav', 'train') 

    directories = os.listdir(p)
    possible_dirs = []
    for dr in directories:
        path = os.path.join(p, dr)

        males = [name for name in os.listdir(path) if name[0] == 'm']
        females = [name for name in os.listdir(path) if name[0] == 'f']

        possible_dirs = it.chain(possible_dirs, it.product([dr], males, females))

    return possible_dirs
        

def prepare_mixture_gm_data(arguments):
    dataset = []
    
    arguments.L2 = 2
    arguments.L1 = 2
    sig0 = 10
    sig = 1

    num_means = arguments.num_means
    means = sig0*torch.randn(num_means, arguments.L2) 
    arguments.means = means.numpy()

    N = arguments.batch_size

    mixinds = torch.multinomial(torch.ones(num_means), N, replacement=True) 
    obsnoise = sig*torch.randn(N, arguments.L2) 

    data = means[mixinds] + obsnoise
    inp = torch.randn(N, arguments.L1) 

    dataset1 = data_utils.TensorDataset(data, data)
    datasetmix = dataset1 

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader1 = data_utils.DataLoader(dataset1, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader_mix = data_utils.DataLoader(datasetmix, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader1, loader_mix
   

def save_image_samples(samples, save_path, exp_info, mode, arguments):
    N = len(samples)
    sqrtN = int(np.sqrt(N))
    for n, sample in enumerate(samples):
        plt.subplot(sqrtN, sqrtN, n+1)
        sample = sample.contiguous().view(arguments.nfts, arguments.T)
        plt.imshow(sample.cpu().numpy(), cmap='binary')
        plt.clim(0,1)

    plt.savefig(os.path.join(save_path, exp_info + '_' + mode +'.png'))

def save_models(generators, discriminators, exp_info, save_folder, arguments):
    
    for n, generator in enumerate(generators):
        torch.save(generator.state_dict(), os.path.join(save_folder, 'generator' + str(n) + '_'  + exp_info + '.trc'))    

    for n, discriminator in enumerate(discriminators):
        torch.save(discriminator.state_dict(), os.path.join(save_folder, 'discriminator' + str(n) + '_' + exp_info + '.trc'))    


def compile_bssevals(bss_evals): 
    sdrs1, sdrs2 = [], []
    sirs1, sirs2 = [], []
    sars1, sars2 = [], []

    for i, (sdr, sir, sar, _) in enumerate(bss_evals):
        sdrs1.append(sdr[0]), sirs1.append(sir[0]), sars1.append(sar[0])
        sdrs2.append(sdr[1]), sirs2.append(sir[1]), sars2.append(sar[1])  

    df = pd.DataFrame({'sdr1': sdrs1, 'sdr2': sdrs2, 
                       'sar1': sars1, 'sar2': sars2,
                       'sir1': sirs1, 'sir2': sirs2})
    return df

def audio_to_bsseval(s1hats, s2hats, s1s, s2s):
    bss_evals = []
    bss_evals_paris = []
    for i, (s1hat, s2hat, s1, s2) in enumerate(zip(s1hats, s2hats, s1s, s2s)):

        print('Computing bssevals for mixture {}'.format(i))

        sourcehat_mat = np.concatenate([s1hat.reshape(1, -1), s2hat.reshape(1, -1)], 0)
        source_mat = np.concatenate([s1.reshape(1, -1), s2.reshape(1, -1)], 0)

        Nhat, N = sourcehat_mat.shape[1], source_mat.shape[1]
        Nmin = min([N, Nhat])

        bss_evals.append(mevalsep.bss_eval_sources(source_mat[:, :Nmin], 
                                                   sourcehat_mat[:, :Nmin]))
        bss_evals_paris.append([tu.bss_eval(sourcehat_mat[0, :Nmin], 0, 
                                            source_mat[:, :Nmin]), 
                                tu.bss_eval(sourcehat_mat[1, :Nmin], 1,
                                            source_mat[:, :Nmin])])
        print(bss_evals)
        print(bss_evals_paris) 


    return bss_evals

def mag2spec_and_audio_wiener(xhat, recons, MS, MSphase, arguments):

    #xhat = xhat.cpu().numpy()
    #recons = recons.cpu().numpy()
    try:   # pytorch case
        MS = MS.cpu().numpy()
        MSphase = MSphase.cpu().numpy()
        Nmix = MSphase.shape[0]

        maghats = np.split(xhat, Nmix, axis=0) 
        reconss = np.split(recons, Nmix, axis=0) 
        mixmags = np.split(MS, Nmix, axis=0) 
        phases = np.split(MSphase, Nmix, axis=0)

    except:
        maghats = [xhat]
        reconss = [recons]
        mixmags = [MS]
        phases = [MSphase]

   
    all_audio = []
    eps = 1e-20
    for maghat, recons, mixmag, phase in zip(maghats, reconss, mixmags, phases):
        mask = (maghat / (recons + eps))
        all_audio.append(lr.istft((mask*mixmag*np.exp(1j*phase)).transpose(), 
                                  win_length=arguments.win_length))

    return all_audio, maghats


def mag2spec_and_audio(xhat, MSphase, arguments):

    MSphase = MSphase.cpu().numpy()
    Nmix = MSphase.shape[0]
    mags = np.split(xhat, Nmix, axis=0) 
    phases = np.split(MSphase, Nmix, axis=0)

    all_audio = []
    for mag, phase in zip(mags, phases):
        all_audio.append(lr.istft((mag*np.exp(1j*phase.squeeze())).transpose(), 
                                  win_length=arguments.win_length))

    return all_audio, mags

def filter_digits(digit, loader, arguments):
    dataset = []
    for i, (ft, tar) in enumerate(loader):   
        # digit 1
        mask = torch.eq(tar, 0) + torch.eq(tar, 1) 
        inds = torch.nonzero(mask).squeeze()
        ft1 = torch.index_select(ft, dim=0, index=inds)
        dataset.append(ft1)
        print(i)
        
    dataset = torch.cat(dataset, dim=0)
    if arguments.input_type == 'noise':
        L1 = dataset.size(2)*dataset.size(3)
        inp = torch.randn(dataset.size(0), arguments.L1) 
    elif arguments.input_type == 'autoenc':
        inp = dataset
    else:
        raise ValueError('Whaaaaaat input_type?')

    N = dataset.size(0)
    dataset = data_utils.TensorDataset(data_tensor=inp,
                                       target_tensor=dataset)
                            
                            
    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader


def plot_embedding(X, images, ax, sz=(28, 28), title=None):
    shown_images = np.array([[1., 1.]])  # just something big
    #cmap=plt.cm.gray_r,
    n = len(X)
    step = round(n / 30)

    for x, image in zip(X, images):
        imagebox = offsetbox.AnnotationBbox(
        offsetbox.OffsetImage(image.reshape(sz[0], sz[1]), zoom=0.3), x.squeeze())
        ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def plot_gmm(X, Y_, means, covariances, splot):
    color_iter = it.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                                                'darkorange'])
    #color_iter = it.cycle(['red', 'yellow'])

    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = sp.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / sp.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        #plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color, zorder=2)
        #ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.8)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())

def form_mixtures(digit1, digit2, loader, arguments): 
    dataset1, dataset2 = [], []
    for i, (ft, tar) in enumerate(loader):   
        # digit 1
        mask = torch.eq(tar, digit1)
        inds = torch.nonzero(mask).squeeze()
        ft1 = torch.index_select(ft, dim=0, index=inds)
        dataset1.append(ft1)

        # digit 2
        mask = torch.eq(tar, digit2)
        inds = torch.nonzero(mask).squeeze()
        ft2 = torch.index_select(ft, dim=0, index=inds)
        dataset2.append(ft2)
        print(i)
        
    dataset1 = torch.cat(dataset1, dim=0)
    dataset2 = torch.cat(dataset2, dim=0)

    if arguments.input_type == 'noise':
        inp1 = torch.randn(dataset1.size(0), arguments.L1) 
        inp2 = torch.randn(dataset2.size(0), arguments.L1) 
    elif arguments.input_type == 'autoenc':
        inp1 = dataset1
        inp2 = dataset2
    else:
        raise ValueError('Whaaaaaat input_type?')

    N1, N2 = dataset1.size(0), dataset2.size(0)
    Nmix = min([N1, N2])

    dataset_mix = dataset1[:Nmix] + dataset2[:Nmix]
        
    dataset1 = TensorDataset(data_tensor=inp1,
                                        target_tensor=dataset1,
                                        lens=[1]*Nmix)
    dataset2 = data_utils.TensorDataset(data_tensor=inp2,
                                        target_tensor=dataset2)
    dataset_mix = data_utils.TensorDataset(data_tensor=dataset_mix,
                                        target_tensor=torch.ones(Nmix))

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader1 = data_utils.DataLoader(dataset1, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader2 = data_utils.DataLoader(dataset2, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader_mix = data_utils.DataLoader(dataset_mix, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader1, loader2, loader_mix

def get_loaders(loader_batchsize, c=-0.1, train_shuffle=True, **kwargs):
    arguments=kwargs['arguments']
    data = arguments.data

    if data == 'mnist':
        lam = lambda x: torch.eq(x,0).float()*c + (1-torch.eq(x,0).float())*x

        kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Lambda(lam) 
                               #transforms.Normalize((0,), (1,))
                           ])),
            batch_size=loader_batchsize, shuffle=train_shuffle, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Lambda(lam) 
                               #transforms.Normalize((7,), (0.3081,))
                           ])),
            batch_size=500, shuffle=False, **kwargs)

    return train_loader, test_loader

def fit_gaussians_to_digits(loader):
    L = 784
    K = 10
    
    means, sts = torch.zeros(L, K), torch.zeros(L, K) 
    for digit in range(10):
        data = []
        for i, (ft, tar) in enumerate(loader):   
            # digit 1
            mask = torch.eq(tar, digit)
            inds = torch.nonzero(mask).squeeze()
            ft = torch.index_select(ft, dim=0, index=inds)
            data.append(ft)
        print(i)
        datacat = torch.cat(data, dim=0)
        means[:, digit] = datacat.mean(0).view(-1)
        sts[:, digit] = datacat.std(0).view(-1) 

    return means, sts


def sort_pack_tensors(ft, tar, lens):
    _, inds = torch.sort(lens, dim=0, descending=True)
    ft, tar, lens = ft[inds], tar[inds], list(lens[inds])

    ft_packed = rnn_utils.pack_padded_sequence(ft, lens, batch_first=True)
    tar_packed = rnn_utils.pack_padded_sequence(tar, lens, batch_first=True)
    return ft_packed, tar_packed

def do_pca(X, K):
    L = X.shape[0]

    X_mean = X.mean(1) 
    X_zeromean = X - X_mean.reshape(L, 1)

    U, S, V = sp.linalg.svd(X_zeromean) 

    U = U[:, :K] 
    X_Kdim = np.dot(U.transpose(), X_zeromean) 

def dim_red(X, K, mode):
    if mode == 'isomap':
        X_low = skt.manifold.Isomap(5, K).fit_transform(X) 
    elif mode == 'mds':
        X_low = skt.manifold.MDS(K).fit_transform(X)
    elif mode == 'tsne':
        X_low = skt.manifold.TSNE(K, init='pca', random_state=0).fit_transform(X)

    #plt.plot(X_low[:, 0], X_low[:, 1], 'o')
    #plt.show()
    return X_low

def preprocess_timit_files(arguments, dr=None):

    L, T, step = 150, 200, 50  

    #random.seed( s)
    #we pick the set according to trial number 
    Z_temp = tu.sound_set(3, dr = dr) 
    Z = Z_temp[0:4]
    mf = Z_temp[4]
    ff = Z_temp[5]


    # Front-end details
    #if hp is None:
    sz = 1024       
    win_length = sz

    #source 1
    #M1paris, P1 = FE.fe( Z[0] )
    S1 = lr.stft( Z[0], n_fft=sz, win_length=win_length).transpose()
    M1, P1 = np.abs(S1), np.angle(S1) 
    M1, P1, lens1 = [M1], [P1], [M1.shape[0]]

    #dim_red(M1[0], 2, 'tsne') 

    # source 2
    S2 = lr.stft( Z[1], n_fft=sz, win_length=win_length).transpose()
    M2, P2 = np.abs(S2), np.angle(S2) 
    M2, P2, lens2 = [M2], [P2], [M2.shape[0]] 

    #dim_red(M2[0], 2, 'tsne') 


    #mixtures
    M = lr.stft( Z[2]+Z[3], n_fft=sz, win_length=win_length).transpose()
    M_t, P_t = np.abs(M), np.angle(M) 
    M_t, P_t, lens_t = [M_t], [P_t], [M_t.shape[0]]

    M_t1 = [np.abs(lr.stft( Z[2], n_fft=sz, win_length=win_length).transpose())]
    M_t2 = [np.abs(lr.stft( Z[3], n_fft=sz, win_length=win_length).transpose())]

    arguments.n_fft = sz
    arguments.L2 = M.shape[1]
    #arguments.K = 100
    arguments.smooth_output = False
    arguments.dataname = '_'.join([mf, ff])
    arguments.win_length = win_length
    arguments.fs = 16000

    T = 200

    if arguments.plot_training:
        plt.subplot(211)
        lr.display.specshow(M1[0][:T].transpose(), y_axis='log') 
        
        plt.subplot(212)
        lr.display.specshow(M2[0][:T].transpose(), y_axis='log') 

        fs = arguments.fs
        lr.output.write_wav('timit_train1_pt.wav', Z[0], fs)
        lr.output.write_wav('timit_train2_pt.wav', Z[1], fs)
        lr.output.write_wav('timit_test1_pt.wav', Z[2], fs)
        lr.output.write_wav('timit_test2_pt.wav', Z[3], fs)

    loader1 = form_torch_audio_dataset(M1, P1, lens1, arguments, 'source') 
    loader2 = form_torch_audio_dataset(M2, P2, lens2, arguments, 'source')
    loadermix = form_torch_mixture_dataset(M_t, P_t, 
                                           M_t1, M_t2,  
                                           [Z[2]], [Z[3]], 
                                           [Z[2].size], [Z[3].size], 
                                           arguments)
    return loader1, loader2, loadermix


def preprocess_audio_files(arguments, overlap=False):
    '''preprocess audio files 
    '''

    if arguments.data == 'synthetic_sounds':
        #dataname = 'generated_sounds_43_71_35_43_64_73'
        #dataname = 'generated_sounds_20_71_43_51_64_73'
        dataname = 'generated_sounds_20_71_30_40_64_73'
        #dataname = 'generated_sounds_20_71_64_73_64_73'
        #dataname = 'generated_sounds_20_71_50_60_50_60'
        audio_path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GANs', dataname)
                    

        arguments.dataname = dataname
        arguments.K = 200

        files = os.listdir(audio_path)
        files_source1 = [fl for fl in files if 'source2' in fl]
        files_source2 = [fl for fl in files if 'source1' in fl]
        #files_mixture = [fl for fl in files if 'mixture' in fl]
    elif arguments.data == 'real_violin':
        audio_path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GANs', 'real_violin')
        files_source1 = os.listdir(audio_path)

    elif 'spoken_digits' in arguments.data:
        digit1 = arguments.data[-1]

        audio_path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GANs', 'free-spoken-digit-dataset2', 'recordings')
        arguments.K = 200

        files = os.listdir(audio_path)
        #files_source1 = [fl for fl in files if str(digit1)+'_' in fl and 'nicolas' not in fl]
        if arguments.data == 'spoken_digits':
            files_source1 = [fl for fl in files if 'nicolas' not in fl]

        else:
            files_source1 = [fl for fl in files if digit1 + '_' in fl and 'nicolas' not in fl]

        N1 = len(files_source1)
    else:
        raise ValueError('Whaaat?')

    if overlap:
        pass
        win_length = 800
        # first load the files and append zeros
        wavfls1 = []
        lens1 = [] 
        inds = []
        for i, fl1 in enumerate(files_source1):
            print(i, fl1)
            #fs, wavfl1 = sp.io.wavfile.read((os.path.join(audio_path, fl1)))
            wavfl1, fs = lr.load(os.path.join(audio_path, fl1), sr=8000)

            wavfl1 = (wavfl1).astype('float32')
            wavfl1 = wavfl1 / wavfl1.std()

            if wavfl1.ndim == 2:
                wavfl1 = wavfl1.mean(1)
            if arguments.data == 'real_violin':
                wavfl1 = wavfl1[64000:]
           
            chnks = get_overlapping_chunks(wavfl1)
            wavfls1.extend(chnks)
            
            inds.extend([i]*len(chnks))

        wf_hat = reconstruct_overlapping_chunks(wavfls1, 400)
        sp.io.wavfile.write('original_data_np.wav', 
                            rate=8000, data=wf_hat)

    else:
        win_length = 8000
        # first load the files and append zeros
        wavfls1, wavfls2 = [], []
        lens1, lens2 = [], []
        inds = []
        for i, fl1 in enumerate(files_source1):
        
            fs, wavfl1 = sp.io.wavfile.read((os.path.join(audio_path, fl1)))
            if wavfl1.ndim > 1:
                wavfl1 = wavfl1.mean(1)

            wavfl1 = (wavfl1).astype('float32')
            wavfl1 = wavfl1 / wavfl1.std()

                    
            len_wf = wavfl1.shape[0]
            if len_wf < win_length:
                wavfl = np.zeros(win_length)
                wavfl[:len_wf] = wavfl1

                wavfls1.append(wavfl)
                N = 1
            else:
                len1 = wavfl1.shape[0] - wavfl1.shape[0] % win_length
                N = len1 / win_length

                wavfls1.extend(np.split(wavfl1[:len1], N, 0))
            
            inds.extend([i]*int(N))
            lens1.append(len_wf <= win_length)
            print(i)

        wavflsnp = np.array(wavfls1).reshape(-1)
        sp.io.wavfile.write('original_data_np.wav', rate=8000, data=wavflsnp)

        wf_hat = None


    wavfls1 = torch.from_numpy(np.array(wavfls1))
    inds = torch.from_numpy(np.array(inds))
    dataset = TensorDataset(data_tensor=wavfls1,
                            target_tensor=wavfls1,
                            lens=inds)
        

    #arguments.fs = 16000 #fs  # add the sampling rate here

    kwargs = {'num_workers': arguments.num_gpus, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader, wf_hat

def pt_to_audio_overlap(data):
    '''
    Converts generated audio chunks into a waveform by overlap-add procedure
    '''
    data = data.squeeze().data.cpu()
    data_list = np.split(data.numpy(), data.size(0), axis=0)
    data_list = [gd.squeeze() for gd in data_list]
    data_concat = reconstruct_overlapping_chunks(data_list, 400)
    return data_concat

def get_overlapping_chunks(inp, hsize=400, fsize=800):
    inplen = inp.shape[0]

    ptr = 0
    chunks = []
    while ptr <= inplen:
        chnk = np.zeros(fsize)
        temp = inp[ptr:ptr + fsize]

        ln = temp.shape[0]
        chnk[:ln] = temp 
        chnk = chnk*sp.signal.windows.hann(fsize)

        ptr = ptr + hsize
        chunks.append(chnk)

    return chunks

def reconstruct_overlapping_chunks(chunks, hsize):
    fsize = chunks[0].shape[0]
    nframes = len(chunks)

    wf_hat = np.zeros(hsize*(nframes+1))
    ptr = 0
    for i, chnk in enumerate(chunks):
        wf_hat[ptr:ptr + fsize] = wf_hat[ptr:ptr + fsize] + chnk
        ptr = ptr + hsize

    return wf_hat

def form_np_audio_list(SPC, SPCSabs, SPCSphase):
    SPCSabs.append((np.abs(SPC)))
    SPCSphase.append((np.angle(SPC)))


def form_torch_audio_dataset(SPCSabs, SPCSphase, lens, arguments, loadertype):
    
    SPCSabs = torch.from_numpy(np.array(SPCSabs))
    if loadertype == 'mixture':
        SPCSphase = torch.from_numpy(np.array(SPCSphase))
        dataset = TensorDataset(data_tensor=SPCSabs,
                                target_tensor=SPCSphase,
                                lens=lens)
    elif loadertype == 'source':
        if arguments.input_type == 'noise':
            if arguments.noise_type == 'gamma': 
                a, b = 1, 10
                b = 1/float(b)
                sz = (SPCSabs.size(0), SPCSabs.size(1), arguments.L1)
                inp_np = np.random.gamma(a, b, sz)
                plt.matshow(inp_np.squeeze().transpose()[:, :50])
                inp = torch.from_numpy(inp_np).float()
            elif arguments.noise_type == 'bernoulli':
                sz = (SPCSabs.size(0), SPCSabs.size(1), arguments.L1)
                mat = (1/float(8))*torch.ones(sz)
                inp = torch.bernoulli(mat) 

                
            elif arguments.noise_type == 'gaussian':
                inp = torch.randn(SPCSabs.size(0), SPCSabs.size(1), arguments.L1)
            else:
                raise ValueError('Whaaaat?')
        elif arguments.input_type == 'autoenc':
            inp = SPCSabs
            arguments.L1 = arguments.L2
        else:
            raise ValueError('Whaaaaaat input_type?')
        dataset = TensorDataset(data_tensor=inp,
                                target_tensor=SPCSabs,
                                lens=lens)
    else:
        raise ValueError('Whaaaat?') 
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True, **kwargs)

    return loader

def form_torch_mixture_dataset(MSabs, MSphase, 
                               SPCS1abs, SPCS2abs,
                               wavfls1, wavfls2, 
                               lens1, lens2, 
                               arguments):

    MSabs = torch.from_numpy(np.array(MSabs))
    MSphase = torch.from_numpy(np.array(MSphase)) 
    SPCS1abs = torch.from_numpy(np.array(SPCS1abs)) 
    SPCS2abs = torch.from_numpy(np.array(SPCS2abs)) 
    wavfls1 = torch.from_numpy(np.array(wavfls1))
    wavfls2 = torch.from_numpy(np.array(wavfls2))
    
    dataset = MixtureDataset(MSabs, MSphase, SPCS1abs, SPCS2abs, 
                             wavfls1, wavfls2, lens1, lens2)

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader


def append_zeros_all(fls1, fls2, mode):
    lens1, lens2 = [], []
    for fl1, fl2 in zip(fls1, fls2):
        if mode == 'audio':
            lens1.append(fl1.shape[0]), lens2.append(fl2.shape[0])
        elif mode == 'specs':
            lens1.append(fl1.shape[0]), lens2.append(fl2.shape[0])
        else:
            raise ValueError('Whaaat?')

    inds1, lens1 = list(np.flip(np.argsort(lens1),0)), np.flip(np.sort(lens1),0)
    inds2, lens2 = list(np.flip(np.argsort(lens2),0)), np.flip(np.sort(lens2),0)
    fls1, fls2 = np.array(fls1)[inds1], np.array(fls2)[inds2]
    maxlen = max([max(lens1), max(lens2)])
    
    mixes = []
    for i, (fl1, fl2) in enumerate(zip(fls1, fls2)):
        if mode == 'audio':
            fls1[i] = np.pad(fl1, (0, maxlen - fl1.shape[0]), 'constant')
            fls2[i] = np.pad(fl2, (0, maxlen - fl2.shape[0]), 'constant')
            mixes.append(fls1[i] + fls2[i])
        elif mode == 'specs':
            fls1[i] = np.pad(fl1, ((0, maxlen - fl1.shape[0]), (0, 0)), 'constant')
            fls2[i] = np.pad(fl2, ((0, maxlen - fl2.shape[0]), (0, 0)), 'constant')
        else:
            raise ValueError('Whaaat?')

    return list(fls1), list(fls2), mixes, lens1, lens2


class TensorDataset(data_utils.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor, lens):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.lens = lens

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.lens[index]

    def __len__(self):
        return self.data_tensor.size(0)

class MixtureDataset(data_utils.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, MSabs, MSphase, SPCS1abs, SPCS2abs, 
                 wavfls1, wavfls2, lens1, lens2):
        assert MSabs.size(0) == wavfls1.size(0)
        assert wavfls1.size(0) == wavfls2.size(0)

        self.MSabs = MSabs
        self.MSphase = MSphase
        self.SPCS1abs = SPCS1abs
        self.SPCS2abs = SPCS2abs
        self.wavfls1 = wavfls1
        self.wavfls2 = wavfls2
        self.lens1 = lens1
        self.lens2 = lens2

    def __getitem__(self, index):
        return self.MSabs[index], self.MSphase[index], \
               self.SPCS1abs[index], self.SPCS2abs[index], \
               self.wavfls1[index], self.wavfls2[index], \
               self.lens1[index], self.lens2[index]

    def __len__(self):
        return self.MSabs.size(0)


def partition_text_file(file, chunk_len, pct_train):
    chunks = []
    tmp = ''
    for idx, c in enumerate(file):
        tmp += c
        if idx % chunk_len == 0:
            chunks.append(tmp)
            tmp = ''
    np.random.shuffle(chunks)
    train_chunks = chunks[0 : int(pct_train * len(chunks))]
    test_chunks = chunks[int(pct_train * len(chunks)) :]
    train_file = ''.join(train_chunks)
    test_file = ''.join(test_chunks)
    return train_file, test_file

def random_batch(file, chunk_len, batch_size):
    file_len = len(file)
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        try:
            start_index = random.randint(0, file_len - chunk_len)
            end_index = start_index + chunk_len + 1
            chunk = file[start_index:end_index]
            inp[bi] = char_tensor(chunk[:-1])
            target[bi] = char_tensor(chunk[1:])
        except:
            continue
    inp = Variable(inp)
    target = Variable(target)
    if arguments.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def get_loader_text(fl, chunk_len, batch_size, cuda):
    
    inputs = fl[:-1]
    targets = fl[1:]

    N = len(inputs)

    N = N - N % chunk_len
    inputs, targets = inputs[:N], targets[:N]
   
    all_inputs = [char_tensor(inputs[i:i+chunk_len]).view(1,-1) for i in range(0, N, chunk_len)]
    all_targets = [char_tensor(targets[i:i+chunk_len]).view(1,-1) for i in range(0, N, chunk_len)]

    all_inputs = torch.cat(all_inputs, 0)
    all_targets = torch.cat(all_targets, 0)

    dataset = data_utils.TensorDataset(data_tensor=all_inputs,
                                       target_tensor=all_targets)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return loader


def char_tensor(inp):
    all_characters = string.printable
    n_characters = len(all_characters)

    tensor = torch.zeros(len(inp)).long()
    for i, c in enumerate(inp):
        tensor[i] = all_characters.index(c)
    return tensor


