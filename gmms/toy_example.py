import torch
import numpy as np
import pdb
import gmm_learn as gmm
import visdom

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev', use_incoming_socket=False)
assert vis.check_connection()

np.random.seed(2)
torch.manual_seed(9)

L = 2
K = 4
N = 2000
cuda = torch.cuda.is_available()

means = torch.randn(K, L)
inds = np.random.randint(0, K, N)

X = means[inds, :]
X = X + 0.1*torch.randn(X.size())
if cuda:
    X = X.cuda()

gmm = gmm.gmm(num_components=K, L=X.size(-1), cuda=cuda, n_iters=50)
gmm.kmeanspp(X)
gmm.kmeans([X[:1000], X[1000:]]) 

gmm.em([X])

gen_data = gmm.sample(200)

if L == 2:
    vis.scatter(X)

    opts = {'markersize' : 17}
    vis.scatter(gmm.means, opts=opts)
else:
    N = 100
    vis.heatmap(X[:N].t().cpu())

    vis.heatmap(gmm.means.t().cpu())

    opts = {'title' : 'Generated data'}
    vis.heatmap(gen_data.t().cpu(), opts=opts)

