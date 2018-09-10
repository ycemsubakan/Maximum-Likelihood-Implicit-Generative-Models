import torch 
import numpy as np
import pdb

class gmm():
    def __init__(self, num_components, L, covariance_type='diag', n_iters=100, tol=1e-6, n_restarts=1, cuda=False): 
        self.K = num_components
        self.L = L
        self.covariance_type = 'diag'
        self.n_iters = n_iters
        self.tol = tol
        self.n_restarts = n_restarts
        self.cuda = cuda

    def kmeans(self, X_all):

        L = X_all[0].size(1)
        eps = 1e-20

        for n in range(self.n_iters):

            all_zs = []
            for X in X_all:
                dists = ((X.view(-1, 1, L) - self.means.view(1, self.K, L))**2).sum(2)
                all_zs.append(dists.min(1)[1])

            nKs = self.tocuda(torch.zeros(self.K))
            all_centers = self.tocuda(torch.zeros(self.K, L))
            for zs, X in zip(all_zs, X_all):
                for k in range(self.K):
                    indices = (zs == k).nonzero().squeeze()
                    
                    try:
                        nKs[k] = nKs[k] + indices.size(0)
                        all_centers[k] = all_centers[k] + torch.index_select(X, dim=0, index=indices).sum(0)
                    except:
                        pass
            all_centers = all_centers / (nKs.unsqueeze(-1) + eps)
            print(n) 

        self.means = (all_centers)
        print(nKs)

        # estimate the covariance for each cluster
        all_covs = self.tocuda(torch.zeros(self.K, L, L))
        for zs, X in zip(all_zs, X_all):
            for k in range(self.K):
                indices = (zs == k).nonzero().squeeze()
                
                try:
                    Xsel = torch.index_select(X, dim=0, index=indices)
                    Xsel_cntr = Xsel - all_centers[k].unsqueeze(0)

                    all_covs[k] = all_covs[k] + torch.matmul(Xsel_cntr.t(), Xsel_cntr)  #(Xsel_cntr.unsqueeze(-1) * Xsel_cntr.unsqueeze(1)).sum(0)
                except:
                    pass

        self.covs = all_covs / (nKs.view(-1, 1, 1) + eps)
        self.pis = nKs / nKs.sum()

        # should I add kmeans with covariances? 

    def em(self, X_all, em_iters=15): 
        L = X_all[0].size(1)
            
        eps = 1e-20
        cov_eps = 1e-8
        icovs = []
        icovs_ten = self.tocuda(torch.zeros(self.K, L, L))
        for k in range(self.K):
            icovs_ten[k, :, :] = torch.inverse(self.covs[k, :, :] + self.tocuda(torch.eye(L))*cov_eps)

        for n in range(em_iters):
            print(n)

            # E-step
            all_zs = [] 
            for nb, X in enumerate(X_all):
                dists = (X.view(-1, 1, L) - self.means.view(1, self.K, L)).unsqueeze(2)

                xS = torch.matmul(dists, icovs_ten.unsqueeze(0))

                vals = -0.5*(xS * dists).sum(-1).squeeze()
                det_terms = torch.Tensor([0.5*(eps + torch.svd(cov.squeeze())[1]).log().sum() for cov in self.covs])
                det_terms = self.tocuda(det_terms)
                vals = vals - det_terms.unsqueeze(0) + torch.log(self.pis).unsqueeze(0)

                lse = self.logsumexp(vals, dim=1) 
                print('loss {} batch {}'.format(lse.mean().data.item(), nb))
                all_zs.append( torch.exp( vals - lse) )

            # M-step
            all_means = self.tocuda(torch.zeros(self.K, L))
            z_sums = self.tocuda(torch.zeros(self.K))
            for zs, X in zip(all_zs, X_all):
                all_means = (X.unsqueeze(-1) * zs.unsqueeze(1)).sum(0).t() + all_means
                z_sums = z_sums + zs.sum(0)
            self.means = all_means / (z_sums.unsqueeze(-1) + eps)
            
            all_covs = self.tocuda(torch.zeros(self.K, L, L))
            for zs, X in zip(all_zs, X_all):
                #Xm = (X.unsqueeze(1) - all_means.unsqueeze(0))
                #all_covs = ((Xm.unsqueeze(-1) * Xm.unsqueeze(2))*zs.view(-1, self.K, 1, 1)).sum(0) + all_covs
                Xm = (X.unsqueeze(1) - self.means.unsqueeze(0))
                all_covs = all_covs + torch.matmul(Xm.permute(1, 2, 0), Xm.permute(1, 0, 2)*zs.t().unsqueeze(-1)  ) 

            self.covs = (all_covs / (z_sums.view(-1, 1, 1) + eps)) + self.tocuda(torch.eye(L))*1e-5
            self.pis = z_sums / z_sums.sum()

            print(self.pis.unsqueeze(0))

            for k in range(self.K):
                icovs_ten[k, :, :] = torch.inverse(self.covs[k, :, :] + self.tocuda(torch.eye(L))*cov_eps)
        
        self.icovs = icovs_ten
        self.zs_all = all_zs

        #self.means = all_means

    def sample(self, N=100):
        
        Us = [torch.svd(cov)[0].mm(torch.sqrt(torch.svd(cov)[1]).diag()).unsqueeze(0) for cov in self.covs]
        Us = torch.cat(Us, dim=0)
        
        noise = self.tocuda(torch.randn(N, self.L, 1))
       
        zs = self.tocuda(torch.from_numpy(np.random.choice(range(self.K), size=N, p=self.pis.cpu().numpy())).type(torch.LongTensor))
        Us_zs = torch.index_select(Us, dim=0, index=zs)
        means_zs = torch.index_select(self.means, dim=0, index=zs)

        gen_data = torch.matmul(Us_zs, noise).squeeze() + means_zs
        return gen_data


    def logsumexp(self, X, dim): 
         m, _ = torch.max(X, dim=dim, keepdim=True)
         first_term = (X - m).exp().sum(dim, keepdim=True).log() 

         return first_term + m 

    def tocuda(self, x):
        if self.cuda:
            return x.cuda()
        else:
            return x


    def kmeanspp(self, X):
        n_comps = 0 
    
        Ndata = X.size(0) 
        L = X.size(1)
        centers = []
        while n_comps < self.K:
            if n_comps == 0:
                ind = np.random.randint(Ndata)  
            else: 
                p = dists.cpu().numpy()
                p = p / p.sum()
                ind = np.random.choice(range(Ndata), 1, p=p)[0]
            centers.append(X[ind].unsqueeze(0))

            centers_pt = torch.cat(centers, 0)

            dists = ((X.view(-1, 1, L) - centers_pt.view(1, -1, L))**2).sum(2)
            dists = dists.min(dim=1)[0]

            n_comps = n_comps + 1

        self.means = centers_pt
