import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad

from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.mixture import GaussianMixture

import numpy as np
import math

from torchvision.utils import save_image

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import png

import os

class Vade(nn.Module):
    
    def __init__(self, input_dim, fc_dim, latent_dim, n_clusters):
        super().__init__()

        self.input_dim = input_dim

        self.fc_1 = nn.Linear(input_dim, fc_dim[0])
        self.fc_2 = nn.Linear(fc_dim[0], fc_dim[1])
        self.fc_3 = nn.Linear(fc_dim[1], fc_dim[2])

        self.fc_z_mean = nn.Linear(fc_dim[2], latent_dim)
        self.fc_z_log_var = nn.Linear(fc_dim[2], latent_dim)
        

        self.fc_4 = nn.Linear(latent_dim, fc_dim[-1])
        self.fc_5 = nn.Linear(fc_dim[-1], fc_dim[-2])
        self.fc_6 = nn.Linear(fc_dim[-2], fc_dim[-3])
        self.fc_7 = nn.Linear(fc_dim[-3], input_dim)



        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

        self.latent_dim = latent_dim
        self.n_clusters = n_clusters




    def gmm_create(self):
        self.theta_p = nn.Parameter(torch.ones(self.n_clusters) / self.n_clusters)
        self.u_p = nn.Parameter(torch.zeros((self.latent_dim, self.n_clusters)))
        self.lambda_p = nn.Parameter(torch.ones((self.latent_dim, self.n_clusters)))

    def gmm_init(self, trainloader):
        #use GMM to init u_p and lambda_p

        self.eval()

        encoded_data = []

        for idx, (x, _) in enumerate(trainloader):
            x = Variable(x).cuda()
            z, _, _ = self.encode(x)
            encoded_data.append(z.data.cpu().numpy())

        encoded_data = np.concatenate(encoded_data)

        gmm = GaussianMixture(n_components = self.n_clusters, covariance_type='diag')
        gmm.fit(encoded_data)

        self.u_p.data.copy_(torch.from_numpy(gmm.means_.T))
        self.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T))



    def get_gamma(self, z, z_mean, z_log_var):
        

        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_clusters) # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_clusters)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_clusters)
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_clusters) # NxK

        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
            (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma


    def loss(self, x, x_decoded_mean, z, z_mean, z_log_var):



        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_clusters) # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_clusters)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_clusters)
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])

        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_clusters) # NxK

        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
            (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        BCE = -torch.sum(x*torch.log(torch.clamp(x_decoded_mean, min=1e-10))+
            (1-x)*torch.log(torch.clamp(1-x_decoded_mean, min=1e-10)), 1)
        logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+\
            torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
        qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
        logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
        logqcx = torch.sum(torch.log(gamma)*gamma, 1)

        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)

        return loss



    def sample(self, z_mean, z_log_var):
        std = z_log_var.mul(0.5).exp_()#why mul 0.5?
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(z_mean)

    def cluster_acc(self, Y, Y_pred):
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y.size

    def encode(self, x):

        h = x.view(-1, self.input_dim)

        h = self.fc_1(h)
        h = F.relu(h, inplace=True)

        h = self.fc_2(h)
        h = F.relu(h, inplace=True)

        h = self.fc_3(h)
        h = F.relu(h, inplace=True)

        z_mean = self.fc_z_mean(h)
        z_log_var = self.fc_z_log_var(h)
        z = self.sample(z_mean, z_log_var)

        return z, z_mean, z_log_var

    def decode(self, z):
        h = z.view(-1, self.latent_dim)

        h = self.fc_4(h)
        h = F.relu(h, inplace=True)

        h = self.fc_5(h)
        h = F.relu(h, inplace=True)

        h = self.fc_6(h)
        h = F.relu(h, inplace=True)

        x_decoded_mean = self.fc_7(h)
        x_decoded_mean = F.sigmoid(x_decoded_mean)
        #another activation?

        return x_decoded_mean


    def forward(self, x):
        z, z_mean, z_log_var = self.encode(x)
        return z, z_mean, z_log_var, self.decode(z)

    def adjust_learning_rate(self, init_lr, optimizer, epoch):
        lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_posterior(self, z):
        z_m=np.repeat(np.transpose(z), self.n_clusters, 1)
        u = self.u_p.data.cpu().numpy()
        l = self.lambda_p.data.cpu().numpy()
        t = self.theta_p.data.cpu().numpy()
        posterior=np.exp(np.sum((np.log(t)-0.5*np.log(2*math.pi*l)-\
                           np.square(z_m-u)/(2*l)),axis=0))
        return posterior/np.sum(posterior,axis=-1,keepdims=True)

    def generate(self, n_samples=10, max_tries=200):
        gen_png=np.zeros((280,280))
        count=0
        tries = 0
        while count < n_samples:
            k=np.randint(self.n_clusters)
            u=self.u_p[:,k].data.cpu().numpy()
            l=self.lambda_p[:,k].data.cpu().numpy()
            z_sample=np.random.multivariate_normal(u,np.diag(l),(1,))
            p=self.get_posterior(z_sample)[k]
            if p > 0.99 or tries > max_tries:
                img=self.decode(Variable(torch.Tensor(z_sample))).data.cpu().numpy()
                img = img.reshape((28,28))*255.0
                gen_png[:,count*28:(count+1)*28]=img
                count += 1
                tries = 0
        return np.asarray(gen_png,dtype=np.uint8)  

    def generate_table(self, n_samples_per_cluster=10, max_tries=200):
        gen_png=np.zeros((280,280))
        for k in range(self.n_clusters):
            u=self.u_p[:,k].data.cpu().numpy()
            l=self.lambda_p[:,k].data.cpu().numpy()
            #print(self.u_p.shape, u.shape, self.lambda_p.shape, l.shape)
            #input()
            count=0
            tries = 0
            while count < n_samples_per_cluster:
                z_sample=np.random.multivariate_normal(u,np.diag(l),(1,))
                p=self.get_posterior(z_sample)[k]
                if p > 0.99 or tries > max_tries:
                    img=self.decode(Variable(torch.Tensor(z_sample))).data.cpu().numpy()
                    img = img.reshape((28,28))*255.0
                    gen_png[k*28:(k+1)*28,count*28:(count+1)*28]=img
                    count += 1
                    tries = 0
                tries += 1

        return np.asarray(gen_png,dtype=np.uint8)

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, 
        visualize=False, anneal=False, num_epochs_pretrain=50, save_every=5):

        use_cuda = torch.cuda.is_available()
        print("using cuda:", use_cuda)
        if use_cuda:
            self.cuda()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if visualize:
            if not os.path.exists("images/"):
                os.makedirs("images/")


        #first need to pre-train as simple autoencoder

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        for epoch in range(num_epochs_pretrain):
            self.train()
            if anneal:
                epoch_lr = self.adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                optimizer.zero_grad()
                inputs = Variable(inputs).cuda()
                
                _, z, _ = self.encode(inputs) #take z_mean
                x_decoded = self.decode(z)

                loss_fn = nn.MSELoss()
                loss = loss_fn(x_decoded, inputs)
                loss.backward()
                optimizer.step()

            print("#Pretrain epoch", epoch, ": mse loss:", loss.data[0])


        self.gmm_create()
        self.gmm_init(trainloader)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        # validate
        self.eval()
        valid_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, mu, logvar, outputs = self.forward(inputs)

            loss = self.loss(outputs, inputs, z, mu, logvar)
            valid_loss += loss.data[0]*len(inputs)



        print("#Epoch -1: Valid Loss: %.5f" % (valid_loss / len(validloader.dataset)))

        for epoch in range(num_epochs):
            self.train()
            if anneal:
                epoch_lr = self.adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                z, mu, logvar, outputs = self.forward(inputs)
                loss = self.loss(outputs, inputs, z, mu, logvar)
                train_loss += loss.data[0]*len(inputs)
                loss.backward()
                optimizer.step()


                

            # validate
            self.eval()
            valid_loss = 0.0
            Y = []
            Y_pred = []
            Z = []
            for batch_idx, (inputs, labels) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, mu, logvar, outputs = self.forward(inputs)

                loss = self.loss(outputs, inputs, z, mu, logvar)
                valid_loss += loss.data[0]*len(inputs)
                gamma = self.get_gamma(z, mu, logvar).data.cpu().numpy()

                Y.append(labels.numpy())
                Y_pred.append(np.argmax(gamma, axis=1))
                Z.append(z.data.cpu().numpy())

                # view reconstruct
                if epoch % 5 == 0 and visualize and batch_idx == 0:
                    n = min(inputs.size(0), 8)
                    comparison = torch.cat([inputs.view(-1, 1, 28, 28)[:n],
                                            outputs.view(-1, 1, 28, 28)[:n]])
                    save_image(comparison.data.cpu(),
                                 'images/reconstruction_' + str(epoch) + '.png', nrow=n)

            Y = np.concatenate(Y)
            Y_pred = np.concatenate(Y_pred)
            Z = np.concatenate(Z)

            acc = self.cluster_acc(Y_pred, Y)
            print("#Epoch %3d: lr: %.5f, Train Loss: %.5f, Valid Loss: %.5f, acc: %.5f" % (
                epoch, epoch_lr, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), acc))

            embed_vis = 1

            if epoch % 5 == 0 and visualize:

                    gen_table = self.generate_table(10)
                    png.from_array(gen_table, 'L').save('images/sample_' + str(epoch) + '.png')

                    Z_embedded = TSNE(n_components=2).fit_transform(Z)
                    plt.figure()
                    for i in range(self.n_clusters):
                        #ind = np.logical_and(Y == i, man_dist_Z < 1000)
                        ind = (Y == i)
                        plt.scatter(Z_embedded[ind, 0], Z_embedded[ind, 1])
                    plt.savefig("images/proj2d_" + str(epoch) + ".png")


            if save_every != 0:
                if epoch % save_every == 0:
                    self.save_model("model.p")



    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

   