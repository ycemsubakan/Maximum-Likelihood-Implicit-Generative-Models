import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from vade import Vade
import argparse

parser = argparse.ArgumentParser(description='VADE MNIST Example')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--save', type=str, default="")
parser.add_argument('--load', type=str, default="")
parser.add_argument('--latent', type=int, default=10)
parser.add_argument('--clusters', type=int, default=10)
parser.add_argument('--pretrain', type=int, default=20)
args = parser.parse_args()

BATCH_SIZE, EPOCHS, LR = args.batch_size, args.epochs, args.lr

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

vade = Vade(input_dim=784, latent_dim=args.latent, n_clusters=args.clusters,
        fc_dim=[500,500,2000])

if args.load != "":
    print("Loading model from %s..." % args.load)
    vade.load_model(args.load)
vade.fit(train_loader, test_loader, lr=LR, batch_size=BATCH_SIZE, num_epochs=EPOCHS, anneal=True, visualize=True, num_epochs_pretrain=args.pretrain)
if args.save != "":
    vade.save_model(args.save)