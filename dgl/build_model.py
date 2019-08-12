

import torch
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dgl.dataset import GraphData
from dgl.config import args
from dgl.model import Encoder, Decoder
from dgl.features import _sliced_wasserstein_distance, rand_uniform2d, d2_distance_matrix
from dgl.utils import show_graph_with_labels, gpu2cpu, sample2d, create_grid


class GraphLayoutVAE(pl.LightningModule):

    def __init__(self, args):
        super(GraphLayoutVAE, self).__init__()
        self.dataset_folder = args.dataset_folder
        self.dataset = args.dataset

        out_dim = args.out_dim
        dims = {'can_96': 96, }
        in_dim = dims[args.dataset]
        self.encoder = Encoder(in_dim, out_dim)
        self.decoder = Decoder(in_dim, out_dim)


    def forward(self, x, adj):
        enc = self.encoder(x, adj)
        dec = self.decoder(enc, adj)
        return dec, enc

    def my_loss(self, y_hat, y, encoded_samples):
        # draw random samples from latent space prior distribution
        z = rand_uniform2d(y_hat.shape[0]).to(encoded_samples.device)
        swd = _sliced_wasserstein_distance(encoded_samples, z,
                                           50, 2, args.device)
        l2 = torch.mean((y_hat-y)**2)
        return l2 + 0.1*swd

    def training_step(self, batch, batch_nb):
        x, adj = batch
        x = d2_distance_matrix(x)
        pos, enc = self.forward(x, adj)
        y_hat = d2_distance_matrix(pos, False)
        if (batch_nb+1) % 100 == 0:
            self.create_sample_grid(adj)
        return {'loss': self.my_loss(y_hat, x, enc)}

    def validation_step(self, batch, batch_nb):
        return
        x, adj = batch
        nsample = 10
        sample = sample2d(nsample)
        sample = torch.from_numpy(sample).type(torch.FloatTensor)
        adj = adj[0:1].repeat(nsample*nsample,1,1)
        pos = self.decoder.forward(sample.to(adj.device), adj)

        create_grid(nsample, 50, pos, adj, self.current_epoch)
        # return
        # return {'loss': self.my_loss(y_hat, x, enc)}

    def create_sample_grid(self, adj):
        nsample = 10
        sample = sample2d(nsample)
        sample = torch.from_numpy(sample).type(torch.FloatTensor)
        adj = adj[0:1].repeat(nsample*nsample,1,1)
        pos = self.decoder.forward(sample.to(adj.device), adj)

        create_grid(nsample, 50, pos, adj, self.current_epoch)

    def validation_end(self, outputs):
        return {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(GraphData(self.dataset, self.dataset_folder), shuffle=True, batch_size=args.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(GraphData(self.dataset, self.dataset_folder, sample=True), batch_size=args.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(GraphData(self.dataset, self.dataset_folder, sample=True), batch_size=args.batch_size)
