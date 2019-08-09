

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dgl.dataset import GraphData
from dgl.config import args
from dgl.model import Encoder, Decoder
from dgl.features import _sliced_wasserstein_distance, rand_uniform2d, d2_distance_matrix


class GraphLayoutVAE(pl.LightningModule):

    def __init__(self, in_dim, out_dim, dataset, dataset_folder):
        super(GraphLayoutVAE, self).__init__()
        self.encoder = Encoder(in_dim, out_dim)
        self.decoder = Decoder(in_dim, out_dim)
        self.dataset_folder = dataset_folder
        self.dataset = dataset


    def forward(self, x, adj):
        enc = self.encoder(x, adj)
        dec = self.decoder(enc, adj)
        return d2_distance_matrix(dec), enc

    def my_loss(self, y_hat, y, encoded_samples):
        # draw random samples from latent space prior distribution
        z = rand_uniform2d(args.batch_size).to(encoded_samples.device)
        swd = _sliced_wasserstein_distance(encoded_samples, z,
                                           50, 2, args.device)
        return torch.mean((y_hat-y)**2) + swd

    def training_step(self, batch, batch_nb):
        x, adj = batch
        x = d2_distance_matrix(x)
        y_hat, enc = self.forward(x, adj)
        y_hat = d2_distance_matrix(y_hat)
        return {'loss': self.my_loss(y_hat, x, enc)}

    def validation_step(self, batch, batch_nb):
        x, adj = batch
        x = d2_distance_matrix(x)
        y_hat, enc = self.forward(x, adj)
        y_hat = d2_distance_matrix(y_hat)
        return {'loss': self.my_loss(y_hat, x, enc)}

    def validation_end(self, outputs):
        return

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(GraphData(self.dataset, self.dataset_folder), batch_size=args.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(GraphData(self.dataset, self.dataset_folder, sample=True), batch_size=args.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(GraphData(self.dataset, self.dataset_folder, sample=True), batch_size=args.batch_size)
