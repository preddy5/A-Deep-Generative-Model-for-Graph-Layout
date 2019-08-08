
from model import Encoder, Decoder
from features import d2_distance_matrix

import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from dataset import GraphData

import pytorch_lightning as pl

class GraphLayoutVAE(pl.LightningModule):

    def __init__(self, in_dim, out_dim, data_folder, data):
        super(GraphLayoutVAE, self).__init__()
        # not the best model...
        self.encoder = Encoder(in_dim, out_dim)
        self.decoder = Decoder(in_dim, out_dim)
        self.data_folder = data_folder
        self.data = data


    def forward(self, x, adj):
        enc = self.encoder(x, adj)
        ## TODO
        dec = self.decoder(enc, adj)
        return d2_distance_matrix(dec)

    def my_loss(self, y_hat, y):
        return torch.mean((y_hat-y)**2)

    def training_step(self, batch, batch_nb):
        x, adj = batch
        y_hat = self.forward(x, adj)
        return {'loss': self.my_loss(y_hat, x)}

    def validation_step(self, batch, batch_nb):
        x, adj = batch
        y_hat = self.forward(x, adj)
        return {'loss': self.my_loss(y_hat, x)}

    def validation_end(self, outputs):
        x, adj = batch
        y_hat = self.forward(x, adj)
        return {'loss': self.my_loss(y_hat, x)}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(GraphData(self.data, self.data_folder), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(GraphData, batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(GraphData, batch_size=32)
