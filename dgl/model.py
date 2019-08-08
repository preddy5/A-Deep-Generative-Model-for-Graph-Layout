
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import DenseGINConv, global_mean_pool


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dim=32):
        super(Encoder, self).__init__()

        nn1 = Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = DenseGINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = DenseGINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = DenseGINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, out_dim)
        # self.fc2_var = Linear(dim, out_dim)

    def forward(self, x, adj):
        x1 = F.relu(self.conv1(x, adj))
        x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, adj))
        x2 = self.bn2(x2)
        x3 = F.relu(self.conv3(x2, adj))
        x3 = self.bn3(x3)

        x_c = torch.cat([x1,x2,x3], dim=-1)
        # Readout
        x = torch.sum(x_c, [1, 3])
        # input.view(input.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dim=32):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.id_mat = torch.eye(in_dim)
        nn1 = Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = DenseGINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = DenseGINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = DenseGINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(in_dim*in_dim, in_dim*in_dim)
        self.fc2 = Linear(in_dim*in_dim, in_dim*out_dim)

    def forward(self, x, adj):
        # fusion layer
        x = self.id_mat*x

        x1 = F.relu(self.conv1(x, adj))
        x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, adj))
        x2 = self.bn2(x2)
        x3 = F.relu(self.conv3(x2, adj))
        x3 = self.bn3(x3)

        x_c = torch.cat([x1,x2,x3], dim=-1)
        x = torch.sum(x_c, -1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = x.view(x.size(0), self.in_dim, self.out_dim)
        return x

