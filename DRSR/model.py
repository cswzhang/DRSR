# coding=utf-8
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_geometric.nn import GCNConv, GATConv


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layer):
        super().__init__()
        self.decoder = nn.ModuleList()
        for i in range(layer):
            if layer > 1:
                if i == 0:
                    self.decoder.append(nn.Linear(input_dim, hidden_dim))
                elif i == layer - 1:
                    self.decoder.append(nn.Linear(hidden_dim, out_dim))
                else:
                    self.decoder.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.decoder.append(nn.Linear(input_dim, out_dim))

    def forward(self, H):
        for index, layer in enumerate(self.decoder):
            H = layer(H)
            H = F.relu(H)
        return H


class MGCAT(nn.Module):
    def __init__(self, h1_dim, h2_dim, h3_dim, input_dim, hidden_dim, out_dim, enc_layer, dec_layer):
        super().__init__()
        self.transform1 = nn.Linear(h1_dim, input_dim)
        self.transform2 = nn.Linear(h2_dim, input_dim)
        self.transform3 = nn.Linear(h3_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.encoder = nn.ModuleList()
        for i in range(enc_layer):
            if i == 0:
                self.encoder.append(CrossAttention(input_dim, hidden_dim))
            elif i == enc_layer - 1:
                self.encoder.append(CrossAttention(hidden_dim, out_dim))
            else:
                self.encoder.append(CrossAttention(hidden_dim, hidden_dim))
        self.lin = nn.Sequential(
            nn.Linear(3 * out_dim, out_dim),
            nn.ReLU()
            # nn.LayerNorm(out_dim)
        )
        self.decoder = Decoder(out_dim, hidden_dim, h1_dim, h2_dim, h3_dim, dec_layer)

    def forward(self, data):
        H1, H2, H3, edge_index = data['adj'], data['refex'], data['motif'], data.edge_index
        H1 = self.transform1(H1)
        H2 = self.transform2(H2)
        H3 = self.transform3(H3)
        # H1 = self.norm(H1)
        # H2 = self.norm(H2)
        # H3 = self.norm(H3)
        for index, layer in enumerate(self.encoder):
            H1, H2, H3 = layer(edge_index, H1, H2, H3)
        H = self.lin(torch.cat([H1, H2, H3], dim=1))
        out2, out3 = self.decoder(H)
        return torch.cat([H1, H2, H3], dim=1), out2, out3


class DisGATConv(MessagePassing):
    def __init__(self, input_dim, out_dim, channel):
        super().__init__(node_dim=-2)
        self.channel = channel
        self.input_dim = input_dim
        self.lin = nn.ModuleList()
        self.q = nn.ModuleList()
        for i in range(channel):
            self.lin.append(nn.Linear(input_dim, out_dim))
            self.q.append(nn.Linear(out_dim, out_dim))

    def forward(self, edge_index, X):
        self.alpha_ = []
        alpha = []
        H = []
        for i in range(self.channel):
            H.append(self.lin[i](X[i]))
        H = torch.stack(H)
        for i in range(self.channel):
            tmp = self.q[i](H[i]) / sqrt(H[i].shape[-1])
            alpha.append(tmp)
        alpha = torch.stack(alpha)
        H = H + self.propagate(edge_index, x=H, alpha=alpha)
        self.alpha_ = torch.stack(self.alpha_)

        return H, self.alpha_

    def message(self, x_j, x_i, alpha_i):
        h = []
        for k in range(self.channel):
            p_ij = (x_j * alpha_i[k]).sum(dim=-1)
            p_ij = F.softmax(F.leaky_relu(p_ij), dim=0)
            h.append(p_ij[k].unsqueeze(-1) * x_j[k])
            # h.append((p_ij.unsqueeze(-1) * x_j).sum(dim=0))
            self.alpha_.append(p_ij)
        return torch.stack(h)


class DisGAT(nn.Module):
    def __init__(self, channel, input_dim, hidden_dim, out_dim, dec1_dim, dec2_dim, dec3_dim, num_classes, enc_layer,
                 dec_layer):
        super().__init__()
        self.enc = nn.ModuleList()
        self.channel = channel

        for i in range(enc_layer):
            if i == 0:
                self.enc.append(DisGATConv(input_dim, hidden_dim, channel))
            elif i == enc_layer - 1:
                self.enc.append(DisGATConv(hidden_dim, out_dim, channel))
            else:
                self.enc.append(DisGATConv(hidden_dim, hidden_dim, channel))
        self.dec2 = Decoder(channel * out_dim, hidden_dim, dec2_dim, dec_layer)
        self.dec3 = Decoder(channel * out_dim, hidden_dim, dec3_dim, dec_layer)

        # self.mlp = Decoder(channel * out_dim, hidden_dim, num_classes, dec_layer)

        # self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        edge_index, H = data.edge_index, data['H']
        for i, layer in enumerate(self.enc):
            H, alpha = layer(edge_index, H)
        H = torch.cat([H[k] for k in range(self.channel)], dim=1)
        # return H, alpha, F.softmax(self.mlp(H), dim=1)
        return H, alpha, self.dec2(H), self.dec3(H)


class GCN(nn.Module):
    def __init__(self, channel, input_dim, hidden_dim, out_dim, dec1_dim, dec2_dim, dec3_dim, num_classes, enc_layer,
                 dec_layer):
        super().__init__()
        self.enc = nn.ModuleList()
        for i in range(enc_layer):
            if i == 0:
                self.enc.append(GCNConv(input_dim, hidden_dim * channel))
            elif i == enc_layer - 1:
                self.enc.append(GCNConv(hidden_dim * channel, out_dim * channel))
            else:
                self.enc.append(GCNConv(hidden_dim * channel, hidden_dim * channel))
        self.dec2 = Decoder(out_dim * channel, hidden_dim, dec2_dim, dec_layer)
        self.dec3 = Decoder(out_dim * channel, hidden_dim, dec3_dim, dec_layer)

    def forward(self, data):
        edge_index, H = data.edge_index, data['H']
        for i, layer in enumerate(self.enc):
            H = layer(H, edge_index)
            H = F.relu(H)
        return H, H, self.dec2(H), self.dec3(H)


class GAT(nn.Module):
    def __init__(self, channel, input_dim, hidden_dim, out_dim, dec1_dim, dec2_dim, dec3_dim, num_classes, enc_layer,
                 dec_layer):
        super().__init__()
        self.enc = nn.ModuleList()
        for i in range(enc_layer):
            if i == 0:
                self.enc.append(GATConv(input_dim, hidden_dim, channel))
            elif i == enc_layer - 1:
                self.enc.append(GATConv(hidden_dim * channel, out_dim, channel))
            else:
                self.enc.append(GATConv(hidden_dim * channel, hidden_dim, channel))
        self.dec2 = Decoder(out_dim * channel, hidden_dim, dec2_dim, dec_layer)
        self.dec3 = Decoder(out_dim * channel, hidden_dim, dec3_dim, dec_layer)

    def forward(self, data):
        edge_index, H = data.edge_index, data['H']
        for i, layer in enumerate(self.enc):
            H = layer(H, edge_index)
            H = F.relu(H)
        return H, H, self.dec2(H), self.dec3(H)


class DisenGCN(MessagePassing):
    def __init__(self, channel, input_dim, hidden_dim, out_dim, dec1_dim, dec2_dim, dec3_dim, num_classes, enc_layer,
                 dec_layer):
        super().__init__(node_dim=-2)
        self.enc = nn.ModuleList()
        self.enc_layer = enc_layer
        self.channel = channel
        for k in range(channel):
            self.enc.append(nn.Linear(input_dim, out_dim))
            # nn.init.xavier_normal_(self.enc[k].weight.data)
            # self.enc[k].bias.data.fill_(0.0)

        self.dec2 = Decoder(out_dim * channel, hidden_dim, dec2_dim, dec_layer)
        self.dec3 = Decoder(out_dim * channel, hidden_dim, dec3_dim, dec_layer)

    def forward(self, data):
        edge_index, X = data.edge_index, data['H']
        H = []
        for k in range(self.channel):
            H.append(F.tanh(self.enc[k](X[k])))
        H = torch.stack(H)
        H = H / H.pow(2).sum(dim=-1, keepdim=True).sqrt()

        for _ in range(self.enc_layer):
            H = H + self.propagate(edge_index, x=H)
            H = H / (H.pow(2).sum(dim=-1, keepdim=True).sqrt())

        H = torch.cat([H[k] for k in range(self.channel)], dim=1)
        return H, H, self.dec2(H), self.dec3(H)

    def message(self, x_j, x_i):
        h = []
        for k in range(self.channel):
            p_ij = (x_j * x_i[k]).sum(dim=-1)
            p_ij = F.softmax(p_ij, dim=0)
            h.append(p_ij[k].unsqueeze(-1) * x_j[k])
        return torch.stack(h)
