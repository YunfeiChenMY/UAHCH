import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from spectral_norm import spectral_norm as SpectralNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from pygat.layers import GraphAttentionLayer
from torch.nn.functional import normalize
import scipy.sparse as sp
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import HypergraphConv
from tools import build_G_from_S, generate_robust_S


class ImgNetHY(nn.Module):
    def __init__(self, code_len, img_feat_len, eplision, k):
        super(ImgNetHY, self).__init__()
        b = 4096
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc_encode1 = HypergraphConv(img_feat_len, b)
        self.fc_encode2 = HypergraphConv(b, int(code_len))
        self.alpha = 1.0
        self.dropout = 0.3
        self.eplision = eplision
        self.k = k


    def forward(self, x):
        G = build_G_from_S(F.normalize(x).mm(F.normalize(x).t()), self.k , self.eplision)
        self.alpha = 1
        G = torch.LongTensor(edge_list(G)).cuda()
        feat = torch.relu(self.fc_encode1(x, G))
        feat = self.fc_encode2(self.dp(feat), G)
        code = torch.tanh(self.alpha * feat)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


def get_hyperedge_attr(features, hyperedge_index, type='mean'):
    if type == 'mean':
        samples = features[hyperedge_index[0]]
        labels = hyperedge_index[1]

        labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        hyperedge_attr = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
        hyperedge_attr = hyperedge_attr / labels_count.float().unsqueeze(1)
    return hyperedge_attr

def edge_list(G):

    mask = G != -1.5

    # Use the mask to find the indices
    list_e, cla = mask.nonzero(as_tuple=True)

    # Convert indices to lists if needed (they are returned as tensors)
    list_e = list_e.tolist()
    cla = cla.tolist()

    # Combine the lists into a final result
    res = [list_e, cla]
    return res
class TxtNetHY(nn.Module):
    def __init__(self, code_len, txt_feat_len, eplision, k):
        super(TxtNetHY, self).__init__()
        a = 4096
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = HypergraphConv(txt_feat_len, a)
        self.fc3 = HypergraphConv(a, int(code_len))

        self.alpha = 1.0
        self.dropout = 0.3
        self.eplision = eplision
        self.k = k

    def forward(self, x):
        G = build_G_from_S(F.normalize(x).mm(F.normalize(x).t()), self.k, self.eplision)
        self.alpha = 1
        G = torch.LongTensor(edge_list(G)).cuda()
        # G = torch.LongTensor(G.cpu()).cuda()
        feat = torch.relu(self.fc1(x, G))
        feat = self.fc3(self.dp(feat), G)
        code = torch.tanh(self.alpha * feat)


        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class FuseTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead, code_len):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model / 2)
        self.fc_encode2 = nn.Linear(self.d_model, int(code_len))

    def forward(self, tokens):
        encoder_X = self.transformerEncoder(tokens)
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        encoder_X_r = normalize(encoder_X_r, p=2, dim=1)
        img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        hashH = torch.tanh(self.fc_encode2(encoder_X_r))
        hashB = torch.sign(hashH)
        return hashB, hashH, img, txt
