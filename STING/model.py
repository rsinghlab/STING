#This code has been partly adapted from https://github.com/JinmiaoChenLab/GraphST since STING uses GraphST as the outer GNN


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import Sequential, BatchNorm, GATv2Conv
from torch.nn import Linear, ReLU,CrossEntropyLoss
from torch_geometric.loader import DataLoader as GNNDataLoader

class inner_GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_genes, device):
        super().__init__()
        self.num_node_features = num_node_features
        self.n_hidden = 1
        self.num_genes = num_genes

        self.conv_first = GATv2Conv(num_node_features, self.n_hidden,weight_initializer = None) #kaiming_uniform

        self.bn1 = BatchNorm(self.n_hidden)
        
        
        self.conv_last = GATv2Conv(self.n_hidden, 1,weight_initializer = None)
        self.bn2 = BatchNorm(1)
        
        self.device = device
        

    def forward(self, x):

        data_loader = GNNDataLoader(x, batch_size=50000, shuffle=False)
        
        
        
        for batch_idx, gnn_batch in enumerate(data_loader):
            y, edge_index = gnn_batch.x, gnn_batch.edge_index

            x, a1 = self.conv_first(y, edge_index, return_attention_weights = True)
            x = self.bn1(x)
            x = F.relu(x)
        
    
            x, a2 = self.conv_last(x, edge_index, return_attention_weights = True)
            x = self.bn2(x)
            x = F.relu(x)
        
        x = torch.reshape(x, (gnn_batch.num_graphs, -1))

        return x, a1, 0#a2
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    
class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, device, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.inner_gnn = inner_GNN(1, in_features, device).to(device)
        self.device = device
        self.n_hidden = 256
        
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj, graph_feat, graph_feat_a):  #forward(self, feat, feat_a, adj, graph_feat, graph_feat_a)
        z, a1, a2 = self.inner_gnn(graph_feat)
        inner = z
        
        z = F.dropout(z, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
     
        
        emb = self.act(z)
        
        z_a, _, _ = self.inner_gnn(graph_feat_a)
        
        z_a = F.dropout(z_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        
        
        emb_a = self.act(z_a)
        
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a, inner, a1, a2
