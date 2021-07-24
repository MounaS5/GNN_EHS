
import torch_geometric
import torch

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_min
class GNN(torch.nn.Module):
    def __init__(self, num_layer, drop_ratio, conv_dim, gnn_type, JK,graph_pooling):
        super(GNN, self).__init__()

        # Initialization
        self.num_layer  = num_layer
        self.drop_ratio = drop_ratio
        self.conv_dim   = conv_dim
        self.gnn_type   = gnn_type
        self.JK         = JK
        #self.num_tasks  = num_tasks    # 1 (output value??)
        self.graph_pooling = graph_pooling

        self.graph_pool_list = ["add","mean","max"]
        

        self.node_dim   =  n_atom_features()      # Num features nodes
        self.edge_dim   =  n_bond_features()      # Num features edges

        # List of GNNs
        self.convs       = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # First embedding atom_dim to conv_dim
        self.linatoms = nn.Linear(self.node_dim , conv_dim)

        # GNN layers
        for layer in range(num_layer):
            if gnn_type == 'NNConv':
              neurons_message = 5
              mes_nn = nn.Sequential(nn.Linear(self.edge_dim, neurons_message), nn.ReLU(), nn.Linear(neurons_message, conv_dim**2))
              if graph_pooling in self.graph_pool_list:
                self.convs.append(gnn.NNConv(conv_dim, conv_dim, mes_nn,graph_pooling))
         
            else:
                ValueError(f'Undefined GNN type called {gnn_type}') 
            self.batch_norms.append(torch.nn.BatchNorm1d(conv_dim))
        self.mlp1 = nn.Linear(conv_dim, conv_dim)
        self.pred = nn.Linear(conv_dim, 1)
    
    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        
        # Original node dimension to first conv dimension
        x = F.leaky_relu(self.linatoms(x))
        
        # GNN layers
        x_list = [x]
        for layer in range(self.num_layer):
            x = self.convs[layer](x_list[layer], edge_index, edge_attr)
            x = self.batch_norms[layer](x)

            # Remove activation function from last layer
            if layer == self.num_layer - 1:
                x = F.dropout(x, p=self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.leaky_relu(x), p=self.drop_ratio, training=self.training)
            x_list.append(x)
        

        # def message(self, x_j, weights, ):
        #     #print(x_j.shape)
        #     #print(weights.shape)
        #     if 
        #     weight = weights.view(-1, self.in_channels, self.out_channels)
        #     #print(weights.shape)
        #     #print(x_j.shape)
        #     return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

        #print(x_list[-1].size)
        ### Jumping knowledge https://arxiv.org/pdf/1806.03536.pdf
        if self.JK == "last":
            x = x_list[-1]
        elif self.JK == "sum":
            x = 0
            for layer in range(self.num_layer):
                x += x_list[layer]
        elif self.JK == "mean":
            x = 0
            for layer in range(self.num_layer):
                x += x_list[layer]
            x = x/self.num_layer
        
        # Graph embedding
        x = scatter_add(x, batched_data.batch, dim=0)
        
        x = self.pred(x)
  
        return x