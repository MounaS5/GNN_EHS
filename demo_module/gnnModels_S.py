import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import global_add_pool, GraphConv

class Net(torch.nn.Module):
    def __init__(self, dim, node_dim, num_layer):
        super(Net, self).__init__()

        self.num_features = node_dim
        self.dim = dim
        self.num_layer= num_layer
        #self.task_type = task_type
        self.conv1 = GraphConv(node_dim, dim)

        # List of GNNs
        self.convs       = torch.nn.ModuleList()
        for i in range(self.num_layer-1): 
          self.convs.append(GraphConv(dim, dim))
          
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 1)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        #x_list = [x]
        for layer in range(self.num_layer-1):
            #x = F.relu(self.convs[layer](x_list[layer], edge_index, edge_weight))
            x = F.relu(self.convs[layer](x, edge_index, edge_weight))
            x = F.dropout(x, p=0.5, training=self.training)
            #x_list.append(x)
        # print(edge_weight)
        # print('****************')
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
 
        return x#F.log_softmax(x, dim=-1)
