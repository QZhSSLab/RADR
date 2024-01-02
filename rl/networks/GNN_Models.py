import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv


class GCN_Net(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels):
        super(GCN_Net, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Output
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GAT_Net(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels, heads=8):
        super(GAT_Net, self).__init__()
        self.dropout_rate = 0.6
        self.conv1 = GATConv(num_node_features, hidden_channels, heads, dropout=self.dropout_rate)
        self.conv2 = GATConv(hidden_channels * heads, output_channels, dropout=self.dropout_rate)

    def forward(self, data_list):
        outputs = []
        for data in data_list:
            x = F.dropout(data.x, p=self.dropout_rate, training=self.training)
            x1 = self.conv1(x, data.edge_index)
            x2 = F.elu(x1)
            x3 = F.dropout(x2, p=self.dropout_rate, training=self.training)
            x4 = self.conv2(x3, data.edge_index)
            x5 = F.elu(x4)
            outputs.append(x5)
        return outputs
