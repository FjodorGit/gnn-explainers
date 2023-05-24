import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GraphConv, Linear
from classifiers.abstract_classifier import GraphClassifier

class GraphConvNet(GraphClassifier):
    def __init__(self, x_dim, num_class, edge_attr_dim, model_config: dict):
        super(GraphConvNet, self).__init__(model_config)

        hidden_size = model_config['hidden_size']
        self.dropout_p = model_config['dropout_p']

        self.conv1 = GraphConv(x_dim, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, hidden_size)
        self.conv4 = GraphConv(hidden_size, hidden_size)
        self.conv5 = GraphConv(hidden_size, hidden_size)
        
        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not self.multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc_out(x)
        return x

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        return x
    
    def get_pred_from_emb(self, emb, batch):
        x = global_add_pool(emb, batch)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc_out(x)
        return x
