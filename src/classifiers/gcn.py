import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, global_mean_pool
from classifiers.abstract_classifier import GraphClassifier

class GCN(GraphClassifier):
    def __init__(self, x_dim, num_class, edge_attr_dim, model_config: dict):
        super(GCN, self).__init__(model_config)

        hidden_channels = model_config['hidden_size']
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', False)
        
        self.node_encoder = Linear(x_dim, hidden_channels)
        if edge_attr_dim != 0 and self.use_edge_attr:
            self.edge_encoder = Linear(edge_attr_dim, hidden_channels)
            
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc_out = nn.Sequential(nn.Linear(hidden_channels, 1 if num_class == 2 and not self.multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)
            
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.fc_out(x)
        
        return x

    def get_emb(self, x, edge_index, batch, edge_attr=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)
            
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        return x
    
    def get_pred_from_emb(self, emb, batch):
        x = global_mean_pool(emb, batch)  # [batch_size, hidden_channels]
        x = self.fc_out(x)
        
        return x
