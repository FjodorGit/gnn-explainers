from abc import abstractmethod
import torch

class GraphClassifier(torch.nn.Module):

    def __init__(self, model_config: dict):
        super().__init__()
        self.multi_label = model_config['multi_label']

    @abstractmethod
    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        pass

    @abstractmethod
    def get_embedding(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        pass
        
    @abstractmethod
    def get_pred_from_emb(self, emb, batch):
        pass
