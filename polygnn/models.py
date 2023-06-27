import torch
import torch.nn.functional as F
import random
import numpy as np
import polygnn_trainer as pt

import polygnn.layers as layers

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

# ##########################
# Multi-task models
# #########################
class polyGNN(pt.std_module.StandardModule):
    """
    Multi-task GNN model.
    """

    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        normalize_embedding=True,
        graph_feats_dim=0,
        debug=False,
    ):
        """
        Initialize the PolyGNN model.

        Args:
            node_size (int): Size of the node features.
            edge_size (int): Size of the edge features.
            selector_dim (int): Dimension of the selector.
            hps (HpConfig): Hyperparameters object.
            normalize_embedding (bool, optional): Flag to normalize embeddings. Defaults to True.
            graph_feats_dim (int, optional): Dimension of the graph features. Defaults to 0.
            debug (bool, optional): Flag to enable debug mode. Defaults to False.
        """
        super().__init__(hps)
        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.normalize_embedding = normalize_embedding
        assert isinstance(graph_feats_dim, int)
        self.graph_feats_dim = graph_feats_dim
        self.debug = debug

        self.mpnn = layers.MtConcat_PolyMpnn(
            node_size,
            edge_size,
            selector_dim,
            self.hps,
            normalize_embedding,
            debug,
        )

        self.final_mlp = pt.layers.Mlp(
            input_dim=self.mpnn.readout_dim + self.selector_dim + self.graph_feats_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )

        self.out_layer = pt.layers.my_output(size_in=32, size_out=1)

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (Data): Input data.

        Returns:
            tensor: Output tensor.
        """
        data.yhat = self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)
        data.yhat = F.leaky_relu(data.yhat)
        data.yhat = self.assemble_data(data)
        data.yhat = self.final_mlp(data.yhat)
        data.yhat = self.out_layer(data.yhat)
        return data.yhat.view(data.num_graphs, 1)
