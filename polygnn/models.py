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
    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        normalize_embedding=True,
        debug=False,
    ):
        super().__init__(hps)

        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.normalize_embedding = normalize_embedding
        self.debug = debug

        self.mpnn = layers.MtConcat_PolyMpnn(
            node_size,
            edge_size,
            selector_dim,
            self.hps,
            normalize_embedding,
            debug,
        )

        # set up linear blocks
        self.final_mlp = pt.layers.Mlp(
            input_dim=self.mpnn.readout_dim + self.selector_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )
        self.out_layer = pt.layers.my_output(size_in=32, size_out=1)

    def forward(self, data):
        x, edge_index, edge_weight, batch, selector = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
            data.selector,
        )  # extract variables
        x = self.mpnn(x, edge_index, edge_weight, batch)
        x = F.leaky_relu(x)
        x = torch.cat((x, selector), dim=1)
        x = self.final_mlp(x)  # hidden layers
        x = self.out_layer(x)  # output layer
        x = torch.clip(  # prevent inf and -inf
            x,
            min=-0.5,
            max=1.5,  # choose -0.5 and 1.5 since the output should be between 0 and 1
        )
        x[torch.isnan(x)] = 1.5  # prevent nan
        return x.view(data.num_graphs, 1)  # get the shape right
