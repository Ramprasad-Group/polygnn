from torch import cat
from polygnn_trainer.hyperparameters import HpConfig, ModelParameter
from torch_geometric.nn import MessagePassing


class StandardMpModule(MessagePassing):
    def __init__(self, hps: HpConfig, aggr: str, node_dim: int):
        """
        Initialize the StandardMpModule.

        Args:
            hps (HpConfig): Hyperparameters object.
            aggr (str): Aggregation method.
            node_dim (int): Dimension of the node feature.
        """
        super().__init__(aggr=aggr, node_dim=node_dim)
        if hps:
            # Delete attributes that are not of type ModelParameter
            del_attrs = []
            for attr_name, obj in hps.__dict__.items():
                if not isinstance(obj, ModelParameter):
                    del_attrs.append(attr_name)
            for attr in del_attrs:
                delattr(hps, attr)
        self.hps = hps
