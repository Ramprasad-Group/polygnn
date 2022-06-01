from polygnn_trainer.hyperparameters import HpConfig, ModelParameter
from torch_geometric.nn import MessagePassing


class StandardMpModule(MessagePassing):
    def __init__(self, hps: HpConfig, aggr: str, node_dim: int):
        super().__init__(aggr=aggr, node_dim=node_dim)
        if hps:
            # delete attributes that are not of type ModelParameter
            del_attrs = []
            for attr_name, obj in hps.__dict__.items():
                if not isinstance(obj, ModelParameter):
                    # log those attributes that are not of
                    # type ModelParameter so we can delete
                    # them later. They need to be deleted
                    # later so that the dictionary size does
                    # not change during the for loop.
                    del_attrs.append(attr_name)
            for attr in del_attrs:
                delattr(hps, attr)
        # assign hps to self
        self.hps = hps
