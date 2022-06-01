import numpy as np
import os
import torch
import random

# fix random seed
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


def make_scalers(train_df, name_map, scaler_class):
    scaler_dict = {}
    n_rows = len(train_df)
    for name, key in name_map.items():
        # do reshape below since Scaler needs 2D input
        transformer = scaler_class().fit(train_df[key].to_numpy().reshape((n_rows, 1)))
        scaler_dict[name] = transformer
    return scaler_dict


def scale_df(property_df, name_map, scaler_dict):
    edit_df = property_df.copy(deep=True)
    n_rows = len(property_df)
    for name, key in name_map.items():
        transformer = scaler_dict[name]
        # do first reshape below since Scaler needs 2D input
        # do second reshape to map 2D data back to 1D
        edit_df[key] = transformer.transform(
            property_df[key].to_numpy().reshape((n_rows, 1))
        ).reshape((n_rows,))

    return edit_df


def assign_class(train_frac, val_frac, test_frac):
    if train_frac + val_frac + test_frac != 1:
        return ValueError("Fractions should add to one")
    rand = np.random.uniform(low=0, high=1)
    if rand < train_frac:
        the_class = "train"
    elif rand < train_frac + val_frac:
        the_class = "val"
    else:
        the_class = "test"
    return the_class


def lazy_property(fn):
    """
    Implementation borrowed from https://towardsdatascience.com/what-is-lazy-evaluation-in-python-9efb1d3bfed0
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
