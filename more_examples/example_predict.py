# The purpose of this file is to provide an example of how to just do prediction
# using previously-trained models. A prediction example is already contained in
# example.py but it is far down in the file (something like line 340), so having
# this new shorter file will be helpful.

# Before running this file, you should have ran "example.py" and which generated a
# directory called 'example_models'. This file will load the model and metadata
# stored in that directory.

import argparse
import random
import numpy as np
import torch

import polygnn
import polygnn_trainer as pt

# Parse command line arguments.
parser = argparse.ArgumentParser()
# parser.add_argument("--polygnn", default=False, action="store_true")
# parser.add_argument("--polygnn2", default=False, action="store_true")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()
# if (not args.polygnn) and (not args.polygnn2):
#     raise ValueError("Neither the polygnn nor the polygnn2 flags are set. Choose one.")
# elif args.polygnn and args.polygnn2:
#     raise ValueError("Both the polygnn and the polygnn2 flags are set. Choose one.")

RANDOM_SEED = 100
root_dir = "example_models/electronic"
bond_config = polygnn.featurize.BondConfig(True, True, True)
atom_config = polygnn.featurize.AtomConfig(
    True,
    True,
    True,
    True,
    True,
    True,
    combo_hybrid=False,  # if True, SP2/SP3 are combined into one feature
    aromatic=True,
)
device = args.device

# fix random seeds
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load and evaluate ensemble on test data.
print("\nRunning predictions on test data.", flush=True)
ensemble = pt.load.load_ensemble(
    root_dir,
    polygnn.models.polyGNN,
    device,
    {
        "node_size": atom_config.n_features,
        "edge_size": bond_config.n_features,
        "selector_dim": selector_dim,
    },
)
# Remake "group_test_data" so that "graph_feats" contains dicts not arrays.
group_test_data = test_data.loc[
    test_data.prop.isin(prop_cols),
    :,
]
y, y_mean_hat, y_std_hat, _selectors = pt.infer.eval_ensemble(
    model=ensemble,
    root_dir=root_dir,
    dataframe=group_test_data,
    smiles_featurizer=smiles_featurizer,
    device=device,
    ensemble_kwargs_dict={"monte_carlo": False},
)
pt.utils.mt_print_metrics(
    y, y_mean_hat, _selectors, scaler_dict, inverse_transform=False
)