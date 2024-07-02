# The purpose of this file is to provide an example of how to obtain predictions and
# fingerprints from previously-trained models.

import torch
import pandas as pd
import argparse
import numpy as np
import random
from torch_geometric.loader import DataLoader

import polygnn
import polygnn_trainer as pt
from polygnn_trainer.prepare import prepare_infer

# Specify that we do want to include fingerprints in the prediction result
INCLUDE_POLYMER_FPS = True

# Create data to predict on. For this example, we will predict on a single polymer
# using the CH4 permeability model.
data = pd.DataFrame({"smiles_string": ["[*]CC[*]"], "prop": ["exp_perm_CH4__Barrer"]})

# Other hardcoded parameters
RANDOM_SEED = 100
root_dir = "trained_models/solubility_and_permeability"
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

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--polygnn", default=False, action="store_true")
parser.add_argument("--polygnn2", default=False, action="store_true")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()

# Check command line arguments.
if (not args.polygnn) and (not args.polygnn2):
    raise ValueError("Neither the polygnn nor the polygnn2 flags are set. Choose one.")
elif args.polygnn and args.polygnn2:
    raise ValueError("Both the polygnn and the polygnn2 flags are set. Choose one.")

# Choose the device to predict on.
if args.device == "cpu":
    device = "cpu"
elif args.device == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU

# Fix random seeds
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load selectors
selectors = pt.load2.load_selectors(root_dir)

# Load scalers
scaler_dict = pt.load2.load_scalers(root_dir)

# Determine the featurization scheme based on command line arguments.
if args.polygnn:
    featurization_scheme = "monocycle"
elif args.polygnn2:
    featurization_scheme = "trimer"

# Define a lambda function for smiles featurization.
smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
    x,
    bond_config,
    atom_config,
    featurization_scheme,
)

# Convert the dataframe into a DataLoader
dataframe = prepare_infer(
    data,
    smiles_featurizer,
    selectors,
    root_dir=root_dir,
)
data_ls = dataframe.data.values.tolist()
loader = DataLoader(data_ls, batch_size=len(data), shuffle=False)

# Compute the number of selector dimensions
selector_dim = len(scaler_dict)

# Load one of the submodels. Other submodels can be loaded
# by changing the index (e.g., 2, 3, 4, or 5).
index = 1
model = pt.load.load_model(
    path=f"{root_dir}/models/model_{index}.pt",
    submodel_cls=polygnn.models.polyGNN,
    node_size=atom_config.n_features,
    edge_size=bond_config.n_features,
    selector_dim=selector_dim,
    hps=pt.load2.load_hps(root_dir),
)

# Make the predictions and fingerprints
print("\nRunning predictions on test data.", flush=True)
y_val, y_val_hat_mean, selectors, polymer_fps = polygnn.infer.eval_submodel(
    model, loader, device, selector_dim, include_polymer_fps=INCLUDE_POLYMER_FPS
)

# Print the results
print()
print("Mean prediction(s): ", y_val)
print("The fingerint(s): ", polymer_fps)
