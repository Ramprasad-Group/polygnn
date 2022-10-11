# example.py shows how to train a multi-task polyGNN starting from data
# with three columns: "smiles_string", "prop", and "value". But what if
# we have additional metadata that we want the model to use, for example
# temperature or structural parameters? These parameters can be leveraged
# by adding another column, "graph_feats", to your data file.
#
# This file walks through such an example. We model three
# properties: band gap (Eg), electron affinity (Ea) and
# ionization potential (Ei). However, we may predict the Eg of either
# the chain or the bulk (i.e., crystal) representation of the polymer. This metadata
# will be specified in "graph_feats".

from nndebugger import dl_debug
import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import torch

dtype = torch.cuda.FloatTensor
from torch import nn
import numpy as np
import random
from tqdm import tqdm
import polygnn_trainer as pt
import polygnn
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from os import mkdir
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--polygnn", default=False, action="store_true")
parser.add_argument("--polygnn2", default=False, action="store_true")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()
if (not args.polygnn) and (not args.polygnn2):
    raise ValueError("Neither the polygnn nor the polygnn2 flags are set. Choose one.")
elif args.polygnn and args.polygnn2:
    raise ValueError("Both the polygnn and the polygnn2 flags are set. Choose one.")


# #########
# constants
# #########
# For improved speed, some settings below differ from those used in the
# companion paper. In such cases, the values used in the paper are provided
# as a comment.
RANDOM_SEED = 100
HP_EPOCHS = 20  # companion paper used 200
SUBMODEL_EPOCHS = 100  # companion paper used 1000
N_FOLDS = 3  # companion paper used 5
HP_NCALLS = 10  # companion paper used 25
MAX_BATCH_SIZE = 50  # companion paper used 450
capacity_ls = list(range(2, 6))  # companion paper used list(range(2, 14))
weight_decay = 0
N_PASSES = 2  # companion paper used 10
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
#####################

# fix random seeds
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# #####################################################################
# Let's reuse the same sample data used in example.py and then process
# it to include graph_feats.
# #####################################################################
master_data = pd.read_csv("./sample_data/sample.csv", index_col=0)
# Add an empty column for graph_feats.
master_data["graph_feats"] = [None] * len(master_data)
# Convert Egc to Eg and add the corresponding graph_feats.
n_Egc = len(master_data[master_data["prop"] == "Egc"])
master_data.loc[master_data["prop"] == "Egc", "graph_feats"] = [
    {"chain": 1, "bulk": 0}
] * n_Egc
master_data.loc[master_data["prop"] == "Egc", "prop"] = "Eg"
# Convert Egb to Eg and add the corresponding graph_feats.
n_Egb = len(master_data[master_data["prop"] == "Egb"])
master_data.loc[master_data["prop"] == "Egb", "graph_feats"] = [
    {"chain": 0, "bulk": 1}
] * n_Egc
master_data.loc[master_data["prop"] == "Egb", "prop"] = "Eg"
# Add graph_feats for Ea and Ei.
master_data.loc[master_data["prop"] != "Eg", "graph_feats"] = [
    {"chain": 0, "bulk": 0}
] * (len(master_data) - n_Egc - n_Egb)
# Print out 10 random rows of master_data to see how the data has changed
# compared to "sample.csv".
print(master_data.sample(n=10))
print("\n")
# #####################################################################

start = time.time()
# The companion paper trains multi-task (MT) models for six groups. In this
# example file, we will only train an MT model for electronic properties.
PROPERTY_GROUPS = {
    "electronic": [
        "Eg",
        "Ea",
        "Ei",
    ],
}

# Choose the device to train our models on.
if args.device == "cpu":
    device = "cpu"
elif args.device == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU

# Split the data.
train_data, test_data = train_test_split(
    master_data,
    test_size=0.2,
    stratify=master_data.prop,
    random_state=RANDOM_SEED,
)
assert len(train_data) > len(test_data)

if args.polygnn:
    featurization_scheme = "monocycle"
elif args.polygnn2:
    featurization_scheme = "trimer"

smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
    x,
    bond_config,
    atom_config,
    featurization_scheme,
)

# Make a directory to save our models in.
mkdir("example_models/")

# Train one model per group. We only have one group, "electronic", in this
# example file.
for group in PROPERTY_GROUPS:
    prop_cols = sorted(PROPERTY_GROUPS[group])
    print(
        f"Working on group {group}. The following properties will be modeled: {prop_cols}",
        flush=True,
    )

    selector_dim = len(prop_cols)
    # Define a directory to save the models for this group of properties.
    root_dir = "example_models/" + group

    group_train_data = train_data.loc[train_data.prop.isin(prop_cols), :]
    group_test_data = test_data.loc[test_data.prop.isin(prop_cols), :]
    if args.polygnn:
        # Create a dictionary of graph tensors for each smiles string
        # in the data set. Creating this dictionary here instead of inside
        # augmented_featurizer will save time during training because the
        # same graph tensor will not need to be created each epoch.
        print(f"\nMaking eq_graph_tensors for group {group}")
        eq_graph_tensors = {
            x: polygnn.featurize.get_equivalent_graph_tensors(
                x,
                upper_bound=5,
                bond_config=bond_config,
                atom_config=atom_config,
            )
            for x in tqdm(group_train_data.smiles_string.values.tolist())
        }

        augmented_featurizer = lambda x: random.sample(eq_graph_tensors[x], k=1)[0]
    elif args.polygnn2:
        augmented_featurizer = None
    ######################
    # prepare data
    ######################
    group_train_inds = group_train_data.index.values.tolist()
    group_test_inds = group_test_data.index.values.tolist()
    group_data = pd.concat([group_train_data, group_test_data], ignore_index=False)
    group_data, scaler_dict = pt.prepare.prepare_train(
        group_data, smiles_featurizer=smiles_featurizer, root_dir=root_dir
    )
    print([(k, str(v)) for k, v in scaler_dict.items()])
    group_train_data = group_data.loc[group_train_inds, :]
    group_test_data = group_data.loc[group_test_inds, :]

    ######################
    # find optimal capacity
    ######################
    model_class_ls = []
    input_dim = pt.utils.get_input_dim(group_data.data.values[0])
    for capacity in capacity_ls:
        hps = pt.hyperparameters.HpConfig()
        hps.set_values(
            {
                "dropout_pct": 0.0,
                "capacity": capacity,
                "activation": torch.nn.functional.leaky_relu,
            }
        )
        model_class_ls.append(
            lambda: polygnn.models.polyGNN(
                node_size=atom_config.n_features,
                edge_size=bond_config.n_features,
                selector_dim=selector_dim,
                hps=hps,
            )
        )

    session = dl_debug.DebugSession(
        model_class_ls=model_class_ls,
        model_type="gnn",
        capacity_ls=capacity_ls,
        data_set=group_data.data.values.tolist(),
        zero_data_set=None,
        loss_fn=pt.loss.sh_mse_loss(),
        device=device,
        do_choose_model_size_by_overfit=True,
        batch_size=MAX_BATCH_SIZE,
    )
    optimal_capacity = session.choose_model_size_by_overfit()

    # ###############
    # do hparams opt
    # ###############
    # split train and val data
    group_fit_data, group_val_data = train_test_split(
        group_train_data,
        test_size=0.2,
        stratify=group_train_data.prop,
        random_state=RANDOM_SEED,
    )
    fit_pts = group_fit_data.data.values.tolist()
    val_pts = group_val_data.data.values.tolist()
    print(
        f"\nStarting hp opt. Using {len(fit_pts)} data points for fitting, {len(val_pts)} data points for validation."
    )
    # create objective function
    def obj_func(x):
        hps = pt.hyperparameters.HpConfig()
        hps.set_values(
            {
                "r_learn": 10 ** x[0],
                "batch_size": x[1],
                "dropout_pct": x[2],
                "capacity": optimal_capacity,
                "activation": nn.functional.leaky_relu,
            }
        )
        print("Using hyperparameters:", hps)
        tc_search = pt.train.trainConfig(
            hps=hps,
            device=device,
            amp=False,  # False since we are on T2
            multi_head=False,
            loss_obj=pt.loss.sh_mse_loss(),
        )  # trainConfig for the hp search
        tc_search.epochs = HP_EPOCHS

        model = polygnn.models.polyGNN(
            node_size=atom_config.n_features,
            edge_size=bond_config.n_features,
            selector_dim=selector_dim,
            hps=hps,
        )
        val_rmse = pt.train.train_submodel(
            model,
            fit_pts,
            val_pts,
            scaler_dict,
            tc_search,
        )
        return val_rmse

    # create hyperparameter space
    hp_space = [
        (np.log10(0.0003), np.log10(0.03)),  # learning rate
        (round(0.25 * MAX_BATCH_SIZE), MAX_BATCH_SIZE),  # batch size
        (0, 0.5),  # dropout
    ]

    # obtain the optimal point in hp space
    opt_obj = gp_minimize(
        func=obj_func,  # defined offline
        dimensions=hp_space,
        n_calls=HP_NCALLS,
        random_state=RANDOM_SEED,
    )
    # create an HpConfig from the optimal point in hp space
    optimal_hps = pt.hyperparameters.HpConfig()
    optimal_hps.set_values(
        {
            "r_learn": 10 ** opt_obj.x[0],
            "batch_size": opt_obj.x[1],
            "dropout_pct": opt_obj.x[2],
            "capacity": optimal_capacity,
            "activation": nn.functional.leaky_relu,
        }
    )
    print(f"Optimal hps are {opt_obj.x}")
    # clear memory
    del group_fit_data
    del group_val_data

    # ################
    # Train submodels
    # ################
    tc_ensemble = pt.train.trainConfig(
        amp=False,  # False since we are on T2
        loss_obj=pt.loss.sh_mse_loss(),
        hps=optimal_hps,
        device=device,
        multi_head=False,
    )  # trainConfig for the ensemble step
    tc_ensemble.epochs = SUBMODEL_EPOCHS
    print(f"\nTraining ensemble using {len(group_train_data)} data points.")
    pt.train.train_kfold_ensemble(
        dataframe=group_train_data,
        model_constructor=lambda: polygnn.models.polyGNN(
            node_size=atom_config.n_features,
            edge_size=bond_config.n_features,
            selector_dim=selector_dim,
            hps=optimal_hps,
        ),
        train_config=tc_ensemble,
        submodel_trainer=pt.train.train_submodel,
        augmented_featurizer=augmented_featurizer,
        scaler_dict=scaler_dict,
        root_dir=root_dir,
        n_fold=N_FOLDS,
        random_seed=RANDOM_SEED,
    )
    ##########################################
    # Load and evaluate ensemble on test data
    ##########################################
    print("\nRunning predictions on test data", flush=True)
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
    # remake "group_test_data" so that "graph_feats" contains dicts not arrays
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
        ensemble_kwargs_dict={"n_passes": N_PASSES},
    )
    pt.utils.mt_print_metrics(
        y, y_mean_hat, _selectors, scaler_dict, inverse_transform=False
    )
    print(f"Done working on group {group}\n", flush=True)

end = time.time()
print(f"Done with everything in {end-start} seconds.", flush=True)
