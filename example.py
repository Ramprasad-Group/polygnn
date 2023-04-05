from nndebugger import dl_debug
import argparse
import time
import random
import pandas as pd
import numpy as np
from os import mkdir
from tqdm import tqdm
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import polygnn
import polygnn_trainer as pt


pd.options.mode.chained_assignment = None

# parse command line arguments
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
# as a comment. If you are using this file as a template to train a new
# model, then it is recommended to use the values from the companion paper.
RANDOM_SEED = 100
# HP_NCALLS is the number of points to sample in the hyperparameter (HP)
# space.
HP_NCALLS = 10  # companion paper used 25
# HP_EPOCHS is the number of training epochs per HP point.
HP_EPOCHS = 20  # companion paper used 200
# N_FOLDS is the number of folds to use during cross-validation. One submodel
# is trained per fold.
N_FOLDS = 3  # companion paper used 5
# SUBMODEL_EPOCHS is the number of training epochs per submodel.
SUBMODEL_EPOCHS = 100  # companion paper used 1000
# MAX_BATCH_SIZE is the maximum batch size. This should be smaller than the
# number of data points in your data set. The upper limit of this parameter
# depends on how much memory your GPU/CPU has. 450 was suitable for a 32GB
# GPU.
MAX_BATCH_SIZE = 50  # companion paper used 450
# capacity_ls is a list of capacity values to try.
capacity_ls = list(range(2, 6))  # companion paper used list(range(2, 14))
weight_decay = 0
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

# Choose the device to train our models on.
if args.device == "cpu":
    device = "cpu"
elif args.device == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU

start = time.time()
# The companion paper trains multi-task (MT) models for six groups. In this
# example file, we will only train an MT model for electronic properties. If
# you are using this file as a template to train a new model, then change
# PROPERTY_GROUPS to suit your data set. The keys are names of property
# groups and the values are lists of properties that belong to the group.
# IMPORTANT: each element of each list should be included at least once
# in the "prop" column of your data set.
PROPERTY_GROUPS = {
    "electronic": [
        "Egc",
        "Egb",
        "Ea",
        "Ei",
    ],
}


# Load data. This data set is a subset of the data used to train the
# electronic-properties MT models shown in the companion paper. The full
# data set can be found at khazana.gatech.edu. If you are using this file
# as a template to train a new model, then load in your data set instead
# of "sample.csv".
master_data = pd.read_csv("./sample_data/sample.csv")

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

# Make a directory to save our models in. If you are using this file
# as a template to train a new model, then feel free to change this
# name to suit your use case.
parent_dir = "example_models/"
mkdir(parent_dir)

# Train one model per group. We only have one group, "electronic", in this
# example file.
for group in PROPERTY_GROUPS:
    # Get the columns of the properties to be modeled for this group.
    prop_cols = sorted(PROPERTY_GROUPS[group])
    print(
        f"Working on group {group}. The following properties will be modeled: {prop_cols}",
        flush=True,
    )

    # For single task models, `selector_dim` should equal 0. `selector_dim` refers to the
    # dimension of the selector vector. For multi-task data, `polygnn.prepare.prepare_train`
    # will, for each row in your data, create a selector vector. Each vector contains one
    # dimension per task. For single task, `polygnn.prepare.prepare_train` creates an empty
    # selector vector, so the dimension is equal to 0. But since this example file deals
    # with multi-task models, I use `selector_dim = len(prop_cols)`.
    selector_dim = len(prop_cols)

    # Define a directory to save the trained model for this group of properties.
    root_dir = parent_dir + group

    # Get the data for this group.
    group_train_data = train_data.loc[train_data.prop.isin(prop_cols), :]
    group_test_data = test_data.loc[test_data.prop.isin(prop_cols), :]

    # Define the augmented_featurizer function for this group based on the input arguments.
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
        # polygnn2 requires no augmentation.
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
        amp=False,
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
        ensemble_kwargs_dict={"monte_carlo": False},
    )
    pt.utils.mt_print_metrics(
        y, y_mean_hat, _selectors, scaler_dict, inverse_transform=False
    )
    print(f"Done working on group {group}\n", flush=True)

end = time.time()
print(f"Done with everything in {end-start} seconds.", flush=True)
