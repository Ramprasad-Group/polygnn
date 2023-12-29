import torch
import pandas as pd
import argparse
import numpy as np
import random

import polygnn
import polygnn_trainer as pt


def make_prediction(data, dir_name):
    """
    Return the mean and std. dev. of a model prediction.

    Args:
        data (pd.DataFrame): The input data for the prediction.
        dir_name (str): The name of the directory containing the model that
            you desire to get predictions from. (e.g., "thermal", "electronic", etc.)
    """
    root_dir = f"./trained_models/{dir_name}"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU

    # Load scalers
    scaler_dict = pt.load2.load_scalers(root_dir)

    # Load selectors
    selectors = pt.load2.load_selectors(root_dir)

    # Load and evaluate ensemble.
    ensemble = pt.load.load_ensemble(
        root_dir,
        polygnn.models.polyGNN,
        device,
        {
            "node_size": atom_config.n_features,
            "edge_size": bond_config.n_features,
            "selector_dim": len(selectors),
        },
    )

    # Define a lambda function for smiles featurization.
    smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
        x,
        bond_config,
        atom_config,
        "monocycle",
    )

    # Perform inference
    y, y_mean_hat, y_std_hat, _selectors = pt.infer.eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=data,
        smiles_featurizer=smiles_featurizer,
        device=device,
        ensemble_kwargs_dict={"monte_carlo": False},
    )
    return y_mean_hat, y_std_hat
