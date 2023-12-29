# This file is an example of how to get predictions for 36 different properties of polyethylene.
# The models used to make these predictions were trained in the polyGNN companion paper
# (DOI: 10.1021/acs.chemmater.2c02991). There are six total models, each one capable of
# predicting multiple properties (i.e., "tasks"). Let's go over each model.
#
# 1) The first model is titled "solubility_and_permeability". It can predict seven properties
# 1.1) "exp_solubility__MPa**0.5", the Hildebrand solubility parameter in units of MPa**0.5
# 1.2) "exp_perm_CH4__Barrer", the permeability of CH4 in units of log10(Barrer)
# 1.3) "exp_perm_CO2__Barrer", the permeability of CO2 in units of log10(Barrer)
# 1.4) "exp_perm_He__Barrer", the permeability of He in units of log10(Barrer)
# 1.5) "exp_perm_N2__Barrer", the permeability of N2 in units of log10(Barrer)
# 1.6) "exp_perm_O2__Barrer", the permeability of O2 in units of log10(Barrer)
# 1.7) "exp_perm_H2__Barrer", the permeability of H2 in units of log10(Barrer)
#
# 2) The second model is titled "thermal". It can predict four properties
# 2.1) "exp_Tm__K", the melting temperature in units of K
# 2.2) "exp_Tg__K", the glass-transition temperature in units of K
# 2.3) "exp_thermal_decomposition_temperature__K", the thermal decomposition temperature in units of K
# 2.4) "exp_thermal_conductivity__W_per_m_per_K", the thermal conductivity in units of watts per meter-kelvin
#
# 3) The third model is titled "thermodynamic_and_physical". It can predict seven properties
# 3.1) "exp_Cp__J_per_gK", the specific heat capacity in units of joules per gram-kelvin
# 3.2) "DFT_Eatomization__eV_per_atom", the atomization energy in units of eV/atom
# 3.3) "exp_limiting_oxygen_index__percentage", the limiting oxygen percentage
# 3.4) "exp_crystallinity_LF", the crystallization tendency, trained on low fidelity data
# 3.5) "exp_crystallinity_HF", the crystallization tendency, trained on high fidelity data
# 3.6) "exp_rho__g_per_cc", the density in units of g/cc
# 3.7) "exp_free_fractional_volume", the fractional free volume
#
# 4) The fourth model is titled "electronic". It can predict four properties
# 4.1) "DFT_bandgap_LF__eV", the band gap in units of eV. 'LF' stands for low
#      fidelity. Meaning the DFT calculations were done on polymer chain structures.
# 4.2) "DFT_bandgap_HF__eV", the band gap in units of eV. 'HF' stands for high
#      fidelity. Meaning the DFT calculations were done on polymer crystal structures.
# 4.3) "DFT_EA__eV", the electron affinity in units of eV
# 4.4) "DFT_Eionization__eV", the ionization energy in units of eV
#
# 5) The fifth model is titled "optical_and_dielectric". It can predict twelve properties
# 5.1) "exp_refractive_index", the refractive index
# 5.2) "DFT_refractive_index", the refractive index
# 5.3) "DFT_dielectric_total", the total dielectric constant
# 5.4) "exp_dielectric_constant_1.78", the dielectric constant at 10^(1.78) Hz
# 5.5) "exp_dielectric_constant_2.0", the dielectric constant at 10^(2.0) Hz
# 5.6) "exp_dielectric_constant_3.0", the dielectric constant at 10^(3.0) Hz
# 5.7) "exp_dielectric_constant_4.0", the dielectric constant at 10^(4.0) Hz
# 5.8) "exp_dielectric_constant_5.0", the dielectric constant at 10^(5.0) Hz
# 5.9) "exp_dielectric_constant_6.0", the dielectric constant at 10^(6.0) Hz
# 5.10) "exp_dielectric_constant_7.0", the dielectric constant at 10^(7.0) Hz
# 5.11) "exp_dielectric_constant_9.0", the dielectric constant at 10^(9.0) Hz
# 5.12) "exp_dielectric_constant_15.0", the dielectric constant at 10^(15.0) Hz
#
# 6) The sixth model is titled "mechanical". It can predict two properties
# 6.1) "exp_TS__MPa", the tensile stength in units of MPa
# 6.2) "exp_YM_GPa", the Young's modulus in units of GPa
#
# Note: Properties that have "exp_" ("DFT_") at the start mean that the corresponding
#       model was trained on experimental (DFT) data for that property.

import pandas as pd
import torch
import polygnn
import polygnn_trainer as pt

pe_smiles = "[*]CC[*]"  # the SMILES string for polyethylene

# For convenience, let's define a function that makes predictions.
def make_prediction(data, dir_name):
    """
    Return the mean and std. dev. of a model prediction.

    Args:
        data (pd.DataFrame): The input data for the prediction.
        dir_name (str): The name of the directory containing the model that
            you desire to get predictions from. (e.g., "thermal", "electronic", etc.)
    """
    root_dir = f"../trained_models/{dir_name}"
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


# Let's make predictions using the "solubility_and_permeability" model.
model_name = "solubility_and_permeability"
properties = [
    "exp_solubility__MPa**0.5",
    "exp_perm_CH4__Barrer",
    "exp_perm_CO2__Barrer",
    "exp_perm_He__Barrer",
    "exp_perm_N2__Barrer",
    "exp_perm_O2__Barrer",
    "exp_perm_H2__Barrer",
]

print(f"Using model {model_name}.")
data = pd.DataFrame(
    {"smiles_string": [pe_smiles] * len(properties), "prop": properties}
)
means, std_devs = make_prediction(data, model_name)
for prop, mean, std_dev in zip(properties, means, std_devs):
    print(f"The mean for {prop} is {mean}.")
    print(f"The std. dev. for {prop} is {std_dev}.")
print()

# Let's make predictions using the "thermal" model.
model_name = "thermal"
properties = [
    "exp_Tm__K",
    "exp_Tg__K",
    "exp_thermal_decomposition_temperature__K",
    "exp_thermal_conductivity__W_per_m_per_K",
]

print(f"Using model {model_name}.")
data = pd.DataFrame(
    {"smiles_string": [pe_smiles] * len(properties), "prop": properties}
)
means, std_devs = make_prediction(data, model_name)
for prop, mean, std_dev in zip(properties, means, std_devs):
    print(f"The mean for {prop} is {mean}.")
    print(f"The std. dev. for {prop} is {std_dev}.")
print()

# Let's make predictions using the "thermodynamic_and_physical" model.
model_name = "thermodynamic_and_physical"
properties = [
    "exp_Cp__J_per_gK",
    "DFT_Eatomization__eV_per_atom",
    "exp_limiting_oxygen_index__percentage",
    "exp_crystallinity_HF",
    "exp_crystallinity_LF",
    "exp_rho__g_per_cc",
    "exp_free_fractional_volume",
]

print(f"Using model {model_name}.")
data = pd.DataFrame(
    {"smiles_string": [pe_smiles] * len(properties), "prop": properties}
)
means, std_devs = make_prediction(data, model_name)
for prop, mean, std_dev in zip(properties, means, std_devs):
    print(f"The mean for {prop} is {mean}.")
    print(f"The std. dev. for {prop} is {std_dev}.")
print()

# Let's make predictions using the "electronic" model.
model_name = "electronic"
properties = [
    "DFT_bandgap_LF__eV",
    "DFT_bandgap_HF__eV",
    "DFT_EA__eV",
    "DFT_Eionization__eV",
]

print(f"Using model {model_name}.")
data = pd.DataFrame(
    {"smiles_string": [pe_smiles] * len(properties), "prop": properties}
)
means, std_devs = make_prediction(data, model_name)
for prop, mean, std_dev in zip(properties, means, std_devs):
    print(f"The mean for {prop} is {mean}.")
    print(f"The std. dev. for {prop} is {std_dev}.")
print()

# Let's make predictions using the "optical_and_dielectric" model.
model_name = "optical_and_dielectric"
properties = [
    "exp_refractive_index",
    "DFT_refractive_index",
    "DFT_dielectric_total",
    "exp_dielectric_constant_1.78",
    "exp_dielectric_constant_2.0",
    "exp_dielectric_constant_3.0",
    "exp_dielectric_constant_4.0",
    "exp_dielectric_constant_5.0",
    "exp_dielectric_constant_6.0",
    "exp_dielectric_constant_7.0",
    "exp_dielectric_constant_9.0",
    "exp_dielectric_constant_15.0",
]

print(f"Using model {model_name}.")
data = pd.DataFrame(
    {"smiles_string": [pe_smiles] * len(properties), "prop": properties}
)
means, std_devs = make_prediction(data, model_name)
for prop, mean, std_dev in zip(properties, means, std_devs):
    print(f"The mean for {prop} is {mean}.")
    print(f"The std. dev. for {prop} is {std_dev}.")
print()

# Let's make predictions using the "mechanical" model.
model_name = "mechanical"
properties = [
    "exp_TS__MPa",
    "exp_YM__GPa",
]

print(f"Using model {model_name}.")
data = pd.DataFrame(
    {"smiles_string": [pe_smiles] * len(properties), "prop": properties}
)
means, std_devs = make_prediction(data, model_name)
for prop, mean, std_dev in zip(properties, means, std_devs):
    print(f"The mean for {prop} is {mean}.")
    print(f"The std. dev. for {prop} is {std_dev}.")
print()
