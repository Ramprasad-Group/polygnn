import os
import pytest
import pandas as pd
from .utils_test import make_prediction

# The tests in this file should only be run if the "trained_models" directory is
# present. Some may wish to delete that folder due to the memory it consumes.

dir_name = "trained_models"  # directory name containing trained models, if present

# Boolean. True if directory exists.
trained_models_exist = os.path.exists(f"./{dir_name}")

# Reason why the test may have been skipped.
skip_reason = f"The {dir_name} directory was not detected."


@pytest.mark.skipif(not trained_models_exist, reason=skip_reason)
def test_tg_prediction__polyethylene():
    """
    Given the SMILES string of polyethylene,
    When making a prediction with the trained Tg model,
    Then the model mean and uncertainty should have particular values.
    """
    # Create data to predict on
    data = pd.DataFrame({"smiles_string": ["[*]CC[*]"], "prop": ["exp_Tg__K"]})

    model_dir_name = "thermal"  # name of the directory containing the thermal model

    mean, std_dev = make_prediction(data, model_dir_name)

    # Check that the mean and std. dev. values are correct to one decimal.
    assert str(round(mean[0], 1)) == "309.4"
    assert str(round(std_dev[0], 1)) == "55.8"


@pytest.mark.skipif(not trained_models_exist, reason=skip_reason)
def test_bandgap_prediction__polyethylene():
    """
    Given the SMILES string of polyethylene,
    When making a prediction with the trained band gap model,
    Then the model mean and uncertainty should have particular values.
    """
    # Create data to predict on
    data = pd.DataFrame({"smiles_string": ["[*]CC[*]"], "prop": ["DFT_bandgap_LF__eV"]})

    model_dir_name = "electronic"  # name of the directory containing the thermal model

    mean, std_dev = make_prediction(data, model_dir_name)

    # Check that the mean and std. dev. values are correct to one decimal.
    assert str(round(mean[0], 1)) == "6.7"
    assert str(round(std_dev[0], 1)) == "0.3"


@pytest.mark.skipif(not trained_models_exist, reason=skip_reason)
def test_dk_prediction__polyethylene():
    """
    Given the SMILES string of polyethylene,
    When making a prediction with the trained dielectric constant @ 100 Hz model,
    Then the model mean and uncertainty should have particular values.
    """
    # Create data to predict on
    data = pd.DataFrame(
        {"smiles_string": ["[*]CC[*]"], "prop": ["exp_dielectric_constant_2.0"]}
    )

    model_dir_name = (
        "optical_and_dielectric"  # name of the directory containing the thermal model
    )

    mean, std_dev = make_prediction(data, model_dir_name)

    # Check that the mean and std. dev. values are correct to one decimal.
    assert str(round(mean[0], 1)) == "3.2"
    assert str(round(std_dev[0], 1)) == "0.7"
