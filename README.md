# polygnn

# IMPORTANT NOTE: The code and data shared here is available for academic non-commercial use only

This repository contains the training code and model weights presented in a companion paper, [*polyGNN: Multitask graph neural networks for polymer informatics*](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c02991).

This repository was also used to train a model in a subsequent paper, [*Gas permeability, diffusivity, and solubility in polymers*](https://www.nature.com/articles/s41524-024-01373-9). Those model weights are provided in this repository, while the training data is in a different [repository](https://github.com/Ramprasad-Group/polyVERSE/tree/main/Other/Gas_permeability_solubility_diffusivity).

## Installation
This repository is currently set up to run on 1) Mac OSX and 2) Linux/Windows machines with CUDA 10.2. Please raise a GitHub issue if you want to use this repo with a different configuration. Otherwise, follow these steps for installation:

1. Install [poetry](https://python-poetry.org/) on your machine.
2. If Python3.7 is installed on your machine skip to step 3, if not you will need to install it. There are many ways to do this, one option is detailed below:
    * Install [Homebrew](https://brew.sh/) on your machine.
    * Run `brew install python@3.7`. Take note of the path to the python executable.
3. Clone this repo on your machine.
4. Open a terminal at the root directory of this repository.
5. Run `poetry env use /path/to/python3.7/executable`. If you installed Python3.7 with Homebrew, the path may be something like
  `/usr/local/Cellar/python\@3.7/3.7.13_1/bin/python3.7`.
7. Run `poetry install`.
8. If your machine is a Mac, run `poetry run poe torch-osx`. If not, run `poetry run poe torch-linux_win-cuda102`.
9. If your machine is a Mac, run `poetry run poe pyg-osx`. If not, run `poetry run poe pyg-linux_win-cuda102`.

## Dependencies
As can be seen in `pyproject.toml`, `polygnn` depends on several other packages, including [`polygnn_trainer`](https://github.com/rishigurnani/polygnn_trainer), 
[`polygnn_kit`](https://github.com/rishigurnani/polygnn_kit), and [`nndebugger`](https://github.com/rishigurnani/nndebugger). The functional relationships between these libraries are described briefly below and in `example.py`.

`polygnn` contains the polyGNN architecture developed in the companion paper. The architecture relies on [`polygnn_kit`](https://github.com/rishigurnani/polygnn_kit), which is a library for performing operations on polymer SMILES strings. Meanwhile, [`polygnn_trainer`](https://github.com/rishigurnani/polygnn_trainer) is a library for training neural network architectures, and was used in the companion paper to train the polyGNN architectures. Part of the training process utilized [`nndebugger`](https://github.com/rishigurnani/nndebugger), a library for debugging neural networks.

## Usage
### `example.py`
The file `example.py` contains example code that illustrates how this package was used to the train models in the companion paper. The code uses training data located in the directory `sample_data` to train an ensemble model (composed of several submodels). The submodels, by default, are saved in a directory named `example_models`. The data in `sample_data` is a small subset of the DFT data used to train the models in the companion paper. A complete set of the DFT data can be found at [Khazana](https://khazana.gatech.edu/).

To train polygnn models run: `poetry run python example.py --polygnn`. To train polygnn2 models run: `poetry run python example.py --polygnn2`. Running either line on a machine with at least 8GB of free GPU memory should not take longer than 3 minutes. To manually specify the device you want to use for training, set the device flag. For example `poetry run python example.py --polygnn --device cpu`. Otherwise, the device will automatically be chosen.

Looking at `sample_data/sample.csv`, you will notice that this dataset contains multiple different properties (e.g., band gap, electron affinity, etc.). In `example.py`, we use this data to train a multitask model, capable of predicting each property. To train your own multitask model, you can replace `sample_data/sample.csv` with your own dataset containing multiple properties. Single task models are also supported.

### `example2.py`
`example.py` is an example of how to train a multitask model with only SMILES strings as features. `example2.py` is an example of how to train a multitask model containing both SMILES and *non-SMILES* features. `example.py` and `example2.py` share the same flags. Read the comments in `example2.py` for more details.

### `more_examples`
A directory containing more example files.
- `more_examples/example_predict_trained_models.py` an example of how to just do prediction using one of the models trained in the [companion paper](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c02991). To run the file do `cd more_examples && poetry run python example_predict_trained_models.py`. The file shows how to get predictions for 36 different properties of polyethylene. Of course, you can change the polymer that you want to get predictions for, but the properties that can be predicted are fixed. If you want to make a prediction for a different property, then you'll need to train your own model (see `example.py` and `example2.py` for more details on training your own model).
- `more_examples/example_predict.py` an example of how to just do prediction using a previously-trained model not included in the [companion paper](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c02991). This file requires that an unmodified `example.py` be run first. This file shares the same flags as `example.py`.
- `more_examples/example_predict_phanEtAl.py` an example of how to do predictions using the model trained in Phan et. al's paper, [*Gas permeability, diffusivity, and solubility in polymers*](https://www.nature.com/articles/s41524-024-01373-9).

## Questions
I ([@rishigurnani](https://github.com/rishigurnani)) am more than happy to answer any questions about this codebase. If you encounter any troubles, please open a new Issue in the "Issues" tab and I will promptly respond. In addition, if you discover any bugs or have any suggestions to improve the codebase (documentation, features, etc.) please also open a new Issue. This is the power of open source!

## Citation
If you use this repository in your work please consider citing the origial polyGNN paper.
```
@article{Gurnani2023,
   annote = {doi: 10.1021/acs.chemmater.2c02991},
   author = {Gurnani, Rishi and Kuenneth, Christopher and Toland, Aubrey and Ramprasad, Rampi},
   doi = {10.1021/acs.chemmater.2c02991},
   issn = {0897-4756},
   journal = {Chemistry of Materials},
   month = {feb},
   number = {4},
   pages = {1560--1567},
   publisher = {American Chemical Society},
   title = {{Polymer Informatics at Scale with Multitask Graph Neural Networks}},
   url = {https://doi.org/10.1021/acs.chemmater.2c02991},
   volume = {35},
   year = {2023}
}
```

And, if relevant, the subsequent papers which have used polyGNN
- [*Gas permeability, diffusivity, and solubility in polymers*](https://www.nature.com/articles/s41524-024-01373-9)
- [*AI-assisted discovery of high-temperature dielectrics for energy storage*](https://www.nature.com/articles/s41467-024-50413-x)

## Companion paper
The results shown in the [companion paper](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c02991) were generated using v0.2.0 of this package.

## License
This repository is protected under a General Public Use License Agreement, the details of which can be found in `GT Open Source General Use License.pdf`.
