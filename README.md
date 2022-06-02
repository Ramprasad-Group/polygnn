# polygnn

This repository contains the code and environment used to train the machine learning models presented in a companion paper, *polyGNN: Multitask graph neural networks for polymer informatics*.

## Installation

- Install [poetry](https://python-poetry.org/) on your machine.
- Clone this repo on your machine.
- Open a terminal at the root directory of this repository.
- Run `poetry install`.

## Dependencies
As can be seen in `pyproject.toml`, `polygnn` depends on several other packages, including [`polygnn_trainer`](https://github.com/rishigurnani/polygnn_trainer), 
[`polygnn_kit`](https://github.com/rishigurnani/polygnn_kit), and [`nndebugger`](https://github.com/rishigurnani/nndebugger). The functional relationships between these libraries are described briefly below and in `example.py`.

`polygnn` contains the polyGNN architecture developed in the companion paper. The architecture relies on [`polygnn_kit`](https://github.com/rishigurnani/polygnn_kit), which is a library for performing operations on polymer SMILES strings. Meanwhile, [`polygnn_trainer`](https://github.com/rishigurnani/polygnn_trainer) is a library for training neural network architectures, and was used in the companion paper to train the polyGNN architectures. Part of the training process utilized [`nndebugger`](https://github.com/rishigurnani/nndebugger), a library for debugging neural networks.

## Usage
The file `example.py` contains example code that illustrates how this package was used to the train models in the companion paper. The code uses training data located in the directory `sample_data` to train an ensemble model (composed of several submodels). The submodels, by default, are saved in a directory named `example_models`. The data in `sample_data` is a small subset of the DFT data used to train the models in the companion paper. A complete set of the DFT data can be found at [Khazana](https://khazana.gatech.edu/).

To train polygnn models run: `poetry run python example.py --polygnn`. To train polygnn2 models run: `poetry run python example.py --polygnn2`. Running either line on a machine with at least 8GB of free GPU memory should not take longer than 3 minutes.

## License
This repository is protected under a General Public Use License Agreement, the details of which can be found in `GT Open Source General Use License.pdf`.
