# polygnn

This repository contains the code and environment used to train the machine learning models presented in a companion paper, [*polyGNN: Multitask graph neural networks for polymer informatics*](https://arxiv.org/abs/2209.13557).

## Installation
This repository is currently set up to run on 1) Mac OSX and 2) Linux/Windows machines with CUDA 10.2. Please raise a GitHub issue if you want to use this repo with a different configuration. Otherwise, please follow these steps for installation:

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
## Citation
If you use this repository in your work please consider citing us.
```
@misc{https://doi.org/10.48550/arxiv.2209.13557,
  doi = {10.48550/ARXIV.2209.13557},
  
  url = {https://arxiv.org/abs/2209.13557},
  
  author = {Gurnani, Rishi and Kuenneth, Christopher and Toland, Aubrey and Ramprasad, Rampi},
  
  keywords = {Materials Science (cond-mat.mtrl-sci), FOS: Physical sciences, FOS: Physical sciences},
  
  title = {Polymer informatics at-scale with multitask graph neural networks},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
## License
This repository is protected under a General Public Use License Agreement, the details of which can be found in `GT Open Source General Use License.pdf`.
