# AstroMorph

The AstroMorph project is an ML project to automatically separate a collection of astronomical objects based on their morphology. The method and science demonstration are detailed in Boschman et al. (in preparation). If you use AstroMorph in your research, please consider citing our paper:

**```astromorph```: self-supervised machine learning pipeline for astronomical morphology analysis**<br>
L. Boschman, O. Maya Lucas, P. Bjerkeli, J. Kainulainen, and M. C. Toribio
*(In preparation)*

## Installation

This project has been developed for Python 3.12.
Lower versions of Python might work, but there is no guarantee.
Anything below Python 3.9 will definitely not work.

The easiest way to set this project up is inside its own virtual environment.
If you are not familiar with those, you can read up on them [here](https://docs.python.org/3/library/venv.html).
In short, they are a very convenient way of separating projects with conflicting
requirements

```bash
# Run this command inside your working directory to create a virtual environment
$ python -m venv .venv
# The virtual environment is created inside its own subdirectory
$ ls -a
.
..
.venv
```

It is very easy to activate this environment, and deactivate it when you no longer need it.

```bash
# activate the virtual environment
$ source .venv/bin/activate
# deactivate the venv when no longer using it, or switching to a different project
$ deactivate
```

The next step is to install the dependencies in the virtual environment

```bash
# If you have deactivated your venv, make sure to activate it again
$ source .venv/bin/activate
# Install requirements using pip
$ pip install -r requirements.txt
```

## Package contents

In this package we provide the following functionalities:

- the `BYOL` class as a PyTorch implementation of the BYOL framework;
- the `ByolTrainer` class wraps around `BYOL` to provide an easy training interface;
- a FilelistDataset for easy handling of sets of FITS files;
- a light-weight 2D convolutional neural network called `AstroMorphologyModel`
- a configurable training script for basic command line use;
- a configurable inference script for easy inspection of the resulting embeddings.

The `BYOL` class provides the most flexibility, but requires some experience with setting up a training routine in PyTorch.
`ByolTrainer` and the training script provide more ease-of-use at the cost of some flexibility.

### BYOL Class

The `BYOL` class is a subclass of `pytorch.nn.Module`, and can therefore be used like any other PyTorch module.

**NB: remember to call the `update_moving_average()` method after every optimization step.**
If you do not know why you have to do this, please have a look at the original paper at <https://arxiv.org/abs/2006.07733>.

### ByolTrainer

The `ByolTrainer` class is a wrapper around `BYOL` providing an easy-to-use interface for those less experienced in neural networks.
Primarily, one only needs to specify the core network and the dimensionality of the resulting embeddings.
More customization is possible through providing an augmentation function, an optimizer, a learning rate, etc.
For training, one only needs two PyTorch `DataLoader` instances for the training- and test-set.

See below for a basic example using the `AstroMorphologyModel` network from this package:

```python
from torch.utils.data import DataLoader
from astromorph import AstroMorphologyModel, ByolTrainer

train_data = DataLoader(...)
test_data = DataLoader(...)

model = ByolTrainer(AstroMorphologyModel(), representation_size=128)

model.train_model(train_data=train_data, test_data=test_data, epochs=10)
```



### Training Script

#### Basic configuration

The settings of the training run are specified through a TOML file.
An example of such a file can be found in `example_settings.toml`.
This file can be passed to the script with the `-c` or `--config-file` flag.
The script should be invoked from the main folder of the repository:

```bash
python astromorph/src/pipeline_01_training.py -c example_settings.toml
```

#### Training

##### Filelist

Input can be specified as a filelist, which we specify with the `-d` flag.
Such a filelist can be made using the `find` command line program.
In the example below, we want to use all the FITS files inside the directory `data`
that are smaller than 10 MB as input for our model.
We do this using the following commands:

```bash
# Find the filenames and store them in data/inputfiles.txt
$ find /full/path/to/datadirectory/ -type f -size -10M -name "**.fits" > data/inputfiles.txt
$ python astromorph/src/pipeline_01_training.py -c training_settings.toml
```

In this example, `training_settings.toml` would look similar to

```toml
# Configfile for using a filelist
datafile = "data/inputfiles.txt"
network_name = "n_layer_resnet"
```

<!-- #### Masked data -->
<!---->
<!-- When extracting objects from a binary mask, the script expects the following input: -->
<!---->
<!-- - a single FITS file containing all the data -->
<!-- - a FITS file of the same size with a binary mask, where all the object pixels are coded with a `1` -->
<!---->
<!-- Now the config file would have the added keyword `maskfile`: -->
<!---->
<!-- ```toml -->
<!-- # Configfile for using masked data in two FITS files -->
<!-- datafile = "data/data_file.fits" -->
<!-- maskfile = "data/masked_file.fits" -->
<!-- network_name = "n_layer_resnet" -->
<!-- ``` -->

##### Epochs

Optionally, the number of training epochs can be specified with the `epochs` keyword, with a default of 10.

```toml
# Configfile for using a filelist
datafile = "data/inputfiles.txt"
epochs = 5
network_name = "n_layer_resnet"
```

##### Reduced ResNet18 network

It is possible to use only a few of the convolutional layers of the ResNet18 network.
There are four convolutional layers in ResNet18, named `layer1`, `layer2`,
`layer3`, and `layer4`.
If we select `layer2` as the last convolutional layer, `layer3` and `layer4`
will be removed from the network.

This might be beneficial, as the earlier layers are usually more generic.
To invoke this possibility, use the `last_layer` keyword inside the `network_settings` dictionary.
By default, `n_layer_resnet` will use the full ResNet18 network.

```toml
# Configfile for using a filelist
datafile = "data/inputfiles.txt"
epochs = 5
network_name = "n_layer_resnet"

[network_settings]
last_layer = "layer2"
```

##### Other settings

Other settings that can be set in a config file are the following:
```toml
# Limit the number of cores used in the training process
core_limit = 4

# If the network expects 3-channel RGB images, but you have single-channel images
[data_settings]
stacksize = 3

# Specify dimensions for the BYOL components
[byol_settings]
representation_size = 128
projection_size = 16
projection_hidden_size = 512
use_momentum = true # Target network is exponential MA of online network.
```

#### Inference

To run a trained network on some data, we will have to specify the location of the trained neural network.
We do this with the `trained_network_name` keyword in the config file.

```toml
# Configfile for using a filelist
datafile = "data/inputfiles.txt"
epochs = 5
network_name = "n_layer_resnet"
trained_network_name = "saved_models/newly_trained_network.pt"

export_to_csv = true # Export embeddings and metadata to a CSV file

[network_settings]
last_layer = "layer2"
```

The non-relevant options (e.g. `epochs`) will be ignored, so you can reuse the config file from the training run.

Alternatively, you can specify the relevant options using the command line, as shown here:

```bash
python astromorph/src/pipeline_02_inference.py -d <data-file> -m <mask-file> -n <trained-network-file>
```

It is even possible to use a combination of config file and command line options.
The options given in the command line will overrule the settings specified in the config file.

```bash
python astromorph/src/pipeline_02_inference.py -c example_settings.toml -n saved_models/newly_trained_network.pt
```

### Visualisation

To view a visualisation of the embeddings after inference, use TensorBoard with:

```bash
tensorboard --logdir=./runs
```
