# AstroMorph

The AstroMorph project is an ML project to automatically separate a collection of astronomical objects based on their morphology.

## Installation

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

## Running

### Basic configuration

The settings of the training run are specified through a TOML file.
An example of such a file can be found in `example_settings.toml`.
This file can be passed to the script with the `-c` or `--config-file` flag.
The script should be invoked from the main folder of the repository:

```bash
python astromorph/src/basic_training.py -c example_settings.toml
```

### Training

#### Filelist

Input can be specified as a filelist, which we specify with the `-d` flag.
Such a filelist can be made using the `find` command line program.
In the example below, we want to use all the FITS files inside the directory `data`
that are smaller than 10 MB as input for our model.
We do this using the following commands:

```bash
# Find the filenames and store them in data/inputfiles.txt
$ find data -type f -size -10M -path "**.fits" > data/inputfiles.txt
$ python astromorph/src/basic_training.py -c training_settings.toml
```

In this example, `training_settings.toml` would look similar to

```toml
# Configfile for using a filelist
datafile = "data/inputfiles.txt"
network_name = "n_layer_resnet"
```

#### Masked data

When extracting objects from a binary mask, the script expects the following input:

- a single FITS file containing all the data
- a FITS file of the same size with a binary mask, where all the object pixels are coded with a `1`

Now the config file would have the added keyword `maskfile`:

```toml
# Configfile for using masked data in two FITS files
datafile = "data/data_file.fits"
maskfile = "data/masked_file.fits"
network_name = "n_layer_resnet"
```

#### Epochs

Optionally, the number of training epochs can be specified with the `epochs` keyword, with a default of 10.

```toml
# Configfile for using a filelist
datafile = "data/inputfiles.txt"
epochs = 5
network_name = "n_layer_resnet"
```

#### Reduced ResNet18 network

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

### Inference

To run a trained network on some data, use the following command:

```bash
python astromorph/src/inference.py -d <data-file> -m <mask-file> -n <trained-network-file>
```

To then view a visualisation of the embeddings, use TensorBoard with:

```bash
tensorboard --logdir=./runs
```
