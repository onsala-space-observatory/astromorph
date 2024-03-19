# AstroMorph

The AstroMorph project is an ML project to automatically separate a collection of astronomical objects based on their morphology.

## Installation

The easiest way to set this project up is inside its own virtual environment.
If you are not familiar with those, you can read up on them [here](https://docs.python.org/3/library/venv.html).
In short, they are a very convenient way of separating projects with conflicting
requirements

```bash
# Run this command inside your working directory to create a virtual environment
> python -m venv .venv
# The virtual environment is created inside its own subdirectory
> ls -a
.
..
.venv
```

It is very easy to activate this environment, and deactivate it when you no longer need it.

```bash
# activate the virtual environment
> source .venv/bin/activate
# deactivate the venv when no longer using it, or switching to a different project
> deactivate
```

The next step is to install the dependencies in the virtual environment

```bash
# If you have deactivated your venv, make sure to activate it again
> source .venv/bin/activate
# Install requirements using pip
> pip install -r requirements
```

## Running

### Training

For now, the software expects data to be loaded in the following way:

- a single fits file containing all the data
- a fits file of the same size with a binary mask, where all the object pixels are coded with a `1`

To train the model, just run the following command from the main directory of the repo:

```bash
> python astromorph/src/basic_training.py -d <data-file> -m <mask-file>
```

Optionally, the number of training epochs can be specified with the `-e` flag, with a default of 10.

```bash
> python astromorph/src/basic_training.py -d <data-file> -m <mask-file> -e 5
```

### Inference

To run a trained network on some data, use the following command:

```bash
> python astromorph/src/inference.py -d <data-file> -m <mask-file> -n <trained-network-file>
```

To then view a visualisation of the embeddings, use TensorBoard with:

```bash
> tensorboard --logdir=./runs
```
