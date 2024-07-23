# Use astromorph

There are several ways to interact with the Astromorph BYOL framework, and in this document we describe the advantages and downsides of each approach.
The easiest way is by using the pipeline, slightly more difficult is the ByolTrainer, and the most difficult is the BYOL class itself.

## Pipeline

### Training
The pipeline needs to be provided with a settings file, by using the following command (The filename still needs to be changed in the code.)
```bash
python astromorph/src/pipeline_training.py -c settings.toml
```

You have to provide it with the following information
- name of the file with a list of data filenames (`datafile = "filelist.txt"`)
- network name, choosing from `"n_layer_resnet"` and `"amm"` (this name still needs to be changed). 
  These are 2 pre-designed neural network architectures to choose from, with the amm being more lightweight.
- representation size, the size of the vector that comes out of the neural network, under the header `[byol_settings]`.

Other optional settings are:
- the number of epochs to train for
- which learning rate to use
- allow for exponential decay of the learning rate, and what decay parameter to use
- limit the number of cores used for the training process
- the projection size and projection hidden size
  - `projection_size` should be smaller than representation size, but not too small
  - `projection_hidden_size` should be larger than representation size
- `stacksize = 3` can be used to turn a single-channel image into a 3-channel RGB image, for pretrained computer vision networks
- if you are using an `n_layer_resnet`, you can specify how many layers of the Resnet18 you want to use with `last_layer = "layer1|2|3|4"`, under the heading `network_settings`.


At the end of training, the model is written to a model file, which is shown in the logs.

### Inference
Running inference on the data can be done with the following command:

```bash
python astromorph/src/inference.py -c settings.toml -n <name_of_model_file>.pt
```

This will load the model, load the data, and calculate the embeddings (representation vectors) of all the images in the dataset.
These embeddings are then saved into a file that can be read by TensorBoard.
Additionally, it is possible to store the embeddings in a CSV file by setting `export_to_csv = true` in the `settings.toml` file.

## `ByolTrainer`

There are several possible reasons why the pipeline does not provide enough configurabilitym and it may be necessary to switch to the `ByolTrainer` class.

These reasons may be:
- different computervision model
- custom augmentation function
- different learning rate scheduler
- different optimizer
- custom loss function 

These can all be provided when using the `ByolTrainer` class.

However, you will then have to take care of the following things:
- data matters
  - loading into Dataset/FilelistDataset
  - splitting into train and test set
  - converting into DataLoaders
- optimizing
  - set optimizer (if not the `Adam` default)
  - set learning rate scheduler (if applicable)
- create the neural network
- build the augmentation function

## `BYOL` class

There might be good reasons to use the `BYOL` class itself over the `ByolTrainer` class.
These reasons could be one of the following:
- use a different learning rate scheduler that does not need to be stepped after every epoch;
- do checkpointing of your model during training;
- perform custom logging;
- use custom evaluation metrics for logging into TensorBoard;
- use a custom training routine
- checkpointing and/or early termination

You will then have to construct your own training routine, instead of using the one provided.

