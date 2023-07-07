# seasfire-ml

## Installation 

Run `make bash` to get a new environment with basic dependencies installed. 
The install torch geometric by hand. You need to follow the pip instructions 
in https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html in 
order to get torch version and cuda version. 

Then run something like:

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install torch-geometric
pip install torch-geometric-temporal
```

## Dataset creation 

To create a dataset you need to call `create_dataset.py` three times. 

```
./create_dataset.py --split=test
./create_dataset.py --split=train --positive-samples-size=50000 --negative-samples-size=50000
./create_dataset.py --split=val --positive-samples-size=2500 --negative-samples-size=5000
```

This assumes that the 100km cube is located at `../seasfire_1deg.zarr`. It samples only from Africa. This 
can be adjusted from the `--area` option. By default the dataset will support a timeseries of at most 24 weeks
as well as a prediction horizon of at most 24 weeks.

## Train a GRU model 

The train dataset from the example above contains 100000 samples. Depending on the target week these samples 
might be positive or negative and highly unbalanced. We use a weighted random sampler to create balanced batches, using 
prestored information in the dataset. 

During training the user can set `--batches-per-epoch` in order to sample a reduced number of batches per epoch. 
If not provided the number of batches will depend on the batch size and the total number of samples. Note however, 
that due to sampling it might not contain all samples or some samples (the minority class) will be sampled multiple 
times.

```
./train_gru.py --target-week=4 --batch-size=16 --timesteps=6 --no-include-oci-variables  --batches-per-epoch=2000
```

Variable `timesteps` contains the length of the timeseries. Variable `include-oci-variables` will result in using 
the oci variables (10 in total) which also contain global information.

## Testing a GRU model 

After training we get two files called `runs/{model-name}.best_model.pt` and `runs/{model-name}.last_model.pt`. You 
can test using 

```
./test_gru.py --target-week=4 --batch-size=16 --no-include-oci-variables --model-path=runs/{model-name}.best_model.pt
```

