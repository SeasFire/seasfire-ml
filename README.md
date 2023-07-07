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
