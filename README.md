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
```

