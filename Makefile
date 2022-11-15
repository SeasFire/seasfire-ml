.PHONY: all clean

all: venv
	$(VENV)/python -m ipykernel install --user --name=gnn

clean: clean-venv

include Makefile.venv

