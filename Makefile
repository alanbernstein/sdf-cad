SHELL := /bin/bash

MM := ./bin/micromamba
ENV := sdf

.PHONY: setup env openvdb sdf run info clean

setup: env openvdb sdf sdf-patch-vdb

env:
	curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
	$(MM) create -y -n $(ENV) python=3.10 pip

openvdb:
	$(MM) install -y -n $(ENV) -c conda-forge openvdb pyopenvdb || \
	$(MM) install -y -n $(ENV) -c conda-forge openvdb

sdf:
	$(MM) run -n $(ENV) pip install git+https://github.com/fogleman/sdf.git

sdf-patch-vdb:
	./bin/micromamba run -n sdf bash -c '\
	  FILE=$$(python -c "import sdf, pathlib; print(pathlib.Path(sdf.__file__).parent / \"mesh.py\")"); \
	  sed -i.bak "s/import pyopenvdb as vdb/import openvdb as vdb/" "$$FILE"; \
	  echo "Patched $$FILE"; \
	'

run:
	$(MM) run -n $(ENV) python generate-stl-shell.py

# alternatively, in an interactive shell:
# eval "$(./bin/micromamba shell hook --shell=bash)"
# micromamba activate sdf
# python whatever.py

info:
	$(MM) info
	$(MM) env list

clean:
	rm -rf bin
