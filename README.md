Example of how to use the [sdf](https://github.com/fogleman/sdf) library to apply dilation to an arbitrary mesh, loaded from an STL file. The SDF readme has some instructions for this (building the OpenVDB dependency), but I'm a dummy and I'm bad at building C code, so I found a way to use OpenVDB without building (on linux).

# Setup
`make setup`
- installs [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
- installs [openvdb](https://www.openvdb.org/documentation/doxygen/python.html) and sdf to the micromamba environment
- patches one import in `sdf`

# Run
Either `make run` to see if it works, or:

```shell
eval "$(./bin/micromamba shell hook --shell=bash)"
micromamba activate sdf
```

Then `python generate-stl-shell.py`, or create and run other python files, without needing to add new Makefile targets

# Example