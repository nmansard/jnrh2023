# JNRH'2023: Pinocchio Tutorial

Support chat: <https://matrix.to/#/#jnrh2023-tuto:laas.fr>

## Getting started

Start by cloning this repository:
```
git clone https://github.com/nmansard/jnrh2023
cd jnrh2023
```

After this, you have to **choose one of the three** following supported methods:

### pip

1. create an environment:

    `python -m venv .venv`

2. activate it:

    `source .venv/bin/activate`

3. update pip:

    `pip install -U pip`

4. install dependencies:

    `pip install example-robot-data-jnrh2023 jupyterlab meshcat scipy ipywidgets matplotlib`

5. start `PYTHONPATH=. jupyter-lab`


### conda

1. update conda:

    `conda update -n base -c defaults conda`

2. create an environment:

    `conda create -n jnrh-2023` (TODO: `python=3.x` ?)

3. activate it :

    `conda activate -n jnrh-2023`

4. install general dependencies:

    `conda install -c conda-forge jupyterlab meshcat-python scipy ipywidgets matplotlib`

5. install specific dependencies:

    `conda install -c olivier.roussel hpp-fcl example-robot-data pinocchio`

6. start `PYTHONPATH=. jupyter-lab`

### docker

1. start `docker run -v ./:/tuto --net=host -it nim65s/jnrh2023`

## Check your installation

Jupyter-lab should show you a couple links in `https://localhost:8888` / `https://127.0.0.1:8000`: you should be able
to open them, and in your webbrowser open and run `0_setup.ipynb` to say hi to Talos.
