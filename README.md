# JNRH'2023: Pinocchio Tutorial

Support chat: <https://matrix.to/#/#jnrh2023-tuto:laas.fr>

Slides: <https://gepettoweb.laas.fr/talks/jnrh2023/>

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


### conda (Linux, macOS and Windows)

1. update conda:

    `conda update -n base -c defaults conda`

2. create an environment:

    `conda create -n jnrh-2023 python=3.9`

3. activate it :

    `conda activate jnrh-2023`

4. add conda-forge channel:

    `conda config --add channels conda-forge`
   
    Note: If you already have the `conda-forge` channel set-up in your config, this will have no effect and you will likely get the warning message `Warning: 'conda-forge' already in 'channels' list, moving to the top`. You can simply discard it.

6. install specific dependencies:

    `conda install -c olivier.roussel hpp-fcl example-robot-data pinocchio=2.99.0`

    Note: Forcing the Pinocchio package version to `2.99.0` is not mandatory depending on your channel priority configuration, which can vary depending on your conda installation source.

8. check your Pinocchio install:

    `conda list`

   Check for correct packages versions and source channel (4th column) in your output:
   ```
   example-robot-data    4.1.0     <some_hash>    olivier.roussel
   hpp-fcl               2.99.0    <some_hash>    olivier.roussel 
   pinocchio             2.99.0    <some_hash>    olivier.roussel
   ```
   If versions or source channel does not match this, then something gone wrong in your installation process.

9. install regular dependencies:

    `conda install -c conda-forge jupyterlab meshcat-python scipy ipywidgets matplotlib`

10. start JupyterLab:
   
    Be sure that you are still in the `jnrh2023` root directory of the repository that you cloned at the *Getting started* step.
    #### Linux / macOS
    `PYTHONPATH=. jupyter-lab`

    #### Windows
    `jupyter-lab`

### docker

1. start `docker run -v ./:/tuto --net=host -it nim65s/jnrh2023`

## Check your installation

Jupyter-lab should show you a couple links in `https://localhost:8888` / `https://127.0.0.1:8000`: you should be able
to open them, and in your webbrowser open and run `0_setup.ipynb` to say hi to Talos.
