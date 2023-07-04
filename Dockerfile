FROM python:3.11

WORKDIR /tuto

# casadi libs can be found with `import casadi` and/or:
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/casadi PYTHONPATH=/tuto

RUN --mount=type=cache,sharing=locked,target=/root/.cache pip install \
    example-robot-data-jnrh2023 jupyterlab meshcat scipy ipywidgets matplotlib

CMD jupyter-lab --allow-root
