# Jupyter layer on top of either toolchain image, for interactive
# experimentation with HAR graphs, calibration sets and HEF inspection.
#
# Build on top of the NVIDIA image:
#   docker build -f docker/jupyter.Dockerfile \
#       --build-arg BASE_IMAGE=dfc-nvidia:5.3.0 -t dfc-jupyter:nvidia docker/
#
# Build on top of the AMD image:
#   docker build -f docker/jupyter.Dockerfile \
#       --build-arg BASE_IMAGE=dfc-amd:5.3.0 -t dfc-jupyter:amd docker/

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir jupyter jupyterlab jupyter-resource-usage

# The DFC images already set NPY_PROMOTION_STATE=legacy; keep it explicit so
# notebooks cannot silently inherit a different value from the client env.
ENV NPY_PROMOTION_STATE=legacy

WORKDIR /workdir

EXPOSE 8888

# No token by default: these images are meant to run on an isolated
# workstation or behind an SSH tunnel. Pass -e JUPYTER_TOKEN=<secret> to
# require a token instead.
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --no-browser --allow-root ${JUPYTER_TOKEN:+--IdentityProvider.token=$JUPYTER_TOKEN}"]
