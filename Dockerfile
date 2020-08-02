# build the image required for setting up the repository
# Example run: 
# $ docker build -t adalmia/coreml:v1.0 .

# base image
FROM nvcr.io/nvidia/pytorch:20.01-py3

RUN apt-get update && apt-get install -y \
    aufs-tools \
    automake \
    build-essential \
    curl \
    dpkg-sig \
    libcap-dev \
    libsqlite3-dev \
    mercurial \
    virtualenv \
    reprepro \
    ruby1.9.1 && rm -rf /var/lib/apt/lists/*

# change working directory to /
WORKDIR /

# set the PYTHONPATH required for using the repository
ENV PYTHONPATH /workspace/coreml

# set actual working directory
WORKDIR /workspace/coreml

# set helpful aliases
RUN echo 'alias c="clear"' >> ~/.bashrc
RUN echo 'alias lab="jupyter lab"' >> ~/.bashrc
RUN echo 'alias wandb_off="export WANDB_MODE=dryrun"' >> ~/.bashrc
RUN echo 'alias wandb_on="export WANDB_MODE=run"' >> ~/.bashrc
RUN echo 'alias l="ls"' >> ~/.bashrc

# copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN cat requirements.txt | xargs -n 1 pip install | while read line; do echo $line; done;
