
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel as torch_base

RUN apt update
RUN apt install -y zip unzip wget

RUN mkdir /workspace/helm
RUN chmod -R 777 /workspace/helm

WORKDIR /workspace/helm

COPY *.py .
COPY src src
COPY docs docs
COPY install-dev.sh .
COPY requirements.txt .
COPY setup.cfg .
COPY pyproject.toml .
COPY README.md .
COPY CHANGELOG.md .
COPY json-urls-root.js .
COPY json-urls.js .

# Install HELM
RUN bash install-dev.sh

# Additional packages
RUN pip install quanto

