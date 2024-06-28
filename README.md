<!--intro-start-->

# Holistic Evaluation of Language Models

[comment]: <> (When using the img tag, which allows us to specify size, src has to be a URL.)
<img src="https://github.com/stanford-crfm/helm/raw/main/src/helm/benchmark/static/images/helm-logo.png" alt=""  width="800"/>

Welcome! The **`crfm-helm`** Python package contains code used in the **Holistic Evaluation of Language Models** project ([paper](https://arxiv.org/abs/2211.09110), [website](https://crfm.stanford.edu/helm/latest/)) by [Stanford CRFM](https://crfm.stanford.edu/). This package includes the following features:

- Collection of datasets in a standard format (e.g., NaturalQuestions)
- Collection of models accessible via a unified API (e.g., GPT-3, MT-NLG, OPT, BLOOM)
- Collection of metrics beyond accuracy (efficiency, bias, toxicity, etc.)
- Collection of perturbations for evaluating robustness and fairness (e.g., typos, dialect)
- Modular framework for constructing prompts from datasets
- Proxy server for managing accounts and providing unified interface to access models
<!--intro-end-->

To get started, refer to [the documentation on Read the Docs](https://crfm-helm.readthedocs.io/) for how to install and run the package.

## Directory Structure

The directory structure for this repo is as follows

```
├── docs # MD used to generate readthedocs
│
├── scripts # Python utility scripts for HELM
│ ├── cache
│ ├── data_overlap # Calculate train test overlap
│ │ ├── common
│ │ ├── scenarios
│ │ └── test
│ ├── efficiency
│ ├── fact_completion
│ ├── offline_eval
│ └── scale
└── src
├── helm # Benchmarking Scripts for HELM
│ │
│ ├── benchmark # Main Python code for running HELM
│ │ │
│ │ └── static # Current JS (Jquery) code for rendering front-end
│ │ │
│ │ └── ...
│ │
│ ├── common # Additional Python code for running HELM
│ │
│ └── proxy # Python code for external web requests
│
└── helm-frontend # New React Front-end
```

# Holistic Evaluation of Text-To-Image Models

<img src="https://github.com/stanford-crfm/helm/raw/heim/src/helm/benchmark/static/heim/images/heim-logo.png" alt=""  width="800"/>

Significant effort has recently been made in developing text-to-image generation models, which take textual prompts as 
input and generate images. As these models are widely used in real-world applications, there is an urgent need to 
comprehensively understand their capabilities and risks. However, existing evaluations primarily focus on image-text 
alignment and image quality. To address this limitation, we introduce a new benchmark, 
**Holistic Evaluation of Text-To-Image Models (HEIM)**.

We identify 12 different aspects that are important in real-world model deployment, including:

- image-text alignment
- image quality
- aesthetics
- originality
- reasoning
- knowledge
- bias
- toxicity
- fairness
- robustness
- multilinguality
- efficiency

By curating scenarios encompassing these aspects, we evaluate state-of-the-art text-to-image models using this benchmark. 
Unlike previous evaluations that focused on alignment and quality, HEIM significantly improves coverage by evaluating all 
models across all aspects. Our results reveal that no single model excels in all aspects, with different models 
demonstrating strengths in different aspects.

This repository contains the code used to produce the [results on the website](https://crfm.stanford.edu/heim/latest/) 
and [paper](https://arxiv.org/abs/2311.04287).

# Instructions on how to evaluate pulsar in fp8 on helm (thanks to Jue!)
Get our [AWS credentials](https://d-926759c7f1.awsapps.com/start/#/?tab=accounts) and your personal github and hugging face tokensGithub token and put them in your bashrc.
```
export HUGGING_FACE_HUB_TOKEN="<your hf token>"
export AWS_ACCESS_KEY_ID="<your aws key id>"
export AWS_SECRET_ACCESS_KEY="<your aws access key>"
export AWS_SESSION_TOKEN="<your aws session token>"
export GITHUB_TOKEN="<your gh token>"
export HF_TOKEN="<your hf token>"
```
source it
```
source ~/.bashrc
```
Build pulsar, first locate to the folder of the pulsar repo then
```
docker build -t pulsar:main -f Dockerfile   --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID   --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY   --build-arg AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN   --build-arg USE_CUSTOM_PYTORCH=true   --build-arg GITHUB_TOKEN=$GITHUB_TOKEN .
```
Move to the helm directory and edit env as necessary. You can keep HELM_IMG_NAME
Check if # instances or trials in run_benchmark_pulsar.sh look good to you, edit otherwise.

```
docker run \
    --gpus '"device=0"' \
    -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
    -v /scratch/$USER/data:/data \
    --entrypoint 'text-generation-server' \
    pulsar:main \
    quantize \
    meta-llama/Meta-Llama-3-8B-Instruct \
    /data/Meta-Llama-3-8B-Instruct-fp8 \
    --method fp8 \
    --no-dynamic-scaling
```

# Run the following command in the helm repo
```
docker compose --file pulsar_compose.yml --env-file env up
```
docker will bring up a pulsar container and helm container. The helm container runs run_benchmark_pulsar.sh 