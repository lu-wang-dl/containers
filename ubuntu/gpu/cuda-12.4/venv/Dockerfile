
# See https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.0.3/ubuntu18.04-x86_64/runtime/cudnn8/Dockerfile
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ubuntu:22.04

RUN apt-get update \
  && apt-get -y upgrade \
  && apt-get install --yes \
    openjdk-8-jdk \
    iproute2 \
    bash \
    sudo \
    coreutils \
    procps \
    acl \
    gnupg \
    gnupg2 \
    software-properties-common \
    wget \
  && /var/lib/dpkg/info/ca-certificates-java.postinst configure \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG NCCL_VERSION=2.22.3

# Add new user for cluster library installation
RUN useradd libraries \
  && usermod -L libraries

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin \
    && mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    && apt-get update \
    && apt-get install --allow-downgrades --no-install-recommends -y \
      # https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist/12.5.1/ubuntu2404/runtime/Dockerfile
      cuda-libraries-12-6=12.6.0-1 \
      libnpp-12-6=12.3.1.23-1 \
      cuda-nvtx-12-6=12.6.37-1 \
      libcublas-12-6=12.6.0.22-1 \
      libcusparse-12-6=12.5.2.23-1 \
      libnccl2=$NCCL_VERSION-1+cuda12.6 \
      # https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.8.0/ubuntu2204/devel/Dockerfile
      cuda-cudart-dev-12-6=12.6.37-1 \
      cuda-command-line-tools-12-6=12.6.0-1 \
      cuda-minimal-build-12-6=12.6.0-1 \
      cuda-libraries-dev-12-6=12.6.0-1 \
      cuda-nvml-dev-12-6=12.6.37-1 \
      cuda-nvprof-12-6=12.6.37-1 \
      libnpp-dev-12-6=12.3.1.23-1 \
      libnccl-dev=$NCCL_VERSION-1+cuda12.6 \
      # Popular models such as MPT and Databricks Dolly require several CUDA dev libraries
      # Find the versions here: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
      libcublas-dev-12-6=12.6.0.22-1 \
      libcusparse-dev-12-6=12.5.2.23-1 \
      libcusolver-dev-12-6=11.6.4.38-1 \
      libcurand-dev-12-6=10.3.7.37-1 \
      # CUPTI for TensorFlow: https://www.tensorflow.org/install/gpu
      cuda-cupti-12-6=12.6.37-1 \
      cuda-cupti-dev-12-6=12.6.37-1 \
      # TensorFlow requires libcusolver
      libcusolver-12-6=11.6.4.38-1 \
      # ibverbs-providers installs user-space drivers for RDMA support
      ibverbs-providers=39.0-1 \
      # Install cudnn
      # Reference: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.8.0/ubuntu2204/runtime/cudnn8/Dockerfile
      libcudnn9-cuda-12=9.3.0.75-1 \
      libcudnn9-dev-cuda-12=9.3.0.75-1 \
      # The dcgm-exporter binary has a runtime dependency on the DCGM library. The version we install
      # must be compatible with the DCGM-Exporter version. The expected DCGM version is the first version
      # number in the name of the DCGM-Exporter release (see
      # https://github.com/NVIDIA/dcgm-exporter/blob/main/Makefile);
      datacenter-gpu-manager=1:3.3.7 \
      # Tensorflow is built with TensorRT 10.0 (https://github.com/tensorflow/tensorflow/blob/r2.17/tensorflow/tools/toolchains/remote_config/configs.bzl#L602)
      # There is no libnvinfer10 binary with cuda 12.6 but only 12.5.
      libnvinfer10=10.2.0.19-1+cuda12.5 \
      libnvinfer-plugin10=10.2.0.19-1+cuda12.5 \
    # We remove all GPU repos to avoid CUDA 12.6 incompatible upgrades.
    && add-apt-repository -r "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    && add-apt-repository -r "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"

# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubuntu2204/base/Dockerfile
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526"

# temporarily using CUDA stubs at build time based on the suggestions from https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-link-against-driver-apis-at-build-time-eg-libcudaso-or-libnvidia-mlso
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubuntu2204/base/Dockerfile
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526"

# temporarily using CUDA stubs at build time based on the suggestions from https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-link-against-driver-apis-at-build-time-eg-libcudaso-or-libnvidia-mlso
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs

ARG python_version="3.10"
ARG pip_version="22.3.1"
ARG setuptools_version="65.6.3"
ARG wheel_version="0.38.4"
ARG virtualenv_version="20.16.7"

# Installs python 3.10 and virtualenv for Spark and Notebooks
RUN apt-get update \
  && apt-get install -y curl software-properties-common python${python_version} python${python_version}-dev python${python_version}-distutils \
  && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
  && /usr/bin/python${python_version} get-pip.py pip==${pip_version} setuptools==${setuptools_version} wheel==${wheel_version} \
  && rm get-pip.py

RUN /usr/local/bin/pip${python_version} install --no-cache-dir virtualenv==${virtualenv_version} \
  && sed -i -r 's/^(PERIODIC_UPDATE_ON_BY_DEFAULT) = True$/\1 = False/' /usr/local/lib/python${python_version}/dist-packages/virtualenv/seed/embed/base_embed.py \
  && /usr/local/bin/pip${python_version} download pip==${pip_version} --dest \
  /usr/local/lib/python${python_version}/dist-packages/virtualenv_support/

# Initialize the default environment that Spark and notebooks will use
RUN virtualenv --python=python${python_version} --system-site-packages /databricks/python3 --no-download --no-setuptools

# These python libraries are used by Databricks notebooks and the Python REPL
# You do not need to install pyspark - it is injected when the cluster is launched
# Versions are intended to reflect latest DBR: https://docs.databricks.com/release-notes/runtime/14.3.html#system-environment

COPY requirements.txt /databricks/.

RUN /databricks/python3/bin/pip install --no-deps -r /databricks/requirements.txt

# Specifies where Spark will look for the python process
ENV PYSPARK_PYTHON=/databricks/python3/bin/python3

RUN virtualenv --python=python${python_version} --system-site-packages /databricks/python-lsp --no-download --no-setuptools

COPY python-lsp-requirements.txt /databricks/.

RUN /databricks/python-lsp/bin/pip install -r /databricks/python-lsp-requirements.txt

RUN apt-get install wget && wget https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    && mv ttyd.x86_64 ttyd && chmod +x ./ttyd \
    && ./ttyd -p 7681 bash >& /tmp/ttyd.log &

# install the pytorch versions
RUN /databricks/python3/bin/pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    && /databricks/python3/bin/pip cache purge
