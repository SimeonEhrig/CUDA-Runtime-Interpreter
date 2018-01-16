FROM       ubuntu:bionic

# general environment for docker
ENV        DEBIAN_FRONTEND=noninteractive \
           FORCE_UNSAFE_CONFIGURE=1

RUN        apt-get update \
           && apt-get install -y --no-install-recommends \
              build-essential \
              ca-certificates \
              curl \
              clang \
              cmake \
              git \
              pkg-config \
              libc6-dev \
              libclang-dev \
              libedit-dev \
              llvm-dev \
              lsb-release \
              nvidia-cuda-toolkit \
              wget \
              zlib1g-dev \
           && rm -rf /var/lib/apt/lists/*

COPY       . /root/cri

RUN        mkdir -p /root/build \
           && cd /root/build \
           && cmake ../cri \
           && make -j2 \
           && make install \
           && rm -rf ../build/* \
           && rm /root/cri/example_prog/*.sh

# select supported host compiler for nvcc
RUN        /bin/echo "alias nvcc='nvcc -ccbin clang++-3.8'" >> /etc/profile.d/nvcc.sh \
           && sed -i 's/nvcc /nvcc -ccbin clang++-3.8 /g' /usr/local/bin/generate_nvcc_fatbin.sh

CMD        cd /root/cri/example_prog \
           && /bin/bash -l
