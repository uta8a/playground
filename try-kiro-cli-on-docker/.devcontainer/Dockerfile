FROM debian:trixie-slim

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bash-completion \
    ca-certificates \
    curl \
    git \
    gpg \
    dirmngr \
    less \
    locales \
    procps \
    sudo \
    unzip \
    vim-tiny \
    wget \
    build-essential \
  && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
  && locale-gen \
  && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
  && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
  && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
  && chmod 0440 "/etc/sudoers.d/${USERNAME}"

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# mise
RUN curl https://mise.run | sh

# Kiro CLI
RUN curl -fsSL https://cli.kiro.dev/install | bash

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# bash起動時にmiseを有効化
RUN echo 'eval "$(mise activate bash)"' >> "/home/${USERNAME}/.bashrc"

WORKDIR /workspaces

CMD ["sleep", "infinity"]
