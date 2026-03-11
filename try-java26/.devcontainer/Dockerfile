FROM debian:trixie-slim

ARG CORRETTO_DEB_URL="https://corretto.aws/downloads/latest/amazon-corretto-26-x64-linux-jdk.deb"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV JAVA_HOME=/usr/lib/jvm/java-26-amazon-corretto
ENV SDKMAN_DIR=/home/${USERNAME}/.sdkman
ENV PATH="/home/${USERNAME}/.local/bin:${JAVA_HOME}/bin:${PATH}"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    openssh-client \
    procps \
    unzip \
    zip \
    sudo \
    locales; \
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen; \
    locale-gen; \
    curl -fsSL "$CORRETTO_DEB_URL" -o /tmp/corretto.deb; \
    apt-get install -y --no-install-recommends /tmp/corretto.deb; \
    rm -f /tmp/corretto.deb; \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"; \
    rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
WORKDIR /workspaces

RUN set -eux; \
    curl -fsSL "https://get.sdkman.io" | bash; \
    echo 'export SDKMAN_DIR="$HOME/.sdkman"' >> "$HOME/.bashrc"; \
    echo '[[ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]] && source "$SDKMAN_DIR/bin/sdkman-init.sh"' >> "$HOME/.bashrc"; \
    bash -lc 'source "$HOME/.sdkman/bin/sdkman-init.sh" && sdk version'; \
    curl -fsSL "https://mise.run" | sh; \
    echo 'eval "$(mise activate bash)"' >> "$HOME/.bashrc"; \
    bash -lc 'mise --version'

CMD ["sleep", "infinity"]
