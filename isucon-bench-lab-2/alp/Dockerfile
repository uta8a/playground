# syntax=docker/dockerfile:1
FROM alpine:3.20 AS download

ARG ALP_VERSION=1.0.22
ARG TARGETARCH

RUN apk add --no-cache curl tar \
 && case "$TARGETARCH" in amd64|arm64) ;; *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1;; esac \
 && archive="alp_linux_${TARGETARCH}.tar.gz" \
 && base_url="https://github.com/tkuchiki/alp/releases/download/v${ALP_VERSION}" \
 && curl --fail --location --silent --show-error --output "/tmp/${archive}" "${base_url}/${archive}" \
 && curl --fail --location --silent --show-error --output /tmp/checksums.txt "${base_url}/alp_${ALP_VERSION}_checksums.txt" \
 && (cd /tmp && grep " ${archive}$" checksums.txt | sha256sum -c -) \
 && tar -xzf "/tmp/${archive}" -C /tmp \
 && mkdir -p /out \
 && cp /tmp/alp /out/alp \
 && chmod 0555 /out/alp

FROM scratch

COPY --from=download /out/alp /alp

ENTRYPOINT ["/alp"]
