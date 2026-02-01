FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o main .

# alpine:latest
FROM alpine@sha256:1882fa4569e0c591ea092d3766c4893e19b8901a8e649de7067188aba3cc0679
WORKDIR /root/
COPY --from=builder /app/main .
EXPOSE 8080
CMD ["./main"]
