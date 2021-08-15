FROM rust:latest AS build

WORKDIR /app
COPY server/Cargo.toml server/Cargo.lock ./
COPY server/src/ src/
RUN cargo build --release
RUN find . -type d | grep "libtorch/lib$" | xargs -I{} mv {} libtorch

FROM debian:buster-slim

RUN apt-get update \
    && apt-get install -y \
        ca-certificates \
        libgomp1 \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/target/release/actix-torch-server /usr/local/bin/actix-torch-server
copy --from=build /app/libtorch/* /usr/lib/
COPY saved_model/ /app/saved_model/

RUN useradd -mU -s /bin/bash actix
USER actix

EXPOSE 8080
ENTRYPOINT ["actix-torch-server", "--model-path=/app/saved_model/model.pt"]
