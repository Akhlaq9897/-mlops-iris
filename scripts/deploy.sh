#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-local}
IMAGE=${2:-}
PORT=${3:-8000}

if [[ -z "${IMAGE}" ]]; then
  echo "IMAGE argument is required"
  exit 1
fi

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -E '^iris-api$' >/dev/null 2>&1 || return 1
}

stop_and_remove() {
  if container_exists; then
    docker rm -f iris-api || true
  fi
}

run_local() {
  echo "Deploying locally: ${IMAGE} on port ${PORT}"
  stop_and_remove
  RUN_ARGS=( -d --name iris-api -p "${PORT}:8000" )
  if [[ -f models/random_forest_model.pkl ]]; then
    echo "Mounting local models/ into container"
    RUN_ARGS+=( -v "${PWD}/models:/app/models" -e MODEL_PATH=/app/models/random_forest_model.pkl )
  fi
  docker run "${RUN_ARGS[@]}" "${IMAGE}"
  docker ps --filter name=iris-api
}

run_ec2() {
  : "${EC2_HOST?EC2_HOST is required}"
  : "${EC2_USER?EC2_USER is required}"
  echo "Deploying to EC2 ${EC2_USER}@${EC2_HOST}: ${IMAGE} on port ${PORT}"

  if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
    mkdir -p ~/.ssh
    echo "${SSH_PRIVATE_KEY}" > ~/.ssh/id_rsa
    chmod 600 ~/.ssh/id_rsa
    SSH_OPTS="-i ~/.ssh/id_rsa -o StrictHostKeyChecking=no"
  else
    SSH_OPTS="-o StrictHostKeyChecking=no"
  fi

  ssh ${SSH_OPTS} "${EC2_USER}@${EC2_HOST}" "
    set -euo pipefail
    if docker ps -a --format '{{.Names}}' | grep -E '^iris-api$' >/dev/null; then
      docker rm -f iris-api || true
    fi
    docker pull ${IMAGE} || true
    docker run -d --name iris-api -p ${PORT}:8000 ${IMAGE}
  "
}

run_localstack() {
  echo "Deploying to LocalStack network (if available): ${IMAGE} on port ${PORT}"
  stop_and_remove
  NETWORK_ARG=()
  if docker network ls --format '{{.Name}}' | grep -E '^localstack$' >/dev/null 2>&1; then
    NETWORK_ARG=( --network localstack )
  fi
  RUN_ARGS=( -d --name iris-api -p "${PORT}:8000" )
  if [[ -f models/random_forest_model.pkl ]]; then
    RUN_ARGS+=( -v "${PWD}/models:/app/models" -e MODEL_PATH=/app/models/random_forest_model.pkl )
  fi
  docker run "${NETWORK_ARG[@]}" "${RUN_ARGS[@]}" "${IMAGE}"
  docker ps --filter name=iris-api
}

case "${MODE}" in
  local)
    run_local
    ;;
  ec2)
    run_ec2
    ;;
  localstack)
    run_localstack
    ;;
  *)
    echo "Unknown MODE: ${MODE}. Expected 'local', 'ec2', or 'localstack'"
    exit 1
    ;;
esac

