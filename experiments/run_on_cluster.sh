#!/usr/bin/env bash
# Build GPU image and submit experiment Jobs to the K8s cluster.
# Usage: ./run_on_cluster.sh [--registry REGISTRY] [--tag TAG] [--dry-run]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export KUBECONFIG="${KUBECONFIG:-$ROOT/../antoniu_iepure.yaml}"
NAMESPACE="${NAMESPACE:-runai-romania-dev}"
REGISTRY="${REGISTRY:-}"
TAG="${TAG:-latest}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --registry) REGISTRY="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=true; shift ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -n "$REGISTRY" ]]; then
  IMAGE="${REGISTRY}/orc-experiments:${TAG}"
else
  IMAGE="orc-experiments:${TAG}"
fi

echo "=== Building Docker image: $IMAGE ==="
docker build -f "$SCRIPT_DIR/Dockerfile.gpu" -t "$IMAGE" "$ROOT"

if [[ -n "$REGISTRY" ]]; then
  echo "=== Pushing image to $REGISTRY ==="
  docker push "$IMAGE"
fi

echo "=== Creating PVC (if not exists) ==="
kubectl get pvc orc-experiment-results -n "$NAMESPACE" >/dev/null 2>&1 \
  || kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/k8s/pvc.yaml"

declare -A CONFIGS=(
  ["exp-a"]="exp_a_reproduce.yaml"
  ["exp-b"]="exp_b_large_scale.yaml"
  ["exp-c"]="exp_c_structured.yaml"
  ["exp-d"]="exp_d_shuffle.yaml"
  ["exp-e"]="exp_e_boltzmann.yaml"
)

for name in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$name]}"
  job_name="orc-${name}-$(date +%s)"
  echo ""
  echo "=== Submitting $name ($cfg) as job $job_name ==="

  export JOB_NAME="$job_name"
  export EXPERIMENT_NAME="$name"
  export CONFIG_FILE="$cfg"
  export IMAGE

  if $DRY_RUN; then
    envsubst < "$SCRIPT_DIR/k8s/job-template.yaml"
    echo "---"
  else
    envsubst < "$SCRIPT_DIR/k8s/job-template.yaml" \
      | kubectl apply -n "$NAMESPACE" -f -
  fi
done

echo ""
echo "=== All jobs submitted. Monitor with: ==="
echo "  kubectl get jobs -n $NAMESPACE -l app=orc-experiments"
echo "  kubectl logs -n $NAMESPACE -l app=orc-experiments --follow"
