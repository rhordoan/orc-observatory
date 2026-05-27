# ORC Batch Experiments

Reproduces and extends thesis/paper benchmarks. Addresses PPSN reviewer requests:
structured instances (TSPLIB/QAPLIB), large-scale enumeration (GPU), fitness-shuffle ablation, Boltzmann-ORC ILS, OTG-based algorithm selection.

## Setup

```bash
cd orc-observatory
pip install -r requirements.txt -r experiments/requirements-experiments.txt
# Optional GPU:
pip install cupy-cuda12x numba
```

## Quick smoke test

```bash
python -m experiments.runner --config experiments/configs/exp_a_quick.yaml --output results/quick
python -m experiments.collect_results --results-dir results/quick
```

## Full experiment suite

| Config | Description |
|--------|-------------|
| `exp_a_reproduce.yaml` | NK/W-model N=16 (thesis tables) |
| `exp_b_large_scale.yaml` | N=20 enumeration (use GPU) |
| `exp_c_structured.yaml` | TSPLIB + QAPLIB |
| `exp_d_shuffle.yaml` | Fitness-shuffle ablation |
| `exp_e_boltzmann.yaml` | Boltzmann vs ORC+Pert ILS |

## Algorithm selector (Experiment F)

After A + E:

```bash
python -m experiments.algorithm_selector \
  --features results/exp-a/otg_features.csv \
  --ils results/exp-e/ils_comparison.csv \
  --output results/exp-f
```

## H100 cluster (Kubernetes)

```bash
export KUBECONFIG=/path/to/antoniu_iepure.yaml

# Dry-run (prints rendered manifests):
bash experiments/run_on_cluster.sh --dry-run

# Submit all experiments:
bash experiments/run_on_cluster.sh

# Monitor:
kubectl get jobs -n runai-romania-dev -l app=orc-experiments
kubectl logs -n runai-romania-dev -l experiment=exp-a --follow
```

Or build and apply manually:

```bash
docker build -f experiments/Dockerfile.gpu -t orc-experiments:latest .
export JOB_NAME=orc-exp-a IMAGE=orc-experiments:latest EXPERIMENT_NAME=exp-a CONFIG_FILE=exp_a_reproduce.yaml
envsubst < experiments/k8s/job-template.yaml | kubectl apply -n runai-romania-dev -f -
```

## Paper integration

Copy `results/*/escape_rates.csv` etc. into paper repo; use `collect_results.py` for summary tables.
