# -mlops-iris
MLOps pipeline using Iris Data Set

## CI/CD

GitHub Actions pipeline runs on every push/PR:
- Lints (ruff) and tests (pytest)
- Builds and pushes a Docker image to Docker Hub when credentials are provided
- Optional deploy to local or EC2 on `main` using `scripts/deploy.sh`

### Secrets required
- `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` to push images
- `DEPLOY_TARGET` = `local` or `ec2`
- For EC2: `EC2_HOST`, `EC2_USER`, `SSH_PRIVATE_KEY`
- Optional: `APP_PORT` (default 8000)

### Local dev

Install dev deps from `pyproject.toml`:

```bash
pip install -U pip
pip install .[dev]
pytest -q
```

### Usage: MLflow tracking

Train models and track experiments locally with MLflow.

1) Install training extras (includes mlflow):

```bash
pip install -U pip
pip install .[train]
```

2) Prepare the dataset:

```bash
python notebooks/load_preprocess.py
```

3) Run training (logs params/metrics/models to `./mlruns`):

```bash
python notebooks/train_models.py
```

4) Launch the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns -p 5000
```

Open http://127.0.0.1:5000 to explore runs, metrics, params, and artifacts.

Optional: serve a model artifact from a specific run (pick a RUN_ID from the UI):

```bash
mlflow models serve -m runs:/<RUN_ID>/random_forest_model -p 1234
```

### Deploy script

Examples:

```bash
bash scripts/deploy.sh local your-dockerhub-username/iris-api:sha-<commit> 8000

# Or to EC2 (requires env vars):
DEPLOY_TARGET=ec2 EC2_HOST=1.2.3.4 EC2_USER=ubuntu \
  SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" \
  bash scripts/deploy.sh ec2 your-dockerhub-username/iris-api:sha-<commit> 8000
```
