# MLflow Docker Infrastructure

This project contains a standalone MLflow tracking infrastructure using PostgreSQL and MinIO.

## Services
- **MLflow Tracking Server**: http://localhost:5001
- **PostgreSQL**: Backend store for experiment metadata.
- **MinIO Console**: http://localhost:9001 (S3-compatible artifact storage)

## Quick Start
1. Ensure Docker and Docker Compose are installed.
2. Run `docker-compose up -d`.
3. Verify accessibility at the links above.

## Connecting Clients
To connect a Python client to this server, use the following environment variables:

```bash
# In your client project's .env or shell:
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=mlflow
export AWS_SECRET_ACCESS_KEY=mlflow_password
```

And in your Python code:
```python
import mlflow
import os

# Optional: set via code if not in environment
# mlflow.set_tracking_uri("http://localhost:5001")

with mlflow.start_run():
    # Your tracking code here...
    pass
```

## Data Persistence
All data is stored in Docker volumes:
- `postgres_data`: PostgreSQL database files.
- `minio_data`: MinIO bucket data.
