#! /bin/bash
unset AWS_SESSION_TOKEN
export MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
export AWS_ACCESS_KEY_ID=changeMe
export AWS_SECRET_ACCESS_KEY=changeMe
export AWS_DEFAULT_REGION=us-east-1
mlflow server --backend-store-uri postgresql://mlflow:mlflow@127.0.0.1/mlflow --default-artifact-root s3://argo-artifacts/mlflow-repo/ -h 127.0.0.1 -p 8000

