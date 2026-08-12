# CI/CD Pipeline Setup Guide

This document outlines the steps required to get the CI/CD pipeline running on GitHub Actions.

## Files Created for CI/CD

The following files have been created to enable the CI/CD pipeline:

### 1. **tests/test_model.py**
   - Unit tests for model prediction functionality
   - Tests model artifacts, prediction shape, prediction range
   - Validates that the model can make predictions on text input

### 2. **tests/test_flask_app.py**
   - Unit tests for the Flask application
   - Tests home page accessibility, form rendering, predict endpoint
   - Tests metrics endpoint and app configuration

### 3. **scripts/promote_model.py**
   - Model promotion script that validates model quality
   - Checks model artifacts and metrics against acceptance criteria
   - Logs promotion status and creates promotion records
   - **Acceptance Criteria:**
     - Minimum Accuracy: 70%
     - Minimum F1 Score: 65%

### 4. **deployment.yaml**
   - Kubernetes deployment manifest for EKS
   - Includes Namespace, Secret, Deployment, Service, and HPA configurations
   - Deployment is configured with:
     - 2 replicas (auto-scales 2-5 based on CPU/memory)
     - Resource requests/limits for proper scheduling
     - Liveness and readiness probes for health checks
     - LoadBalancer service for external access
     - Prometheus metrics scraping configuration

### 5. **.github/workflows/ci.yml** (Extended)
   - Added Docker build and push to ECR
   - Added EKS deployment steps
   - Triggered on main branch for production deployment

---

## Required GitHub Secrets & Variables

To run the CI/CD pipeline successfully, add the following secrets to your GitHub repository:

### Environment Secrets
Go to: **Settings → Secrets and variables → Actions**

#### Required Secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `DAGSHUB_TOKEN` | DagHub authentication token | `your_dagshub_token_here` |
| `DAGSHUB_USERNAME` | DagHub username for MLFlow tracking | `your_dagshub_username` |
| `AWS_ACCESS_KEY_ID` | AWS IAM user access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM user secret key | `wJal...` |
| `AWS_ACCOUNT_ID` | Your AWS account ID (12 digits) | `123456789012` |
| `AWS_REGION` | AWS region for resources | `us-east-1` |
| `ECR_REPOSITORY` | ECR repository name | `capstone-proj` |
| `CAPSTONE_TEST` | Authentication token for Flask app | Your token value |

---

## Step-by-Step Setup

### 1. **Create GitHub Secrets**

1. Go to your GitHub repository
2. Navigate to **Settings → Secrets and variables → Actions**
3. Click **New repository secret**
4. Add each secret from the table above

### 2. **Prepare AWS Resources**

#### Create IAM User (if not done):
```bash
# Via AWS Console or CLI
aws iam create-user --user-name cicd-user
```

#### Attach Policies to IAM User:
- `AmazonEC2FullAccess` (for EC2)
- `AmazonEC2ContainerRegistryFullAccess` (for ECR)
- `AmazonEKSFullAccess` (for EKS)

#### Generate Access Keys:
```bash
aws iam create-access-key --user-name cicd-user
```

#### Create S3 Bucket (for DVC storage):
```bash
aws s3 mb s3://your-dvc-storage-bucket --region us-east-1
```

#### Create ECR Repository:
```bash
aws ecr create-repository --repository-name capstone-proj --region us-east-1
```

### 3. **Create EKS Cluster** (if not done):

```bash
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name flask-app-nodes \
  --node-type t3.small \
  --nodes 1 \
  --nodes-min 1 \
  --nodes-max 1 \
  --managed
```

### 4. **Update kubectl Config**:

```bash
aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster
```

### 5. **Configure DVC S3 Remote** (locally):

```bash
dvc remote add -d myremote s3://your-dvc-storage-bucket
aws configure  # Set up AWS credentials
dvc push       # Push data to S3
git add dvc.lock
git commit -m "Update DVC remote"
```

---

## CI/CD Pipeline Workflow

The CI/CD pipeline runs on every push and follows these stages:

### **Stage 1: Testing**
- ✅ Checkout code
- ✅ Setup Python 3.10
- ✅ Cache pip dependencies
- ✅ Install dependencies
- ✅ Install DVC with S3 support
- ✅ Run DVC pipeline (`dvc repro`)
- ✅ Run model unit tests
- ✅ Promote model (if tests pass)
- ✅ Run Flask app tests

### **Stage 2: Containerization** (if tests pass)
- 🐳 Configure AWS credentials
- 🐳 Login to Amazon ECR
- 🐳 Build Docker image
- 🐳 Push image to ECR with commit SHA tag and latest tag

### **Stage 3: Deployment** (only on `main` branch, if all pass)
- ☸️ Update kubectl config
- ☸️ Deploy to EKS using deployment.yaml
- ☸️ Wait for rollout completion
- ☸️ Verify deployment status

---

## Debugging & Troubleshooting

### Check GitHub Actions Logs
1. Go to repository **Actions** tab
2. Click on the failed workflow run
3. Click on the failed job to see detailed logs

### Common Issues

#### 1. **DVC Remote Connection Failed**
```
Error: Could not connect to S3 bucket
```
- Ensure AWS credentials are correct in GitHub secrets
- Verify S3 bucket exists and is accessible

#### 2. **ECR Login Failed**
```
Error: RequestError - An error occurred (UnrecognizedClientException) when calling GetAuthorizationToken
```
- Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct
- Check IAM user has `AmazonEC2ContainerRegistryFullAccess` permission

#### 3. **EKS Deployment Failed**
```
Error: failed to call webhook "webhook.linkerd.io"
```
- Ensure EKS cluster exists and is running
- Update kubectl config: `aws eks update-kubeconfig --name flask-app-cluster --region us-east-1`

#### 4. **Model Tests Failed**
```
FileNotFoundError: models/model.pkl not found
```
- Ensure `dvc repro` completes successfully before running tests
- Check that model artifacts are being generated

---

## Manual Testing Locally

Before pushing to GitHub, test locally:

```bash
# Install test dependencies
pip install unittest

# Run model tests
python -m unittest tests.test_model

# Run Flask app tests
python -m unittest tests.test_flask_app

# Run DVC pipeline
dvc repro

# Promote model
python scripts/promote_model.py

# Build Docker image
docker build -t capstone-app:latest .

# Test Docker image locally
docker run -p 8888:5000 \
  -e CAPSTONE_TEST=your_token \
  -e DAGSHUB_TOKEN=your_dagshub_token \
  -e DAGSHUB_USERNAME=your_dagshub_username \
  capstone-app:latest
```

---

## Monitoring & Observability

### View EKS Deployment
```bash
# Get pods
kubectl get pods -n flask-app

# Get services
kubectl get svc -n flask-app

# Get deployment status
kubectl describe deployment flask-app -n flask-app

# View pod logs
kubectl logs -n flask-app -l app=flask-app --tail=100
```

### Access Flask App
```bash
# Get LoadBalancer external IP
kubectl get svc flask-app-service -n flask-app

# Access the app
curl http://<EXTERNAL-IP>:5000

# Access metrics endpoint
curl http://<EXTERNAL-IP>:5000/metrics
```

### Monitor with Prometheus & Grafana
- Prometheus server monitors metrics endpoint: `/metrics`
- Grafana visualizes metrics from Prometheus
- See projectflow.txt steps 50-67 for detailed setup

---

## Next Steps

1. ✅ **Create GitHub secrets** with AWS credentials
2. ✅ **Set up AWS resources** (IAM user, S3, ECR, EKS)
3. ✅ **Push to GitHub** - Pipeline will trigger automatically
4. ✅ **Monitor** workflow runs in Actions tab
5. ✅ **Access deployed app** using LoadBalancer external IP

---

## File Reference

| File | Purpose |
|------|---------|
| `tests/test_model.py` | Model unit tests |
| `tests/test_flask_app.py` | Flask app unit tests |
| `scripts/promote_model.py` | Model promotion logic |
| `deployment.yaml` | Kubernetes deployment manifest |
| `.github/workflows/ci.yml` | GitHub Actions workflow |
| `Dockerfile` | Container image definition |
| `flask_app/requirements.txt` | Python dependencies for Flask app |
| `requirements.txt` | Root project dependencies |

---

## Support

For issues or questions:
1. Check GitHub Actions logs for error messages
2. Review this guide for debugging steps
3. Verify all secrets and AWS resources are configured correctly
4. Test components locally before deploying

