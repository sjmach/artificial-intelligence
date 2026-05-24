#!/usr/bin/env bash
set -euo pipefail

# Step 1: Install NVIDIA driver and container toolkit
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart containerd

# Verify GPU access
nvidia-container-cli info

# Step 2: Install k3s
# Copy config.yaml to /etc/rancher/k3s/config.yaml before running this
sudo mkdir -p /etc/rancher/k3s
sudo cp config.yaml /etc/rancher/k3s/config.yaml

curl -sfL https://get.k3s.io | sh -
sudo systemctl restart k3s

# Step 3: Install Helm and configure kubeconfig
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo chmod 644 /etc/rancher/k3s/k3s.yaml

# Step 4: Install NVIDIA GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

kubectl create namespace gpu-operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=false

# Step 5: Test GPU access
kubectl apply -f cuda-test.yaml
kubectl get pods
kubectl logs cuda-test
