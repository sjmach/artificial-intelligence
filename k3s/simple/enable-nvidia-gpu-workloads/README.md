# How to Enable NVIDIA GPU Workloads on k3s

A setup script and supporting manifests for configuring a k3s Kubernetes cluster to recognise and schedule NVIDIA GPU workloads.

**Article:** [How to Enable NVIDIA GPU Workloads on k3s](https://www.sundeepmachado.com/2025/12/how-to-enable-nvidia-gpu-workloads-on.html)

## Files

| File | Purpose |
|---|---|
| `config.yaml` | k3s cluster config — placed at `/etc/rancher/k3s/config.yaml` before installation |
| `setup.sh` | End-to-end setup script covering all five steps below |
| `cuda-test.yaml` | Pod manifest that requests one GPU and runs `nvidia-smi` to verify access |

## Prerequisites

- Ubuntu host with a supported NVIDIA GPU
- `sudo` access
- Internet access (driver, k3s, and Helm are downloaded during setup)

## Quick start

```bash
bash setup.sh
```

Or follow the steps manually:

### Step 1 — Install NVIDIA driver and container toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit
sudo systemctl restart containerd
nvidia-container-cli info   # verify GPU access
```

### Step 2 — Configure and install k3s

`config.yaml` must be in place before k3s is installed so the installer picks it up:

```bash
sudo mkdir -p /etc/rancher/k3s
sudo cp config.yaml /etc/rancher/k3s/config.yaml
curl -sfL https://get.k3s.io | sh -
sudo systemctl restart k3s
```

`config.yaml` sets `write-kubeconfig-mode: "0644"` so Helm and kubectl can read the kubeconfig without sudo.

### Step 3 — Install Helm and expose kubeconfig

```bash
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
```

### Step 4 — Deploy NVIDIA GPU Operator

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
kubectl create namespace gpu-operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=false
```

`driver.enabled=false` tells the Operator not to manage the driver — you installed it manually in Step 1.

### Step 5 — Verify GPU access

```bash
kubectl apply -f cuda-test.yaml
kubectl logs cuda-test
```

A successful run prints the `nvidia-smi` table showing your GPU model and driver version.
