#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-llama3.2:3b-clean}"
PROBE="${2:-dan.DanInTheWild}"

source garak-env/bin/activate

garak --target_type ollama --target_name "$MODEL_NAME" --probes "$PROBE"
