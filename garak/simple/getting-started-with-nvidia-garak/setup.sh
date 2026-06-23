#!/usr/bin/env bash
set -euo pipefail

python -m venv garak-env
source garak-env/bin/activate
pip install --upgrade pip
pip install garak

echo "Garak installed. Activate with: source garak-env/bin/activate"
