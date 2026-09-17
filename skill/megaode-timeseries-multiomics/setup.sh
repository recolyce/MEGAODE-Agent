#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m pip install --upgrade pip
python3 -m pip install -r "${ROOT}/requirements.txt"

echo "megaode-timeseries-multiomics setup complete"
