#!/usr/bin/env bash
set -euo pipefail
umask 022

tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_dir=${1:-$tool_dir/.runtime}
if [[ -e "$runtime_dir/venv" ]]; then
  printf 'Runtime already exists; choose a new directory to preserve active/resumable runs.\n' >&2
  exit 1
fi
mkdir -p "$runtime_dir"
runtime_dir=$(cd "$runtime_dir" && pwd -P)
mkdir -p "$runtime_dir/manifests"

# Run on the Linux cluster. uv must already be installed on the login node.
command -v uv >/dev/null
# The interpreter must be on shared storage too, not in the installer's home.
export UV_PYTHON_INSTALL_DIR="$runtime_dir/python"
export UV_LINK_MODE=copy
uv python install --no-bin 3.12.12
uv venv --managed-python --python 3.12.12 --seed "$runtime_dir/venv"
uv pip install --python "$runtime_dir/venv/bin/python" -r "$tool_dir/requirements-gpu.txt"
"$runtime_dir/venv/bin/python" -m pip check
"$runtime_dir/venv/bin/python" -m pip freeze > "$runtime_dir/manifests/environment.txt"
"$runtime_dir/venv/bin/python" -I - "$runtime_dir" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1])
assert Path(sys.base_prefix).resolve().is_relative_to(root / 'python'), sys.base_prefix
versions = {name: importlib.metadata.version(name)
            for name in ['vllm', 'torch', 'transformers', 'huggingface-hub', 'jsonschema']}
(root / 'manifests/runtime.json').write_text(
    json.dumps({'python': platform.python_version(), 'base_prefix': sys.base_prefix,
                'packages': versions}, indent=2) + '\n')
PY
printf 'Runtime installed at %s\n' "$runtime_dir"
