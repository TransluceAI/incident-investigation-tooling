#!/bin/bash
# Build a cluster release of inference/qwen from a git commit, never from a working tree.
set -euo pipefail
usage='usage: build_release.sh <commit> <new-release-dir>'
commit=${1:?$usage}
dest=${2:?$usage}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
sha=$(git -C "$repo" rev-parse --verify "$commit^{commit}")
if [[ -e "$dest" ]]; then
  echo "Release directory already exists: $dest" >&2
  exit 1
fi
mkdir -p "$dest"
git -C "$repo" archive "$sha" inference/qwen | tar -x -C "$dest" --strip-components=2
printf '%s\n' "$sha" > "$dest/RELEASE"
echo "Release $sha at $dest"
