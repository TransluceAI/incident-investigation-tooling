"""Download a pinned model profile and write a config with a local snapshot path."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def prepare(profile, runtime):
    from huggingface_hub import snapshot_download

    config = json.loads(profile.read_text())
    model_id, revision = config['model'], config['model_revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('model_revision must be a pinned 40-character commit SHA')
    runtime = runtime.expanduser().absolute()
    snapshot = Path(snapshot_download(model_id, revision=revision,
                                     cache_dir=str(runtime / 'models'), max_workers=8))
    if not any(snapshot.glob('*.safetensors')):
        raise RuntimeError('No weights in downloaded snapshot')
    config['model'] = str(snapshot)
    config_path = runtime / 'config' / profile.name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT / 'config/qwen3.8-27b-fp8.json')
    parser.add_argument('--runtime-dir', type=Path, default=ROOT / '.runtime')
    args = parser.parse_args()
    print(prepare(args.profile, args.runtime_dir))


if __name__ == '__main__':
    main()
