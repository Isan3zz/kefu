"""Explicit network-only downloader for the pinned Chinese-CLIP snapshot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


REPO = 'OFA-Sys/chinese-clip-vit-base-patch16'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision')
    parser.add_argument('--local-dir', type=Path, default=Path('models/chinese-clip'))
    args = parser.parse_args()
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ.pop(name, None)
    from huggingface_hub import HfApi, snapshot_download

    manifest = args.local_dir/'model_manifest.json'
    try:
        pinned = args.revision
        if not pinned and manifest.is_file():
            pinned = json.loads(manifest.read_text(encoding='utf-8'))['revision']
        info = HfApi().model_info(REPO, revision=pinned, timeout=30)
        commit = info.sha
        if not re.fullmatch(r'[a-f0-9]{40}', commit or ''):
            raise ValueError('Provider did not return an immutable model commit')
        files = {file.rfilename for file in info.siblings}
        weight = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
        required = {'config.json', 'preprocessor_config.json', weight, 'vocab.txt'}
        if not required <= files:
            raise ValueError(f'Missing snapshot files: {sorted(required-files)}')
        prior = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
        if prior and prior.get('revision') != commit:
            raise ValueError('Use a separate --local-dir for a different revision, then switch paths and rebuild')
        candidates = required | {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'processor_config.json'}
        snapshot_download(REPO, revision=commit, local_dir=args.local_dir, allow_patterns=sorted(candidates & files),
                          etag_timeout=30)
        alternate = args.local_dir / ('pytorch_model.bin' if weight == 'model.safetensors' else 'model.safetensors')
        if alternate.exists():
            raise ValueError('The target contains an alternate weight file. Use a clean --local-dir.')
        payload = dict(repo_id=REPO, revision=commit, dimension=512, weight=weight,
                       downloaded_at=datetime.now(timezone.utc).isoformat())
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temporary.replace(manifest)
        print(json.dumps(payload))
    except Exception as error:
        parser.exit(1, f'Chinese-CLIP download failed: {error}\n')


if __name__ == '__main__':
    main()
