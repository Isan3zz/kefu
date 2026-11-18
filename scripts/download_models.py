"""The only weight-download entry point. Run explicitly, never at API startup."""
import os
import argparse
import hashlib
import json
import sys
import threading
import time
from urllib.parse import urlparse,parse_qs
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor,as_completed
import re
import fnmatch
from pathlib import Path
from datetime import datetime,timezone

MODELS={'embedding':('BAAI/bge-m3',Path('models/embedding/bge-m3')),'reranker':('BAAI/bge-reranker-v2-m3',Path('models/reranker/bge-reranker-v2-m3'))}
MODELSCOPE_REPOS={'embedding':'BAAI/bge-m3','reranker':'AI-ModelScope/bge-reranker-v2-m3'}

def alternate_location(source,key,repo,revision,filename):
    if source=='modelscope': return f'https://modelscope.cn/models/{MODELSCOPE_REPOS[key]}/resolve/master/{filename}'
    if source=='hf-mirror': return f'https://hf-mirror.com/{repo}/resolve/{revision}/{filename}'
    return None

def file_matches(path,metadata):
    if not path.is_file() or path.stat().st_size!=metadata.size: return False
    etag=metadata.etag.strip('"')
    if re.fullmatch('[a-f0-9]{64}',etag): digest=hashlib.sha256()
    elif re.fullmatch('[a-f0-9]{40}',etag):
        digest=hashlib.sha1(); digest.update(b'blob '+str(metadata.size).encode()+b'\0')
    else: return False
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()==etag

def download_ranges(repo,revision,filename,local_dir,metadata=None,transport=None,chunk_size=8*1024*1024,workers=4):
    """Bounded HTTP Range fallback; resume only pinned, length-verified generated parts."""
    import httpx
    if metadata is None:
        from huggingface_hub import get_hf_file_metadata,hf_hub_url
        metadata=get_hf_file_metadata(hf_hub_url(repo,filename,revision=revision),timeout=30)
    local_dir=Path(local_dir).resolve(); target=local_dir/filename
    if not target.resolve().is_relative_to(local_dir): raise ValueError('Download path escapes model directory')
    target.parent.mkdir(parents=True,exist_ok=True)
    if file_matches(target,metadata): return target
    if not metadata.size or metadata.size<1: raise ValueError('Missing authoritative file size')
    identity=hashlib.sha256(f'{repo}:{revision}:{filename}:{metadata.etag}:{metadata.size}'.encode()).hexdigest()
    parts=local_dir/'.cache/http-ranges'/identity; parts.mkdir(parents=True,exist_ok=True)
    invalid=parts/'invalid'
    force=invalid.exists()
    def part_path(index): return parts/f'{index:06d}.part'
    with httpx.Client(timeout=30,follow_redirects=True,transport=transport) as client:
        def fetch(index):
            start=index*chunk_size; end=min(metadata.size,start+chunk_size)-1; count=end-start+1
            part=part_path(index)
            if not force and part.is_file() and part.stat().st_size==count: return count
            for attempt in range(3):
                try:
                    with client.stream('GET',metadata.location,headers={'Range':f'bytes={start}-{end}'}) as response:
                        full_small_file=response.status_code==200 and start==0 and count==metadata.size
                        if not full_small_file and (response.status_code!=206 or response.headers.get('content-range')!=f'bytes {start}-{end}/{metadata.size}'):
                            raise ValueError(f'Invalid range response: HTTP {response.status_code}')
                        temporary=part.with_suffix('.tmp'); received=0
                        with temporary.open('wb') as file:
                            for block in response.iter_bytes(chunk_size=1024*1024):
                                received+=len(block)
                                if received>count: raise ValueError('Range exceeds declared size')
                                file.write(block)
                        if received!=count: raise ValueError('Incomplete HTTP range')
                        temporary.replace(part)
                        return count
                except Exception:
                    if attempt==2: raise
                    time.sleep(.5*2**attempt)
        completed=0; last_progress=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(fetch,index) for index in range((metadata.size+chunk_size-1)//chunk_size)]
            try:
                for future in as_completed(jobs):
                    completed+=future.result()
                    if completed-last_progress>=64*1024*1024 or completed==metadata.size:
                        print(f'{repo}/{filename}: {completed}/{metadata.size} bytes ({completed/metadata.size:.1%})',flush=True)
                        last_progress=completed
            except BaseException:
                for job in jobs: job.cancel()
                raise
    assembled=target.with_suffix(target.suffix+'.http-download')
    with assembled.open('wb') as file:
        for index in range((metadata.size+chunk_size-1)//chunk_size):
            with part_path(index).open('rb') as part:
                for block in iter(lambda:part.read(8*1024*1024),b''): file.write(block)
    if not file_matches(assembled,metadata):
        invalid.write_text('Redownload generated parts on the next explicit retry.',encoding='utf-8')
        raise ValueError('Downloaded file failed pinned upstream hash verification')
    assembled.replace(target)
    # Only generated single-part files below the verified local directory are removed.
    for file in parts.glob('*.part'): file.unlink()
    if invalid.exists(): invalid.unlink()
    return target

def main():
    p=argparse.ArgumentParser(); p.add_argument('--all',action='store_true'); p.add_argument('--model',choices=MODELS); p.add_argument('--revision'); p.add_argument('--disable-xet',action='store_true'); p.add_argument('--diagnose',action='store_true'); p.add_argument('--probe-bytes',type=int,default=4096); p.add_argument('--http-range',action='store_true'); p.add_argument('--source',choices=['huggingface','modelscope','hf-mirror'],default='huggingface'); p.add_argument('--workers',type=int,default=4); p.add_argument('--timeout-seconds',type=int,default=1800); args=p.parse_args()
    # Only explicit CLI execution opts into network mode, before importing the Hub SDK.
    os.environ['HF_HUB_OFFLINE']='0'
    os.environ['TRANSFORMERS_OFFLINE']='0'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
    if args.timeout_seconds<1: p.error('--timeout-seconds must be positive')
    if not 1<=args.workers<=32: p.error('--workers must be between 1 and 32')
    if not 1<=args.probe_bytes<=8*1024*1024: p.error('--probe-bytes must be between 1 and 8388608')
    if args.source!='huggingface' and not (args.http_range or args.diagnose): p.error('Alternate sources require --http-range or --diagnose')
    def timeout():
        print('Model download exceeded wall-clock timeout; incomplete files preserved for explicit retry. No success manifest was written for incomplete models.',file=sys.stderr,flush=True)
        os._exit(2)
    watchdog=threading.Timer(args.timeout_seconds,timeout); watchdog.daemon=True; watchdog.start()
    if args.disable_xet: os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi,snapshot_download
    if not args.all and not args.model: p.error('Choose --all or --model')
    if args.all and args.revision: p.error('--revision requires a single --model')
    manifest_path=Path('models/model_manifest.json'); manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    root=Path('models').resolve()
    for key in MODELS if args.all else [args.model]:
        repo,path=MODELS[key]
        if not path.resolve().is_relative_to(root): raise ValueError('Invalid model target')
        info=HfApi(endpoint='https://huggingface.co').model_info(repo,revision=args.revision,timeout=30)
        revision=info.sha
        files=[s.rfilename for s in info.siblings]
        weights=[n for n in files if n.endswith('.safetensors') and not n.startswith('onnx/')]
        if not weights: weights=[n for n in files if Path(n).name.startswith('pytorch_model') and n.endswith('.bin')]
        if not weights: raise SystemExit(f'No supported PyTorch weights in {repo}@{revision}')
        if args.diagnose:
            from huggingface_hub import get_hf_file_metadata,hf_hub_url
            import httpx
            for weight in weights:
                started=time.monotonic()
                try:
                    metadata=get_hf_file_metadata(hf_hub_url(repo,weight,revision=revision),timeout=15)
                    alternative=alternate_location(args.source,key,repo,revision,weight)
                    if alternative: metadata=replace(metadata,location=alternative)
                    location=urlparse(metadata.location)
                    expiration=parse_qs(location.query).get('Expires',[None])[0]
                    print(f'{repo}@{revision}: file={weight}, size={metadata.size}, download_host={location.hostname}, expired_link={float(expiration)<time.time() if expiration else "unknown"}',flush=True)
                    with httpx.stream('GET',metadata.location,headers={'Range':f'bytes=0-{args.probe_bytes-1}'},follow_redirects=True,timeout=15) as response:
                        status=response.status_code
                        sample=next(response.iter_bytes(chunk_size=args.probe_bytes),b'') if response.is_success else b''
                        print(f'Download probe: status={status}, sample_bytes={len(sample)}, sample_sha256={hashlib.sha256(sample).hexdigest() if sample else "none"}, elapsed_seconds={time.monotonic()-started:.1f}',flush=True)
                except Exception as error:
                    # Never print signed URLs or auth headers from exception strings.
                    print(f'Download probe failed: {type(error).__name__}, elapsed_seconds={time.monotonic()-started:.1f}',flush=True)
            continue
        print(f'Downloading {repo}@{revision}; weights={weights}',flush=True)
        try:
            if args.http_range:
                small=['config.json','tokenizer_config.json','modules.json','1_Pooling/config.json','sentence_bert_config.json','config_sentence_transformers.json']
                snapshot_download(repo_id=repo,revision=revision,local_dir=str(path),allow_patterns=small,max_workers=4)
                selected=[name for name in files if not name.startswith('onnx/') and any(fnmatch.fnmatch(name,pattern) for pattern in ['*.json','*.model','*.txt',*weights])]
                for name in selected:
                    metadata=None
                    if args.source!='huggingface' and name in weights:
                        from huggingface_hub import get_hf_file_metadata,hf_hub_url
                        metadata=get_hf_file_metadata(hf_hub_url(repo,name,revision=revision,endpoint='https://huggingface.co'),timeout=30)
                        metadata=replace(metadata,location=alternate_location(args.source,key,repo,revision,name))
                    download_ranges(repo,revision,name,path,metadata=metadata,workers=args.workers)
            else: snapshot_download(repo_id=repo,revision=revision,local_dir=str(path),allow_patterns=['*.json','*.model','*.txt',*weights],ignore_patterns=['onnx/**'],max_workers=4)
        except Exception as e: raise SystemExit(f'Download failed for {repo}@{revision}; rerun to resume: {e}')
        checks={}
        for file in sorted(path.rglob('*')):
            if file.is_file() and '.cache' not in file.parts:
                h=hashlib.sha256()
                with file.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
                checks[str(file.relative_to(path))]=h.hexdigest()
        manifest[key]=dict(repo=repo,revision=revision,path=str(path),source=args.source,downloaded_at=datetime.now(timezone.utc).isoformat(),sha256=checks)
        manifest_path.parent.mkdir(parents=True,exist_ok=True)
        temp=manifest_path.with_suffix('.tmp'); temp.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); temp.replace(manifest_path)
        print(f'{key}: {repo}@{revision}; files={len(checks)}')
    watchdog.cancel()
if __name__=='__main__': main()
