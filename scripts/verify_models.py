from app.rag import local_models,model_manifest
def main():
    from pathlib import Path
    import hashlib
    manifest=model_manifest()
    for key in ('embedding','reranker'):
        entry=manifest[key]
        if not isinstance(entry,dict): raise FileNotFoundError('Run python -m scripts.download_models --all first')
        for name,expected in entry['sha256'].items():
            path=Path(entry['path'])/name
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
            if h.hexdigest()!=expected: raise ValueError(f'Model integrity mismatch: {path}')
    m=local_models(); v=m.encode(['耳机可以七天无理由退货吗？'])
    score=m.score([('退款政策','符合条件的订单支持七天退款。')])
    print({'embedding_shape':v.shape,'reranker_scores':score,'manifest':model_manifest()})
if __name__=='__main__': main()
