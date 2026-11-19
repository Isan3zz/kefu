"""Local-only dense + Chinese BM25 + RRF + cross encoder retrieval."""
import json
import hashlib
import re
import time
from pathlib import Path
from functools import lru_cache
from threading import Lock
from contextlib import contextmanager
import logging
from datetime import datetime, timezone
import numpy as np
from sqlalchemy import select
from app.config import settings
from app.db import documents, document_chunks, knowledge_snapshots, uid, now

CHUNKER_VERSION='paragraph-v2-400-60'
MODEL_LOAD_LOCK = Lock()

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.'+uid()+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

def require_model(path):
    path=Path(path).resolve()
    required=['config.json','tokenizer_config.json']
    missing=[f for f in required if not (path/f).is_file()]
    if not any((path/name).is_file() for name in ('tokenizer.json','vocab.txt','tokenizer.model','sentencepiece.bpe.model','spiece.model')): missing.append('tokenizer vocabulary')
    weights=list(path.glob('*.safetensors'))+list(path.glob('pytorch_model*.bin'))
    if not any(file.is_file() and file.stat().st_size>0 for file in weights): missing.append('*.safetensors or pytorch_model*.bin')
    if not path.is_dir() or missing:
        raise FileNotFoundError(f'缺少本地模型文件: {path}; missing={missing}; 请显式运行 python -m scripts.download_models --all')
    return path

@contextmanager
def model_memory_guard():
    try: yield
    except RuntimeError as error:
        import torch
        if isinstance(error,torch.cuda.OutOfMemoryError):
            logging.getLogger(__name__).error('Local model CUDA OOM; configure MODEL_DEVICE=cpu and restart')
            raise RuntimeError('本地模型显存不足；请设置 MODEL_DEVICE=cpu 后重新启动，不会在请求中反复切换模型。') from error
        raise

class LocalModels:
    def __init__(self,config=settings):
        import os
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        ep=require_model(config.embedding_model_path); rp=require_model(config.reranker_model_path)
        import torch
        from sentence_transformers import SentenceTransformer,CrossEncoder
        device=('cuda' if torch.cuda.is_available() else 'cpu') if config.model_device=='auto' else config.model_device
        self.config=config
        with model_memory_guard():
            self.embedder=SentenceTransformer(str(ep),device=device,local_files_only=True)
            self.embedder.max_seq_length=config.embed_max_seq_length
            self.reranker=CrossEncoder(str(rp),device=device,local_files_only=True,max_length=config.rerank_max_length)
    def encode(self,texts):
        with model_memory_guard():
            v=self.embedder.encode(texts,batch_size=self.config.embed_batch_size,convert_to_numpy=True,normalize_embeddings=True)
        if v.shape!=(len(texts),1024): raise ValueError('BGE-M3 embedding must have 1024 dimensions')
        return v
    def score(self,pairs):
        with model_memory_guard():
            return self.reranker.predict(pairs,batch_size=self.config.rerank_batch_size).tolist()

@lru_cache(maxsize=1)
def _cached_models(model_config, revision):
    # The revision is part of the key even when local paths stay the same.
    return LocalModels(settings.model_copy(update=json.loads(model_config)))

def local_models(config=settings):
    keys=('embedding_model_path','reranker_model_path','model_device','embed_batch_size',
          'rerank_batch_size','embed_max_seq_length','rerank_max_length')
    values={key:getattr(config,key) for key in keys}
    with MODEL_LOAD_LOCK:
        return _cached_models(json.dumps(values,sort_keys=True),json.dumps(model_identity(model_manifest()),sort_keys=True))

local_models.cache_info=_cached_models.cache_info
local_models.cache_clear=_cached_models.cache_clear

def tokenize(text):
    import jieba
    return [t.lower() for t in jieba.lcut(text) if t.strip()]+re.findall(r'[A-Za-z]+[\w-]*|\d+',text.lower())

def chunks(text,size=400,overlap=60):
    if size<1 or overlap<0 or overlap>=size: raise ValueError('Require size > overlap >= 0')
    text=text.strip(); result=[]; start=0
    while start<len(text):
        end=min(start+size,len(text))
        if end<len(text):
            boundary=text.rfind('\n\n',max(start+size//2,start+overlap+1),end)
            if boundary>=0: end=boundary+2
        piece=text[start:end]
        if piece.strip(): result.append(piece)
        if end==len(text): break
        start=end-overlap
    return result

def visible(d,tenant,clock=None):
    clock=clock or datetime.now(timezone.utc)
    return d['tenant_id'] in (tenant,'public') and d['status']=='published' and d['visibility'] in ('public','customer') and datetime.fromisoformat(d['effective_at'])<=clock and (not d['expires_at'] or clock<datetime.fromisoformat(d['expires_at']))

class RAG:
    def __init__(self,db,models=None,config=settings):
        self.db,self.models,self.config=db,models,config
        self.path=Path(config.chroma_path)
        self._client=None
        self._client_lock=Lock()
    def model(self): return self.models or local_models(self.config)
    def provenance(self):
        if self.models is None: return model_manifest()
        entry={'repo':'INJECTED','revision':'INJECTED_NOT_VERIFIED','implementation':type(self.models).__module__+'.'+type(self.models).__qualname__}
        return {'embedding':entry.copy(),'reranker':entry.copy()}
    def client(self):
        import chromadb
        from chromadb.config import Settings
        with self._client_lock:
            if self._client is None:
                self._client=chromadb.PersistentClient(path=str(self.path),settings=Settings(anonymized_telemetry=False))
            return self._client
    def close(self):
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client=None
    def active_manifest(self,connection=None):
        if connection is None:
            with self.db.tx() as c: return self.active_manifest(c)
        row=connection.execute(select(knowledge_snapshots.c.payload).where(knowledge_snapshots.c.id=='active')).first()
        if row is None: raise FileNotFoundError('缺少有效知识快照；请显式运行 python -m scripts.build_kb')
        return json.loads(row[0])
    def build(self,knowledge):
        import yaml
        rows=[]; docs=[]
        for path in sorted(Path(knowledge).rglob('*.md')):
            text=path.read_text(encoding='utf-8')
            _,front,body=text.split('---',2); meta=yaml.safe_load(front)
            d=dict(id=meta['id'],tenant_id=meta.get('tenant_id','public'),title=meta['title'],version=str(meta['version']),status=meta.get('status','published'),effective_at=meta.get('effective_at','2020-01-01T00:00:00+00:00'),expires_at=meta.get('expires_at'),visibility=meta.get('visibility','public'),source_path=str(path))
            docs.append(d)
            for idx,content in enumerate(chunks(body.strip())):
                h=hashlib.sha256(content.encode()).hexdigest()
                cid=hashlib.sha256(f"{d['id']}:{d['version']}:{idx}:{h}".encode()).hexdigest()[:24]
                rows.append(dict(chunk_id=cid,document_id=d['id'],chunk_index=idx,content=content,content_hash=h,metadata_json=json.dumps(d,ensure_ascii=False)))
        if not rows: raise ValueError('Empty knowledge corpus')
        vectors=np.asarray(self.model().encode([r['content'] for r in rows]))
        if vectors.shape!=(len(rows),1024): raise ValueError('Expected 1024 dimensional embeddings')
        snapshot='kb-'+uid(); collection=self.client().create_collection(snapshot,embedding_function=None,configuration={'hnsw':{'space':'cosine'}})
        collection.add(ids=[r['chunk_id'] for r in rows],embeddings=vectors.tolist(),metadatas=[{'tenant_id':json.loads(r['metadata_json'])['tenant_id'],'document_id':r['document_id']} for r in rows])
        # Both stores are immutable per snapshot. The pointer changes only after full success.
        artifact=Path(self.config.bm25_index_path)/(snapshot+'.json')
        atomic_json(artifact,rows)
        manifest={'snapshot':snapshot,'bm25':str(artifact),'chunk_count':len(rows),'corpus_hash':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'built_at':now(),'models':self.provenance(),'chunker':CHUNKER_VERSION}
        with self.db.tx() as c:
            for d in docs:
                exists=c.execute(select(documents.c.id).where(documents.c.id==d['id'])).first()
                if exists: c.execute(documents.update().where(documents.c.id==d['id']).values(**d))
                else: c.execute(documents.insert().values(**d))
            for r in rows:
                if not c.execute(select(document_chunks.c.chunk_id).where(document_chunks.c.chunk_id==r['chunk_id'])).first(): c.execute(document_chunks.insert().values(**r))
            # A failed pointer write also rolls back document version/governance updates.
            atomic_json(self.path/'active.json',manifest)
            payload=json.dumps(manifest,ensure_ascii=False)
            c.execute(knowledge_snapshots.insert().values(id=snapshot,payload=payload))
            if c.execute(select(knowledge_snapshots.c.id).where(knowledge_snapshots.c.id=='active')).first():
                c.execute(knowledge_snapshots.update().where(knowledge_snapshots.c.id=='active').values(payload=payload))
            else: c.execute(knowledge_snapshots.insert().values(id='active',payload=payload))
        return manifest
    def search(self,query,tenant,mode='hybrid_rerank',k=None,*,document_ids=None):
        if mode not in ('dense','bm25','hybrid','hybrid_rerank'): raise ValueError('Invalid retrieval mode')
        started=time.perf_counter()
        with self.db.tx() as c:
            manifest=self.active_manifest(c)
            docs={d['id']:dict(d) for d in c.execute(select(documents)).mappings()}
        if self.models is None and model_identity(manifest['models'])!=model_identity(model_manifest()): raise ValueError('Model revision changed: rebuild knowledge index')
        rows=json.loads(Path(manifest['bm25']).read_text(encoding='utf-8'))
        # Enforce current DB governance BEFORE both retrieval branches, including stale index entries.
        rows=[r for r in rows if r['document_id'] in docs and visible(docs[r['document_id']],tenant) and json.loads(r['metadata_json'])['version']==docs[r['document_id']]['version']]
        if document_ids is not None:
            rows = [row for row in rows if row['document_id'] in document_ids]
        if not rows: return []
        by_id={r['chunk_id']:r for r in rows}; rankings=[]; stages={}
        if mode!='bm25':
            t=time.perf_counter(); v=np.asarray(self.model().encode([query]))
            if v.shape!=(1,1024): raise ValueError('Expected 1024 dimensional query')
            col=self.client().get_collection(manifest['snapshot'],embedding_function=None)
            found=col.query(query_embeddings=v.tolist(),n_results=min(len(rows),self.config.retrieval_dense_k),where={'$and':[{'tenant_id':{'$in':[tenant,'public']}},{'document_id':{'$in':sorted({r['document_id'] for r in rows})}}]},include=['distances'])
            rankings.append([cid for cid in found['ids'][0] if cid in by_id]); stages['dense_ms']=(time.perf_counter()-t)*1000
        if mode!='dense':
            from rank_bm25 import BM25Okapi
            t=time.perf_counter(); bm=BM25Okapi([tokenize(r['content']) for r in rows]); scores=bm.get_scores(tokenize(query))
            order=sorted(range(len(rows)),key=lambda j:(-scores[j],rows[j]['chunk_id']))[:self.config.retrieval_bm25_k]
            rankings.append([rows[j]['chunk_id'] for j in order]); stages['bm25_ms']=(time.perf_counter()-t)*1000
        fusion={}
        for rank in rankings:
            for idx,cid in enumerate(rank,1): fusion[cid]=fusion.get(cid,0)+1/(self.config.retrieval_rrf_k+idx)
        chosen=sorted(fusion,key=lambda cid:(-fusion[cid],cid))
        rerank={}
        if mode=='hybrid_rerank':
            t=time.perf_counter(); chosen=chosen[:self.config.retrieval_rerank_k]
            scores=self.model().score([(query,by_id[cid]['content']) for cid in chosen])
            rerank=dict(zip(chosen,map(float,scores))); chosen.sort(key=lambda cid:(-rerank[cid],cid)); stages['rerank_ms']=(time.perf_counter()-t)*1000
        revisions={key+'_revision':value.get('revision') if isinstance(value,dict) else value for key,value in manifest['models'].items() if key in ('embedding','reranker')}
        return [dict(chunk_id=cid,doc_id=by_id[cid]['document_id'],content=by_id[cid]['content'],document_title=docs[by_id[cid]['document_id']]['title'],version=docs[by_id[cid]['document_id']]['version'],source=docs[by_id[cid]['document_id']]['source_path'],retrieval_rank=idx,rrf_score=fusion[cid],rerank_score=rerank.get(cid),snapshot=manifest['snapshot'],stages=stages,latency_ms=(time.perf_counter()-started)*1000,**revisions) for idx,cid in enumerate(chosen[:k or self.config.retrieval_context_k],1)]

def model_manifest():
    path=Path('models/model_manifest.json')
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'embedding':'NOT_DOWNLOADED','reranker':'NOT_DOWNLOADED'}

def model_identity(manifest):
    # Download timestamps, source mirrors and physical paths do not change vector space.
    result={}
    for key in ('embedding','reranker'):
        entry=manifest.get(key,'NOT_DOWNLOADED')
        result[key]={field:entry[field] for field in ('repo','revision','sha256','implementation') if field in entry} if isinstance(entry,dict) else entry
    return result
