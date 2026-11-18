import math

from sqlalchemy import select

from app.db import memories
from app.memory.repository import active


class ChromaMemoryIndex:
    def __init__(self, rag, config):
        self.rag, self.name = rag, config.memory_collection_name

    @property
    def version(self):
        metadata = self.rag.provenance().get('embedding', {})
        return ':'.join(str(metadata[key]) for key in ('repo', 'revision', 'implementation') if metadata.get(key)) or 'unverified-local'

    def collection(self):
        result = self.rag.client().get_or_create_collection(
            self.name, metadata={'embedding_version': self.version}, embedding_function=None,
            configuration={'hnsw': {'space': 'cosine'}},
        )
        if (result.metadata or {}).get('embedding_version') != self.version:
            raise ValueError('Memory embedding version changed; rebuild the memory collection')
        return result

    def vectors(self, texts):
        vectors = [list(map(float, vector)) for vector in self.rag.model().encode(texts)]
        if len(vectors) != len(texts) or any(len(vector) != 1024 or not all(map(math.isfinite, vector)) for vector in vectors):
            raise ValueError('Memory embeddings must be finite BGE-M3 1024-dimensional vectors')
        return vectors

    def query(self, principal, query, count):
        collection = self.collection()
        result = collection.query(query_embeddings=self.vectors([query]), n_results=count,
                                  where={'$and': [{'tenant_id': principal.tenant_id}, {'user_id': principal.user_id}]},
                                  include=['distances', 'metadatas'])
        return [(ident, 1 - distance, metadata) for ident, distance, metadata in
                zip(result['ids'][0], result['distances'][0], result['metadatas'][0], strict=True)]

    def sync(self, repository, memory_id):
        with repository.db.tx() as connection:
            snapshot = connection.execute(select(memories).where(memories.c.id == memory_id)).mappings().first()
        if not snapshot or not snapshot['index_pending']:
            return
        is_active = snapshot['status'] == 'active' and (not snapshot['expires_at'] or snapshot['expires_at'] > repository.stamp())
        version = self.version
        vector = self.vectors([snapshot['content']]) if is_active else None
        with repository.db.tx() as connection:
            row = connection.execute(select(memories).where(memories.c.id == memory_id)).mappings().one()
            if not row['index_pending'] or row['version'] != snapshot['version'] or version != self.version:
                return
            if is_active and (not row['expires_at'] or row['expires_at'] > repository.stamp()):
                metadata = {key: row[key] for key in ('tenant_id', 'user_id', 'memory_type', 'subject_key', 'version')}
                self.collection().upsert(ids=[memory_id], embeddings=vector, documents=[row['content']],
                                         metadatas=[{**metadata, 'embedding_version': version}])
            else:
                client = self.rag.client()
                if self.name in {item.name for item in client.list_collections()}:
                    client.get_collection(self.name, embedding_function=None).delete(ids=[memory_id])
            connection.execute(memories.update().where(memories.c.id == memory_id)
                               .values(index_pending=0, embedding_version=version if is_active else None))

    def rebuild(self, repository, dry_run=False):
        with repository.db.tx() as connection:
            ids = list(connection.execute(select(memories.c.id).where(*active(repository.stamp()))).scalars())
            if dry_run:
                return len(ids)
            client = self.rag.client()
            if self.name in {item.name for item in client.list_collections()}:
                client.delete_collection(self.name)
            connection.execute(memories.update().values(index_pending=1, embedding_version=None))
        for memory_id in ids:
            self.sync(repository, memory_id)
        return len(ids)
