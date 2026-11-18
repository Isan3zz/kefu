"""Explicit offline testing models. Never selected implicitly by runtime."""
import hashlib
import json
import numpy as np
from app.harness.model_gateway import FakeModelGateway
class FakeModels:
    def encode(self,texts):
        vectors=[]
        for text in texts:
            v=np.zeros(1024,dtype=np.float32)
            for char in text: v[int.from_bytes(hashlib.sha256(char.encode()).digest()[:2],'big')%1024]+=1
            vectors.append(v/max(np.linalg.norm(v),1))
        return np.array(vectors)
    def score(self,pairs): return [len(set(q)&set(p)) for q,p in pairs]


class FixtureMemoryGateway(FakeModelGateway):
    """Offline only: candidates are scripted per exact source message, never inferred."""
    def __init__(self, config, fixtures):
        super().__init__(config)
        self.fixtures = fixtures

    async def complete(self, **kwargs):
        if kwargs['purpose'] == 'memory_extract':
            sources = json.loads(kwargs['messages'][1]['content'])['user_messages']
            candidates = [{'source_message_id': source['source_message_id'], 'is_user_asserted': True, **candidate}
                          for source in sources for candidate in self.fixtures.get(source['text'], [])]
            kwargs['task_state'] = {'structured': {'candidates': candidates}}
        return await super().complete(**kwargs)
