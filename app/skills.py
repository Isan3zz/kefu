import re
import time
import hashlib
import yaml
from pathlib import Path
from app.config import ROOT
from app.mcp import ALLOW
from pydantic import BaseModel

class SkillSelection(BaseModel):
    skill_name:str|None
    confidence:float
    reason:str

class SkillManager:
    def __init__(self,root=None):
        self.root=Path(root or ROOT/'app/catalog').resolve()
        self.catalog={}
        self.discover()
    def _metadata(self,meta,path):
        if not isinstance(meta,dict): raise ValueError('Invalid Skill frontmatter')
        name=meta.get('name')
        if not isinstance(name,str) or name!=path.parent.name or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',name) or name not in ALLOW: raise ValueError('Invalid Skill')
        if not isinstance(meta.get('description'),str) or not meta['description'].strip(): raise ValueError('Missing description')
        details=meta.get('metadata',{})
        if not isinstance(details,dict) or not isinstance(details.get('version','1.0.0'),str): raise ValueError('Skill version must be a string')
        return dict(name=name,description=meta['description'],version=details.get('version','1.0.0'),enabled=True)
    def discover(self):
        catalog={}
        for path in sorted(self.root.glob('*/SKILL.md')):
            if not path.resolve().is_relative_to(self.root): raise ValueError('Skill path escape')
            # Read only frontmatter, leaving the body out of initial context.
            with path.open(encoding='utf-8') as f:
                if f.readline().strip()!='---': raise ValueError('Missing frontmatter')
                lines=[]
                for line in f:
                    if line.strip()=='---': break
                    lines.append(line)
                else: raise ValueError('Unclosed frontmatter')
            meta=yaml.safe_load(''.join(lines))
            metadata=self._metadata(meta,path)
            catalog[metadata['name']]=metadata
        self.catalog=catalog
        return list(catalog.values())
    def load(self,name):
        start=time.perf_counter()
        if name not in self.catalog: raise ValueError('Unknown Skill')
        path=self.root/name/'SKILL.md'
        if not path.resolve().is_relative_to(self.root): raise ValueError('Skill path escape')
        text=path.read_text(encoding='utf-8'); lines=text.splitlines(keepends=True)
        if not lines or lines[0].strip()!='---': raise ValueError('Missing frontmatter')
        end=next((i for i in range(1,len(lines)) if lines[i].strip()=='---'),None)
        if end is None: raise ValueError('Unclosed frontmatter')
        metadata=self._metadata(yaml.safe_load(''.join(lines[1:end])),path)
        self.catalog[name]=metadata
        body=''.join(lines[end+1:]).strip()
        return dict(**metadata,instructions=body,hash=hashlib.sha256(text.encode()).hexdigest(),estimated_tokens=len(body),duration_ms=(time.perf_counter()-start)*1000)
    def select(self,query,context=None):
        context=context or {}
        intent=context.get('intent')
        name=INTENT_SKILL.get(intent)
        if intent is None:
            for words,candidate in [(('投诉','人工'),'complaint-handling'),(('退款','退货'),'refund-processing'),(('订单','物流'),'order-tracking'),(('耳机','产品','商品','保修','颜色','价格','库存'),'product-consultation')]:
                if any(word in query for word in words): name=candidate; break
        if name not in self.catalog: name=None
        return SkillSelection(skill_name=name,confidence=1.0 if intent is not None else .8 if name else 0,reason='intent='+str(intent) if intent is not None else 'catalog keyword baseline')
    def allowed_tools(self,name): return ALLOW.get(name,set()).copy()

INTENT_SKILL={'order_query':'order-tracking','shipping_query':'order-tracking','refund':'refund-processing','product_consultation':'product-consultation','faq':'product-consultation','complaint':'complaint-handling'}
