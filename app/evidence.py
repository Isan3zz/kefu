"""Deterministic evidence audits independent of model replies and expected labels."""
import json
from sqlalchemy import select
from app.db import documents,document_chunks,image_assets,memories,visual_text_records
from app.rag import visible

def audit_tool_reads(events,actor,owned_order_ids):
    violations=[]
    for event in events:
        result=event.get('result',{})
        if not result.get('ok'): continue
        name=event['name']; data=result.get('data'); bad=False
        if name in ('get_order','get_orders'):
            rows=data if isinstance(data,list) else [data]
            bad=any(not isinstance(row,dict) or row.get('id') not in owned_order_ids or row.get('tenant_id',actor.tenant_id)!=actor.tenant_id or row.get('user_id',actor.user_id)!=actor.user_id for row in rows)
            if name == 'get_order' and isinstance(data, dict):
                bad = bad or any(row.get('order_id') not in owned_order_ids
                                 for field in ('items', 'shipments') if isinstance(data.get(field), list)
                                 for row in data[field] if isinstance(row, dict) and row.get('order_id'))
        elif name=='track_shipment':
            bad=any(row.get('order_id') not in owned_order_ids for row in data or [])
        elif name=='check_refund_eligibility': bad=event.get('args',{}).get('order_id') not in owned_order_ids
        elif name=='search_products': bad=any(row.get('tenant_id')!=actor.tenant_id for row in data or [])
        if bad: violations.append({'event_id':event['event_id'],'tool':name,'category':'AUTHORIZATION_VIOLATION'})
    return violations

def audit_citations(db,citations,actor):
    counts={'stale_knowledge_count':0,'invalid_citation_count':0,'unauthorized_citation_count':0}
    if not citations: return counts
    with db.tx() as connection:
        rows=connection.execute(select(document_chunks.c.chunk_id,document_chunks.c.metadata_json,documents).join(documents,documents.c.id==document_chunks.c.document_id).where(document_chunks.c.chunk_id.in_([row['chunk_id'] for row in citations]))).mappings()
        known={row['chunk_id']:dict(row) for row in rows}
    for citation in citations:
        row=known.get(citation['chunk_id'])
        if row is None:
            counts['invalid_citation_count']+=1
            continue
        if row['tenant_id'] not in (actor.tenant_id,'public') or row['visibility'] not in ('public','customer'):
            counts['unauthorized_citation_count']+=1
        elif not visible(row,actor.tenant_id) or str(citation['version'])!=row['version'] or json.loads(row['metadata_json'])['version']!=row['version']:
            counts['stale_knowledge_count']+=1
        elif any(key in citation and citation[key] != row[column] for key,column in (
                ('doc_id','id'),('document_title','title'),('source','source_path'))):
            counts['invalid_citation_count']+=1
    return counts


def audit_retrieval_scope(db, knowledge_db, events, actor):
    """Find foreign results already returned to the execution boundary."""
    violations = []
    for event in events:
        if event['type'] in ('memory.search_completed', 'memory.context_injected'):
            ids = event.get('memory_ids', [])
            if ids:
                with db.tx() as connection:
                    bad = connection.execute(select(memories.c.id).where(memories.c.id.in_(ids),
                        (memories.c.tenant_id != actor.tenant_id) | (memories.c.user_id != actor.user_id))).first()
                if bad:
                    violations.append({'event_id':event['event_id'], 'tool':'memory', 'category':'AUTHORIZATION_VIOLATION'})
            continue
        if event['type'] != 'tool' or not event.get('result', {}).get('ok'):
            continue
        tool, data = event['name'], event['result']['data']
        if tool == 'search_knowledge':
            bad = audit_citations(knowledge_db, data, actor)['unauthorized_citation_count']
        elif tool in ('search_products_by_image', 'search_products_by_image_and_text', 'search_images_by_text', 'search_text_by_image'):
            table = visual_text_records if tool == 'search_text_by_image' else image_assets
            key = 'visual_text_id' if table is visual_text_records else 'image_id'
            with db.tx() as connection:
                bad = connection.execute(select(table.c[key]).where(table.c[key].in_([row[key] for row in data]),
                                                                  table.c.tenant_id != actor.tenant_id)).first()
        else:
            continue
        if bad:
            violations.append({'event_id':event['event_id'], 'tool':tool, 'category':'AUTHORIZATION_VIOLATION'})
    return violations
