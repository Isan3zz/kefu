"""HTTP walkthrough against running API + MCP. Does not require real LLM."""
import argparse,json
import httpx
def main():
    p=argparse.ArgumentParser(); p.add_argument('--url',default='http://127.0.0.1:8000'); p.add_argument('--mode',choices=['deterministic','real'],default='deterministic'); p.add_argument('--order',default='O1001'); a=p.parse_args()
    with httpx.Client(base_url=a.url,timeout=120) as c:
        def call(method,path,**kw):
            r=c.request(method,path,**kw); r.raise_for_status(); return r.json()['data']
        token=call('POST','/api/v1/auth/demo-login',json={'demo_user_key':'user_a'})['token']; c.headers['Authorization']='Bearer '+token
        conv=call('POST','/api/v1/conversations')['id']
        def chat(text):
            r=call('POST','/api/v1/chat',json={'conversation_id':conv,'message':text,'mode':a.mode}); print(json.dumps({k:v for k,v in r.items() if k!='pending_confirmation'},ensure_ascii=False,indent=2)); return r
        knowledge=chat('耳机支持七天无理由退款吗')
        assert knowledge['citations'], 'Build the real local knowledge index before the full walkthrough'
        chat('我的订单'); chat('第二个订单物流到哪了')
        pending=chat('帮我退订单 '+a.order)['pending_confirmation']
        if not pending: raise RuntimeError('Order is not refundable; use a fresh database or another eligible --order')
        product=chat('你们有蓝色耳机吗')
        live_products=[row for action in product['tool_actions'] if action['name']=='search_products' and action['result']['ok'] for row in action['result']['data']]
        # Availability may be answered by the live catalog instead of static RAG evidence.
        assert product['citations'] or any('蓝色' in json.loads(row.get('attributes_json','{}')).get('colors',[]) for row in live_products)
        workflow=call('GET','/api/v1/workflows/'+pending['workflow_id'])
        assert workflow['status']=='WAITING_CONFIRMATION'
        endpoint='/api/v1/workflows/'+pending['workflow_id']+'/confirm'; body={'confirm_nonce':pending['confirm_nonce']}
        first=call('POST',endpoint,json=body); second=call('POST',endpoint,json=body)
        assert first['refund_request_id']==second['refund_request_id']
        print('confirmed:',first,'replay:',second)
        status=chat('订单 '+a.order+' 退款进度如何')
        assert first['refund_request_id'] in status['answer']
        chat('我需要转人工处理其他问题')
        print('HTTP_WALKTHROUGH=PASS; real local BGE retrieval; explicit '+a.mode+' routing')
if __name__=='__main__': main()
