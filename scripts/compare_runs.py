import argparse,json
from pathlib import Path
from app.evaluation import compare
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--candidate',required=True); a=p.parse_args()
    def load(r): return json.loads((Path('data/eval_runs')/r/'report.json').read_text(encoding='utf-8'))
    r=compare(load(a.baseline),load(a.candidate)); print(json.dumps(r,ensure_ascii=False,indent=2))
