"""Run the isolated fault matrix using the existing integration fixtures."""
import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import ROOT
from app.db import now, uid
from app.rag import atomic_json


def run_faults(matrix_path, output_root):
    source = Path(matrix_path).read_bytes()
    matrix = json.loads(source)
    entries = matrix['scenarios']
    if not entries or len({row['id'] for row in entries}) != len(entries):
        raise ValueError('Fault scenario IDs must be unique and nonempty')
    nodes = {}
    for entry in entries:
        selector = entry['pytest']
        file, function = selector.split('::', 1)
        if not file.startswith('tests/') or not (ROOT / file).resolve().is_relative_to(ROOT / 'tests'):
            raise ValueError('Fault tests must be local repository tests')
        nodes[selector] = (file.removesuffix('.py').replace('/', '.'), function)
    run_id = uid('eval-faults-')
    output = (Path(output_root) / run_id).resolve()
    output.mkdir(parents=True)
    xml = output / 'pytest.xml'
    log = output / 'pytest.log'
    command = [sys.executable, '-m', 'pytest', '-q', '-m', 'not real and not models', '--junitxml', str(xml), *nodes]
    with log.open('wb') as log_file:
        completed = subprocess.run(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    tests = list(ET.parse(xml).iter('testcase')) if xml.is_file() else []
    rows = []
    for entry in entries:
        module, function = nodes[entry['pytest']]
        matching = [test for test in tests if test.get('classname') == module and
                    (test.get('name') == function or test.get('name', '').startswith(function+'['))]
        failed = any(test.find('failure') is not None or test.find('error') is not None for test in matching)
        skipped = any(test.find('skipped') is not None for test in matching)
        rows.append({**entry, 'test_names': [test.get('name') for test in matching],
                     'status': 'FAIL' if failed else 'NOT_EVALUATED' if not matching or skipped else 'PASS'})
    status = 'BLOCKED' if completed.returncode not in (0, 1) or any(r['status']=='NOT_EVALUATED' for r in rows) else 'FAIL' if completed.returncode else 'PASS'
    report = {'run_id':run_id, 'created_at':now(), 'matrix_version':matrix['version'], 'matrix_sha256':hashlib.sha256(source).hexdigest(),
              'gate':status, 'passed':sum(r['status']=='PASS' for r in rows), 'total':len(rows), 'cases':rows,
              'pytest_exit':completed.returncode, 'provenance':{'simulated_commerce':True,'external_model':False,'fault_injection':True},
              'command':command, 'log':str(log), 'junit':str(xml)}
    atomic_json(output / 'report.json', report)
    return report, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', default='datasets/fault_injection/matrix_v1.json')
    parser.add_argument('--output-root', default='data/eval_runs')
    args = parser.parse_args()
    report, output = run_faults(args.matrix, args.output_root)
    print(f"{report['gate']}: {report['passed']}/{report['total']} scenarios — {output}")
    if report['gate'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
