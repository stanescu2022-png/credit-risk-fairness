from pathlib import Path
import ast, csv, hashlib, json, os
ROOT = Path.cwd()
OUT = ROOT / 'docs/context'
rows = []
for folder, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in {'.git', '.venv', '__pycache__', 'node_modules', 'context'}]
    for name in sorted(files):
        p = Path(folder) / name
        b = p.read_bytes()
        r = {'path': p.relative_to(ROOT).as_posix(), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
        if p.suffix == '.csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                r['columns'] = next(reader, [])
                r['rows'] = sum(1 for _ in reader)
        elif p.suffix == '.ipynb':
            nb = json.loads(b)
            r['cells'] = [{'type': c['cell_type'], 'source': ''.join(c['source']), 'execution_count': c.get('execution_count'), 'output_types': [o['output_type'] for o in c.get('outputs', [])]} for c in nb['cells']]
        elif p.suffix == '.py':
            r['definitions'] = [n.name for n in ast.parse(b).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        rows.append(r)
(OUT / 'repository_inventory.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
sources = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((OUT / 'sources').iterdir()) if p.is_file()]
(OUT / 'source_documents.json').write_text(json.dumps(sources, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
checks = []
for path in ROOT.glob('outputs/tables/*/run_manifest.json'):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for category in ('protected_artifact_hashes', 'locked_input_hashes', 'source_hashes', 'output_hashes'):
        for name, expected in manifest.get(category, {}).items():
            p = ROOT / name
            actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            checks.append({'manifest': path.relative_to(ROOT).as_posix(), 'category': category, 'path': name, 'matches': actual == expected})
(OUT / 'artifact_hash_check.json').write_text(json.dumps(checks, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'artifacts': len(rows), 'csvs': sum('columns' in r for r in rows), 'notebooks': sum('cells' in r for r in rows), 'source_files': len(sources), 'hash_checks': len(checks), 'hash_mismatches': [r for r in checks if not r['matches']]}))