import ssl, os, json, base64, urllib.request, urllib.parse
from pathlib import Path

creds_path = Path.home() / ".kaggle" / "kaggle.json"
with open(creds_path) as f:
    creds = json.load(f)

ctx = ssl._create_unverified_context()
auth = "Basic " + base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
headers = {'Authorization': auth}

def search(q, n=8):
    qs = urllib.parse.urlencode({'search': q, 'pageSize': n})
    req = urllib.request.Request(f'https://www.kaggle.com/api/v1/datasets/list?{qs}', headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        return json.loads(r.read())

# ── Check GenImage options structure ─────────────────────────────────
print("=== GenImage candidates ===")
for slug in ['cartografia/unbiased-tiny-genimage', 'renhuang8/genimage-subset-detection']:
    print(f"\n{slug}:")
    # Try listing files via old REST API
    for ver in ['1', '2']:
        url = f'https://www.kaggle.com/api/v1/datasets/{slug}/versions/{ver}/files?pageSize=30'
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
                data = json.loads(r.read())
                files = data.get('files') or (data if isinstance(data, list) else [])
                if files:
                    for fi in files[:20]:
                        name = fi.get('name') or fi.get('path', '?')
                        mb = (fi.get('totalBytes') or 0) / 1024 / 1024
                        print(f"  v{ver}: {name}  ({mb:.0f} MB)")
                    break
        except Exception as e:
            pass

# ── ForenSynths: search for smaller subsets ───────────────────────────
print("\n\n=== ForenSynths smaller alternatives ===")
for q in ['forensynths subset small sample', 'progan stylegan real fake sample images small', 'gan generated images forensic subset 5000']:
    print(f"\nSearch: {q}")
    for ds in search(q, 6):
        gb = (ds.get('totalBytes') or 0) / 1024**3
        print(f"  {ds.get('ref'):<55} {gb:.1f} GB  |  {ds.get('title')}")
