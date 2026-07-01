import ssl, os, json, base64, urllib.request, urllib.parse
from pathlib import Path

creds_path = Path.home() / ".kaggle" / "kaggle.json"
with open(creds_path) as f:
    creds = json.load(f)

ctx = ssl._create_unverified_context()
auth = "Basic " + base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()

def search(q):
    qs = urllib.parse.urlencode({'search': q, 'pageSize': 10})
    req = urllib.request.Request(
        f'https://www.kaggle.com/api/v1/datasets/list?{qs}',
        headers={'Authorization': auth}
    )
    with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
        return json.loads(r.read())

for query in ['dfdc extracted face images real fake', 'celebdf real face images extracted', 'real fake face images dataset deepfake']:
    print(f"\n--- Search: {query} ---")
    try:
        results = search(query)
        for ds in results[:6]:
            mb = (ds.get('totalBytes') or 0) / 1024 / 1024
            print(f"  {ds.get('ref')}  |  {mb:.0f} MB  |  {ds.get('title')}")
    except Exception as e:
        print(f"  ERROR: {e}")
