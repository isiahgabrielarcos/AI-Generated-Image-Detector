import ssl, os, json, urllib.request, base64

creds_path = os.path.join(os.environ['USERPROFILE'], '.kaggle', 'kaggle.json')
with open(creds_path) as f:
    creds = json.load(f)
username = creds['username']
key = creds['key']

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

token = base64.b64encode(f'{username}:{key}'.encode()).decode()
headers = {'Authorization': f'Basic {token}'}

for slug in ['ashifurrahman34/dfdc-dataset', 'fakecatcherai/dfdc-dataset']:
    url = f'https://www.kaggle.com/api/v1/datasets/{slug}'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx) as r:
            data = json.loads(r.read())
            mb = (data.get('totalBytes') or 0) / 1024 / 1024
            print(f"[{slug}]")
            print(f"  title: {data.get('title')}")
            print(f"  size:  {mb:.0f} MB")
    except Exception as e:
        print(f"[{slug}] ERROR: {e}")

    # Also list files
    url2 = f'https://www.kaggle.com/api/v1/datasets/{slug}/versions/1/files'
    req2 = urllib.request.Request(url2, headers=headers)
    try:
        with urllib.request.urlopen(req2, context=ctx) as r:
            files = json.loads(r.read())
            for fi in (files.get('files') or [])[:15]:
                print(f"  file: {fi.get('name')} ({(fi.get('totalBytes') or 0)/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"  files ERROR: {e}")
