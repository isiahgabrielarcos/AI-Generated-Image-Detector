import ssl, os, json, urllib.request, base64, urllib.parse

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

query = urllib.parse.urlencode({'search': 'dfdc images frames', 'page': 1, 'pageSize': 20})
url = f'https://www.kaggle.com/api/v1/datasets/list?{query}'
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx) as r:
        results = json.loads(r.read())
        for ds in results:
            mb = (ds.get('totalBytes') or 0) / 1024 / 1024
            print(f"{ds.get('ownerRef')}/{ds.get('currentDatasetVersionNumber')} | {ds.get('ref')} | {mb:.0f} MB")
            print(f"  title: {ds.get('title')}")
except Exception as e:
    print(f"ERROR: {e}")
