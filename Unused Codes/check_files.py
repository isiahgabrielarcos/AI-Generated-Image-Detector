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

datasets_to_check = [
    'pranabkc/deepfake-with-cropped-faces-from-video',
    'itamargr/dfdc-faces-of-the-train-sample',
]

for slug in datasets_to_check:
    print(f"\n=== {slug} ===")
    # Get dataset info
    url = f'https://www.kaggle.com/api/v1/datasets/{slug}'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx) as r:
            data = json.loads(r.read())
            mb = (data.get('totalBytes') or 0) / 1024 / 1024
            print(f"Title: {data.get('title')}  |  {mb:.0f} MB")
            print(f"Description: {str(data.get('description',''))[:300]}")
    except Exception as e:
        print(f"INFO error: {e}")

    # List files
    url2 = f'https://www.kaggle.com/api/v1/datasets/{slug}/versions/1/files?pageSize=30'
    req2 = urllib.request.Request(url2, headers=headers)
    try:
        with urllib.request.urlopen(req2, context=ctx) as r:
            result = json.loads(r.read())
            files = result.get('files') or result if isinstance(result, list) else []
            print(f"Files ({len(files)}):")
            for fi in files[:20]:
                name = fi.get('name') or fi.get('path') or str(fi)
                size_mb = (fi.get('totalBytes') or 0) / 1024 / 1024
                print(f"  {name}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FILES error: {e}")
