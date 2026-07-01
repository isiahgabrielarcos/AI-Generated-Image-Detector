"""Check chandlertimm/forensynths file list and try small alternatives."""
import ssl, os, json, base64, urllib.request, urllib.parse
from pathlib import Path

creds_path = Path.home() / ".kaggle" / "kaggle.json"
with open(creds_path) as f:
    creds = json.load(f)
ctx = ssl._create_unverified_context()
auth = "Basic " + base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
headers = {'Authorization': auth}

# Try various file-listing endpoints for forensynths
slug = 'chandlertimm/forensynths'
print(f"=== Trying to list files in {slug} ===")
for url_template in [
    f'https://www.kaggle.com/api/v1/datasets/{slug}/versions/1/files?pageSize=50',
    f'https://www.kaggle.com/api/v1/datasets/{slug}/files?pageSize=50',
]:
    try:
        req = urllib.request.Request(url_template, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            data = json.loads(r.read())
            files = data.get('files') or (data if isinstance(data, list) else [])
            if files:
                print(f"  URL {url_template} returned {len(files)} files:")
                for fi in files[:30]:
                    name = fi.get('name') or fi.get('path', '?')
                    mb = (fi.get('totalBytes') or 0) / 1024 / 1024
                    print(f"    {name}  ({mb:.0f} MB)")
            else:
                print(f"  URL returned empty: {str(data)[:200]}")
    except Exception as e:
        print(f"  URL failed: {e}")

# Check the two best small alternatives
print("\n\n=== Small alternative candidates ===")
for slug2, label in [
    ('shanmuk4622/real-and-fake-ai-generated-512px-dataset', 'Real+Fake AI 512px'),
    ('donaasu/mukhbir-project-dataset', 'GAN/Real Mukhbir'),
    ('ciplab/real-and-fake-face-detection', 'CIPLAB Real/Fake'),
    ('virajinduruwa/real-vs-fakeai-image-dataset', 'Real vs FakeAI'),
]:
    print(f"\n{label} ({slug2}):")
    for ver in ['1', '2']:
        url = f'https://www.kaggle.com/api/v1/datasets/{slug2}/versions/{ver}/files?pageSize=30'
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
                data = json.loads(r.read())
                files = data.get('files') or (data if isinstance(data, list) else [])
                if files:
                    for fi in files[:15]:
                        name = fi.get('name') or fi.get('path', '?')
                        mb = (fi.get('totalBytes') or 0) / 1024 / 1024
                        print(f"  {name}  ({mb:.0f} MB)")
                    break
        except:
            pass
