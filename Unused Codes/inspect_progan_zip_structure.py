import zipfile
from pathlib import Path
from collections import Counter
zip_path = Path(r'D:\Dataset\Progan\progan_train.zip')
with zipfile.ZipFile(zip_path, 'r') as zf:
    patterns = Counter()
    for name in zf.namelist():
        if name.endswith('/'):
            continue
        parts = Path(name).parts
        if len(parts) >= 2:
            patterns['/'.join(parts[:2])] += 1
        else:
            patterns[name] += 1
    for item, count in patterns.most_common(50):
        print(f'{item}: {count}')
