import zipfile, itertools
zip_path = r'D:\Dataset\Progan\progan_train.zip'
with zipfile.ZipFile(zip_path,'r') as zf:
    names = [info.filename for info in itertools.islice(zf.infolist(),50)]
    print('First 50 entries:')
    for name in names:
        print('  ', name)
    fake_count = 0
    real_count = 0
    top_dirs = set()
    for info in zf.infolist():
        n = info.filename.lower()
        if '/fake/' in n or n.startswith('fake') or '\\fake\\' in n:
            fake_count += 1
        if '/real/' in n or n.startswith('real') or '\\real\\' in n:
            real_count += 1
        if '/' in n:
            top_dirs.add(n.split('/')[0])
    print('Top-level dirs:', sorted(top_dirs)[:20])
    print('Fake entries:', fake_count)
    print('Real entries:', real_count)
