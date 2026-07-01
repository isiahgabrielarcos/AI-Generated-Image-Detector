import py7zr, itertools
archive = r'D:\Dataset\Progan\progan_train.7z.001'
print('Archive:', archive)
with py7zr.SevenZipFile(archive,'r') as z:
    names = list(itertools.islice(z.getnames(),50))
    print('First 50 entries:')
    for name in names:
        print('  ', name)
    fake_count = 0
    real_count = 0
    top_dirs = set()
    for name in z.getnames():
        n = name.lower()
        if '/fake/' in n or n.startswith('fake') or '\\fake\\' in n:
            fake_count += 1
        if '/real/' in n or n.startswith('real') or '\\real\\' in n:
            real_count += 1
        if '/' in n:
            top_dirs.add(n.split('/')[0])
    print('Top-level dirs:', sorted(top_dirs)[:20])
    print('Fake entries:', fake_count)
    print('Real entries:', real_count)
