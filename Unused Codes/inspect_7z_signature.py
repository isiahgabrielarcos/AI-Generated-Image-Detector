from pathlib import Path
path = Path(r'D:\Dataset\Progan\progan_train.7z.001')
with path.open('rb') as f:
    head = f.read(64)
print(head)
print(head.hex())
