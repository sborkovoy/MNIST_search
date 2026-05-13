import tb
import os

directory = '/home/guest/BespyatyyIV/project/images'
jpgs  = sorted(f for f in os.listdir(directory) if f.endswith('.jpeg'))

for j in jpgs:
    with open(os.path.join(directory, j), 'rb') as jf:
        tb.write(jf.read())