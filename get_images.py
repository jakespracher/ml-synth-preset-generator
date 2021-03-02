import os
import shutil


# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk("."):
    for file in files:
        print(root)
        print(files)
        if 'final' in file:
            id = ''.join([s for s in root if s.isdigit()])
            shutil.copy(os.path.join(root, file), f'/tmp/plot_images/{id}_{file}')

