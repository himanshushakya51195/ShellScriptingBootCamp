#!//usr/bin/python3

import os
import sys

folder_name = ['folder1', 'folder2', 'folder3']

for name in folder_name:
	os.system("mkdir -p "+name)
	print(name)




