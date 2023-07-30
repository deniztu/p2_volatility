# -*- coding: utf-8 -*-
"""
First set working directory to files to be changed
"""





import os 

d = os.getcwd()
old = os.listdir(d)

new = [i.replace('_check', '') for i in old]

[os.rename(os.path.join(d, old[i]), os.path.join(d, new[i])) for i, j in enumerate(old) if '_check' in j]
