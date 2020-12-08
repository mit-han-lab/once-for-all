#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

# readme = open('README.md').read()
readme = '''
# Once for All: Train One Network and Specialize it for Efficient Deployment [[arXiv]](https://arxiv.org/abs/1908.09791) [[Slides]](https://file.lzhu.me/projects/OnceForAll/OFA%20Slides.pdf) [[Video]](https://youtu.be/a_OeT8MXzWI)
```BibTex
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/pdf/1908.09791.pdf}
}
```

## News
- Fisrt place in the 4th [Low-Power Computer Vision Challenge](https://lpcv.ai/competitions/2019), both classification and detection track.
- First place in the 3rd [Low-Power Computer Vision Challenge](https://lpcv.ai/competitions/2019), DSP track at ICCV’19 using the Once-for-all Network.

## Check our [GitHub](https://github.com/mit-han-lab/once-for-all) for more details.
'''
VERSION = '0.1.0'

requirements = [
    'torch',
]

# import subprocess
# commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True).decode('UTF-8').rstrip()
# VERSION += "_" + str(int(commit_hash, 16))[:8]
VERSION += "_" + datetime.datetime.now().strftime('%Y%m%d%H%M')
# print(VERSION)

setup(
    # Metadata
    name='ofa',
    version=VERSION,
    author='MTI HAN LAB ',
    author_email='hanlab.eecs+github@gmail.com',
    url='https://github.com/mit-han-lab/once-for-all',
    description='Once for All: Train One Network and Specialize it for Efficient Deployment.',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
