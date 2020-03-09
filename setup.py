from setuptools import setup, Command
import os
import sys

setup(name='pandapy',
      version='0.0.1',
      description='PVance - Advanced Pandas Utilities, Functions and Snippets',
      url='https://github.com/firmai/pvance',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['pvance'],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'sklearn',
          'ipython'

      ],
      zip_safe=False)
