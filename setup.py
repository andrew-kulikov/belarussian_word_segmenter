from os.path import join, dirname
from setuptools import setup, find_packages

setup(
    name='Bel Segment',
    version='1.0',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    install_requires=[
        'numpy==1.14.5',
    ],
)
