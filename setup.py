# from skbuild import setup
from setuptools import setup
# https://scikit-build.readthedocs.io/en/latest/usage.html
setup(
    name="toyml",
    version="1.0",
    packages=['toyml', 'native', 'scripts'],
    cmake_source_dir='.',
)