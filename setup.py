from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pyMultiobjective',
    version='1.5.0',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyMultiobjective',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'pygmo',
        'scipy'
    ],
    description='A python library for Multiobjective Objectives Optimization Algorithms or Many Objectives Optimization Algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
