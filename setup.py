""" SeismiPro is a library for seismic data processing. """

from setuptools import setup, find_packages
import re

with open('seismicpro/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='SeismicPro',
    packages=find_packages(exclude=['tutorials', 'docker_containers', 'datasets', 'models']),
    version=version,
    url='https://github.com/gazprom-neft/SeismicPro',
    license='Apache License 2.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='A framework for seismic data processing',
    long_description='',
    zip_safe=False,
    platforms='any',
    include_package_data=True,
    package_data={'': ['datasets/demo_data/*.sgy']},
    install_requires=[
        'matplotlib>=3.3.1',
        'numba>=0.52.0',
        'numpy>=1.19.5',
        'pandas>=1.1.5',
        'scikit-learn>=0.23.2',
        'scipy>=1.5.2',
        'segyio>=1.9.5',
        'tdigest>=0.5.2.2',
        'tqdm>=4.56.0',
        'batchflow @ git+https://github.com/analysiscenter/batchflow.git@9823f369#egg=batchflow',
    ],
    extras_require={
        'torch': ['torch>=1.7'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
)
