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
    license='CC BY-NC-SA 4.0',
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

        # 'dill>=0.2.7.1',
        # 'scikit-image>=0.13.1',
        # 'numba>=0.35.0'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.12'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.12'],
        'keras': ['keras>=2.0.0'],
        'torch': ['torch>=1.0.0'],
        'hmmlearn': ['hmmlearn==0.2.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
)
