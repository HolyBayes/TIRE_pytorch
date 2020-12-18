__author__ = 'Artem Ryzhikov'

from setuptools import setup

setup(
    name="tire_pytorch",
    version='0.1.0',
    description="PyTorch implementation of TIRE for deep change point and anomaly detection in timeseries",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/HolyBayes/TIRE_pytorch',
    author='Artem Ryzhikov',

    packages=['TIRE'],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 ',
    ],
    keywords='pytorch, deep learning, timeseries, anomaly detection, change point detection',
    install_requires=[
        'torch>=1.6.0',
        'scipy>=1.5.2',
        'numpy>=1.18.5'
    ]
)
